// Originally made by Sam Turner and Finlay Sanders, 2025.
// Included in pufferlib under the original project's MIT license.
// https://github.com/tensaur/drone

#pragma once

#include <limits.h>
#include <math.h>
#include <stdbool.h>
#include <stdlib.h>

#include "dronelib.h"
#include "tasks.h"

#define HORIZON 1024

typedef struct Client Client;
typedef struct DroneEnv DroneEnv;

struct DroneEnv {
    Log log;
    float* observations;
    float* actions;
    float* rewards;
    float* terminals;
    int num_agents;
    unsigned int rng;

    int tick;
    DroneTask task;
    Drone* agents;

    int max_rings;
    Target* ring_buffer;

    Client* client;

    // reward scaling
    float alpha_dist;
    float alpha_hover;
    float alpha_shaping;
    float alpha_omega;
    float alpha_omega_xy;
    float alpha_omega_z;
    float alpha_omega_z_sq;
    float alpha_omega_z_mult;

    // hover task parameters
    float hover_target_dist;
    float oob_radius;
    float hover_dist;
    float hover_omega;
    float hover_vel;
    float domain_randomization;
    float dr_mass;
    float dr_inertia;
    float dr_k_thrust;
    float dr_linear_drag;
    float dr_yaw_drag;
    float dr_motor_lag;
    float dr_com_xy;
    float dr_com_z;
    float dr_authority_gated;
    float dr_usable_t2w_min;
    float dr_usable_t2w_max;
    float dr_mass_min;
    float dr_mass_max;
    float dr_inertia_min;
    float dr_inertia_max;
    float dr_motor_thrust_min;
    float dr_motor_thrust_max;
    float dr_motor_tau_min;
    float dr_motor_tau_max;
    float dr_yaw_torque_min;
    float dr_yaw_torque_max;
    float dr_linear_drag_min;
    float dr_linear_drag_max;
    float dr_angular_damping_min;
    float dr_angular_damping_max;
    float dr_profile_mix;
    float action_scale;
    int action_mode;
    float normalized_thrust_min;
    float normalized_thrust_max;
    float reset_pos_scale;
    float reset_yaw_range;
    float reset_vel_max;
    float action_latency;
    float sensor_noise;
};

void init(DroneEnv* env) {
    env->agents = (Drone*)calloc(env->num_agents, sizeof(Drone));
    env->ring_buffer = (Target*)calloc(env->max_rings, sizeof(Target));

    for (int i = 0; i < env->num_agents; i++) {
        env->agents[i].target = (Target*)calloc(1, sizeof(Target));
        env->agents[i].buffer_idx = 0;
    }

    env->log = (Log){0};
    env->tick = 0;
}

static inline DomainRandomization env_domain_randomization(DroneEnv* env);

static inline void record_step_metrics(Drone* agent, float raw_actions[4], float r_dist,
                                        float r_hover, float r_shaping, float r_omega,
                                        float r_omega_xy, float r_omega_z, float r_terminal) {
    float action_abs_sum = 0.0f;
    float action_clipped_abs_sum = 0.0f;
    float action_max_abs = 0.0f;
    float action_saturation_count = 0.0f;
    float motor_clip_low_count = 0.0f;
    float motor_clip_high_count = 0.0f;

    for (int i = 0; i < 4; i++) {
        float abs_action = fabsf(raw_actions[i]);
        action_abs_sum += abs_action;
        if (abs_action > action_max_abs) action_max_abs = abs_action;
        if (abs_action >= 0.99f) action_saturation_count += 1.0f;

        float env_clipped = clampf(raw_actions[i], -1.0f, 1.0f);
        action_clipped_abs_sum += fabsf(env_clipped);
        if (agent->params.action_mode == M4D_ACTION_NORMALIZED_THRUST) {
            float motor_cmd = normalized_thrust_command(&agent->params, env_clipped);
            float lo = clampf(agent->params.normalized_thrust_min, 0.0f, 1.0f);
            float hi = clampf(agent->params.normalized_thrust_max, lo, 1.0f);
            if (motor_cmd <= lo + 1e-5f) motor_clip_low_count += 1.0f;
            if (motor_cmd >= hi - 1e-5f) motor_clip_high_count += 1.0f;
        } else {
            float motor_action = clampf(env_clipped * agent->params.action_scale, -1.0f, 1.0f);
            if (motor_action <= -0.99f) motor_clip_low_count += 1.0f;
            if (motor_action >= 0.99f) motor_clip_high_count += 1.0f;
        }

        agent->rpm_sum[i] += agent->state.rpms[i];
    }

    agent->action_abs_sum += action_abs_sum / 4.0f;
    agent->action_clipped_abs_sum += action_clipped_abs_sum / 4.0f;
    if (action_max_abs > agent->action_max_abs) agent->action_max_abs = action_max_abs;
    agent->action_saturation_count += action_saturation_count / 4.0f;
    agent->motor_clip_low_count += motor_clip_low_count / 4.0f;
    agent->motor_clip_high_count += motor_clip_high_count / 4.0f;
    agent->instrumentation_steps += 1.0f;

    agent->r_dist_sum += r_dist;
    agent->r_hover_sum += r_hover;
    agent->r_shaping_sum += r_shaping;
    agent->r_omega_sum += r_omega;
    agent->r_omega_xy_sum += r_omega_xy;
    agent->r_omega_z_sum += r_omega_z;
    agent->r_terminal_sum += r_terminal;
}

void add_log(DroneEnv* env, int idx, bool oob, bool timeout) {
    Drone* agent = &env->agents[idx];
    float steps = fmaxf(agent->instrumentation_steps, 1.0f);

    env->log.episode_return += agent->episode_return;
    env->log.episode_length += agent->episode_length;
    env->log.collisions += agent->collisions;

    if (oob) env->log.oob += 1.0f;
    if (timeout) env->log.timeout += 1.0f;

    env->log.score += agent->hover_score;
    env->log.perf += agent->hover_ema;
    env->log.rings_passed += agent->rings_passed;
    env->log.ema_dist += agent->ema_dist;
    env->log.ema_vel += agent->ema_vel;
    env->log.ema_omega += agent->ema_omega;
    env->log.ema_omega_x += agent->ema_omega_x;
    env->log.ema_omega_y += agent->ema_omega_y;
    env->log.ema_omega_z += agent->ema_omega_z;
    env->log.mean_abs_action += agent->action_abs_sum / steps;
    env->log.mean_abs_action_clipped += agent->action_clipped_abs_sum / steps;
    env->log.max_abs_action += agent->action_max_abs;
    env->log.action_saturation_frac += agent->action_saturation_count / steps;
    env->log.motor_clip_low_frac += agent->motor_clip_low_count / steps;
    env->log.motor_clip_high_frac += agent->motor_clip_high_count / steps;

    float trim[4];
    float trim_rpm_sum = 0.0f;
    float trim_rpm_max = 0.0f;
    hover_trim_thrusts(&agent->params, trim);
    for (int m = 0; m < 4; m++) {
        float rpm = thrust_to_rpm_i(&agent->params, m, trim[m]);
        trim_rpm_sum += rpm;
        if (rpm > trim_rpm_max) trim_rpm_max = rpm;
    }
    env->log.hover_trim_rpm_mean += trim_rpm_sum / 4.0f;
    env->log.hover_trim_rpm_max += trim_rpm_max;
    env->log.hover_trim_rpm_frac_of_max += trim_rpm_max / fmaxf(agent->params.max_rpm, 1.0f);

    env->log.mean_rpm_FL += agent->rpm_sum[0] / steps;
    env->log.mean_rpm_FR += agent->rpm_sum[1] / steps;
    env->log.mean_rpm_RL += agent->rpm_sum[2] / steps;
    env->log.mean_rpm_RR += agent->rpm_sum[3] / steps;
    env->log.r_dist += agent->r_dist_sum;
    env->log.r_hover += agent->r_hover_sum;
    env->log.r_shaping += agent->r_shaping_sum;
    env->log.r_omega += agent->r_omega_sum;
    env->log.r_omega_xy += agent->r_omega_xy_sum;
    env->log.r_omega_z += agent->r_omega_z_sum;
    env->log.r_terminal += agent->r_terminal_sum;
    env->log.mass_mult_mean += agent->params.mass_mult;
    env->log.ixx_mult_mean += agent->params.ixx_mult;
    env->log.iyy_mult_mean += agent->params.iyy_mult;
    env->log.izz_mult_mean += agent->params.izz_mult;
    float motor_scale_min = agent->params.motor_thrust_scale[0];
    float motor_scale_max = agent->params.motor_thrust_scale[0];
    float motor_scale_sum = 0.0f;
    for (int m = 0; m < 4; m++) {
        float s = agent->params.motor_thrust_scale[m];
        if (s < motor_scale_min) motor_scale_min = s;
        if (s > motor_scale_max) motor_scale_max = s;
        motor_scale_sum += s;
    }
    env->log.k_thrust_mult_mean += agent->params.k_thrust_mult * motor_scale_sum * 0.25f;
    DomainRandomization dr = env_domain_randomization(env);
    if (dr_authority_gated(&dr)) {
        env->log.k_thrust_mult_min += agent->params.k_thrust_mult * motor_scale_min;
        env->log.k_thrust_mult_max += agent->params.k_thrust_mult * motor_scale_max;
    } else {
        float k_thrust_range = dr_param_range(&dr, dr.k_thrust);
        env->log.k_thrust_mult_min += 1.0f - k_thrust_range;
        env->log.k_thrust_mult_max += 1.0f + k_thrust_range;
    }
    env->log.linear_drag_mult_mean += agent->params.linear_drag_mult;
    env->log.yaw_drag_mult_mean += agent->params.yaw_drag_mult;
    env->log.motor_lag_mult_mean += agent->params.motor_lag_mult;
    env->log.com_x_mean += agent->params.com_x;
    env->log.com_y_mean += agent->params.com_y;
    env->log.com_z_mean += agent->params.com_z;

    env->log.n += 1.0f;

    agent->episode_length = 0;
    agent->episode_return = 0.0f;
    agent->collisions = 0.0f;
    agent->score = 0.0f;
    agent->rings_passed = 0.0f;
}

void compute_observations(DroneEnv* env) {
    for (int i = 0; i < env->num_agents; i++) {
        float* obs = env->observations + i*23;
        compute_drone_observations(&env->agents[i], obs);
        if (env->sensor_noise > 0.0f) {
            float noise = fminf(fabsf(env->sensor_noise), 1.0f);
            for (int j = 0; j < 23; j++) {
                obs[j] = clampf(obs[j] + rndf(-noise, noise, &env->rng), -2.0f, 2.0f);
            }
        }
    }
}

static inline int env_action_latency_steps(DroneEnv* env) {
    int steps = (int)floorf((fmaxf(env->action_latency, 0.0f) / ACTION_DT) + 0.5f);
    if (steps < 0) return 0;
    if (steps > MAX_ACTION_LATENCY_STEPS) return MAX_ACTION_LATENCY_STEPS;
    return steps;
}

static inline void apply_action_latency(Drone* agent, float raw_actions[4], int delay_steps,
                                        float delayed_actions[4]) {
    delay_steps = delay_steps < 0 ? 0 : delay_steps;
    delay_steps = delay_steps > MAX_ACTION_LATENCY_STEPS ? MAX_ACTION_LATENCY_STEPS : delay_steps;

    agent->action_history_idx = (agent->action_history_idx + 1) % (MAX_ACTION_LATENCY_STEPS + 1);
    for (int m = 0; m < 4; m++) {
        agent->action_history[agent->action_history_idx][m] = raw_actions[m];
    }

    int read_idx = agent->action_history_idx - delay_steps;
    if (read_idx < 0) read_idx += MAX_ACTION_LATENCY_STEPS + 1;
    for (int m = 0; m < 4; m++) {
        delayed_actions[m] = agent->action_history[read_idx][m];
    }
}

static inline DomainRandomization env_domain_randomization(DroneEnv* env) {
    return (DomainRandomization){
        .enabled = env->domain_randomization,
        .mass = env->dr_mass,
        .inertia = env->dr_inertia,
        .k_thrust = env->dr_k_thrust,
        .linear_drag = env->dr_linear_drag,
        .yaw_drag = env->dr_yaw_drag,
        .motor_lag = env->dr_motor_lag,
        .com_xy = env->dr_com_xy,
        .com_z = env->dr_com_z,
        .authority_gated = env->dr_authority_gated,
        .normalized_thrust_max = env->normalized_thrust_max,
        .usable_t2w_min = env->dr_usable_t2w_min,
        .usable_t2w_max = env->dr_usable_t2w_max,
        .mass_min = env->dr_mass_min,
        .mass_max = env->dr_mass_max,
        .inertia_min = env->dr_inertia_min,
        .inertia_max = env->dr_inertia_max,
        .motor_thrust_min = env->dr_motor_thrust_min,
        .motor_thrust_max = env->dr_motor_thrust_max,
        .motor_tau_min = env->dr_motor_tau_min,
        .motor_tau_max = env->dr_motor_tau_max,
        .yaw_torque_min = env->dr_yaw_torque_min,
        .yaw_torque_max = env->dr_yaw_torque_max,
        .linear_drag_min = env->dr_linear_drag_min,
        .linear_drag_max = env->dr_linear_drag_max,
        .angular_damping_min = env->dr_angular_damping_min,
        .angular_damping_max = env->dr_angular_damping_max,
        .profile_mix = env->dr_profile_mix,
    };
}

void reset_agent(DroneEnv* env, Drone* agent, int idx) {
    agent->episode_return = 0.0f;
    agent->episode_length = 0;
    agent->collisions = 0.0f;
    agent->rings_passed = 0;
    agent->score = 0.0f;
    agent->hover_score = 0.0f;
    agent->hover_ema = 0.0f;
    agent->ema_dist = 0.0f;
    agent->ema_vel = 0.0f;
    agent->ema_omega = 0.0f;
    agent->ema_omega_x = 0.0f;
    agent->ema_omega_y = 0.0f;
    agent->ema_omega_z = 0.0f;
    agent->action_abs_sum = 0.0f;
    agent->action_clipped_abs_sum = 0.0f;
    agent->action_max_abs = 0.0f;
    agent->action_saturation_count = 0.0f;
    agent->motor_clip_low_count = 0.0f;
    agent->motor_clip_high_count = 0.0f;
    for (int i = 0; i < 4; i++) agent->rpm_sum[i] = 0.0f;
    agent->instrumentation_steps = 0.0f;
    agent->r_dist_sum = 0.0f;
    agent->r_hover_sum = 0.0f;
    agent->r_shaping_sum = 0.0f;
    agent->r_omega_sum = 0.0f;
    agent->r_omega_xy_sum = 0.0f;
    agent->r_omega_z_sum = 0.0f;
    agent->r_terminal_sum = 0.0f;

    agent->buffer = env->ring_buffer;
    agent->buffer_size = env->max_rings;

    DomainRandomization dr = env_domain_randomization(env);
    init_drone(agent, &env->rng, &dr);
    agent->params.action_scale = env->action_scale;
    agent->params.action_mode = env->action_mode;
    agent->params.normalized_thrust_min = env->normalized_thrust_min;
    agent->params.normalized_thrust_max = env->normalized_thrust_max;

    float pos_scale = clampf(env->reset_pos_scale, 0.0f, 1.0f);
    agent->state.pos = (Vec3){
        rndf(-MARGIN_X * pos_scale, MARGIN_X * pos_scale, &env->rng),
        rndf(-MARGIN_Y * pos_scale, MARGIN_Y * pos_scale, &env->rng),
        rndf(-MARGIN_Z * pos_scale, MARGIN_Z * pos_scale, &env->rng)
    };

    if (env->reset_yaw_range > 0.0f) {
        float yaw = rndf(-env->reset_yaw_range, env->reset_yaw_range, &env->rng);
        agent->state.quat = quat_from_axis_angle((Vec3){0.0f, 0.0f, 1.0f}, yaw);
    }

    if (env->reset_vel_max > 0.0f) {
        float u = rndf(0.0f, 1.0f, &env->rng);
        float v = rndf(0.0f, 1.0f, &env->rng);
        float z = 2.0f * v - 1.0f;
        float a = 2.0f * (float)M_PI * u;
        float r_xy = sqrtf(fmaxf(0.0f, 1.0f - z * z));
        Vec3 dir = (Vec3){r_xy * cosf(a), r_xy * sinf(a), z};
        float speed = env->reset_vel_max * cbrtf(rndf(0.0f, 1.0f, &env->rng));
        agent->state.vel = scalmul3(dir, speed);
    }

    if (env->task == RACE) {
        while (norm3(sub3(agent->state.pos, env->ring_buffer[0].pos)) < 2.0f * RING_RADIUS) {
            agent->state.pos = (Vec3){
                rndf(-MARGIN_X * pos_scale, MARGIN_X * pos_scale, &env->rng),
                rndf(-MARGIN_Y * pos_scale, MARGIN_Y * pos_scale, &env->rng),
                rndf(-MARGIN_Z * pos_scale, MARGIN_Z * pos_scale, &env->rng)
            };
        }
    }

    agent->prev_pos = agent->state.pos;
}

static inline void finalize_reset_potential(DroneEnv* env, Drone* agent) {
    agent->prev_pos = agent->state.pos;
    agent->prev_potential = hover_potential(agent, env->hover_dist, env->hover_omega, env->hover_vel);
}

void c_reset(DroneEnv* env) {
    if (env->task == RACE) {
        reset_rings(&env->rng, env->ring_buffer, env->max_rings);
    }

    for (int i = 0; i < env->num_agents; i++) {
        Drone* agent = &env->agents[i];
        reset_agent(env, agent, i);
        set_target(&env->rng, env->task, env->agents, i, env->num_agents, env->hover_target_dist);
        finalize_reset_potential(env, agent);
    }

    compute_observations(env);
}

void c_step(DroneEnv* env) {
    env->tick = (env->tick + 1) % HORIZON;

    for (int i = 0; i < env->num_agents; i++) {
        Drone* agent = &env->agents[i];

        agent->prev_pos = agent->state.pos;
        float raw_actions[4] = {
            env->actions[4 * i + 0],
            env->actions[4 * i + 1],
            env->actions[4 * i + 2],
            env->actions[4 * i + 3],
        };
        float delayed_actions[4];
        apply_action_latency(agent, raw_actions, env_action_latency_steps(env), delayed_actions);
        move_drone(agent, delayed_actions);
        agent->episode_length++;

        bool oob = norm3(sub3(agent->target->pos, agent->state.pos)) > env->oob_radius;
        bool timeout = (agent->episode_length >= HORIZON);

        float curr = hover_potential(agent, env->hover_dist, env->hover_omega, env->hover_vel);
        float prev_dist = norm3(sub3(agent->target->pos, agent->prev_pos));
        float curr_dist = norm3(sub3(agent->target->pos, agent->state.pos));
        float omega = norm3(agent->state.omega);
        float omega_xy = sqrtf(agent->state.omega.x * agent->state.omega.x
                             + agent->state.omega.y * agent->state.omega.y);
        float omega_z = agent->state.omega.z;
        float omega_z_abs = fabsf(omega_z);
        float r_omega_xy = -env->alpha_omega_xy * omega_xy;
        float r_omega_z = -env->alpha_omega_z * omega_z_abs
                        - env->alpha_omega_z_sq * env->alpha_omega_z_mult * omega_z * omega_z;

        // Branch goal: penalize yaw spin without destroying translational navigation.
        float r_dist = env->alpha_dist * (prev_dist - curr_dist);
        float r_hover = env->alpha_hover * curr;
        float r_shaping = env->alpha_shaping * (curr - agent->prev_potential);
        float r_omega = r_omega_xy + r_omega_z;
        float r_terminal = 0.0f;
        float reward = r_dist + r_hover + r_shaping + r_omega + r_terminal;
        
        agent->prev_potential = curr;

        float h = check_hover(agent, env->hover_dist, env->hover_omega, env->hover_vel);
        agent->hover_score += h;
        agent->hover_ema = (1.0f - 0.02f) * agent->hover_ema + 0.02f * h;
        agent->ema_dist = 0.99f * agent->ema_dist + 0.01f * curr_dist;
        agent->ema_vel = 0.99f * agent->ema_vel + 0.01f * norm3(agent->state.vel);
        agent->ema_omega = 0.99f * agent->ema_omega + 0.01f * omega;
        agent->ema_omega_x = 0.99f * agent->ema_omega_x + 0.01f * fabsf(agent->state.omega.x);
        agent->ema_omega_y = 0.99f * agent->ema_omega_y + 0.01f * fabsf(agent->state.omega.y);
        agent->ema_omega_z = 0.99f * agent->ema_omega_z + 0.01f * fabsf(agent->state.omega.z);
        record_step_metrics(agent, raw_actions, r_dist, r_hover, r_shaping, r_omega,
                            r_omega_xy, r_omega_z, r_terminal);
        agent->episode_return += reward;
        env->rewards[i] = reward;

        bool reset = oob || timeout;
        env->terminals[i] = reset ? 1.0f : 0.0f;

        if (reset) {
            add_log(env, i, oob, timeout);
            reset_agent(env, agent, i);
            set_target(&env->rng, env->task, env->agents, i, env->num_agents, env->hover_target_dist);
            finalize_reset_potential(env, agent);
        }
    }

    compute_observations(env);
}

void c_close_client(Client* client);

void c_close(DroneEnv* env) {
    for (int i = 0; i < env->num_agents; i++) {
        free(env->agents[i].target);
    }

    free(env->agents);
    free(env->ring_buffer);

    if (env->client != NULL) {
        c_close_client(env->client);
    }
}
