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
    float alpha_action_delta;
    float alpha_reset_action_delta;
    int reset_action_interval;

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
    float adr_enabled;
    float adr_mode;
    float adr_probe_prob;
    float adr_success_threshold;
    float adr_contract_threshold;
    float adr_step;
    float adr_eval_episodes;
    float adr_init_usable_t2w_min;
    float adr_init_usable_t2w_max;
    float adr_init_mass_min;
    float adr_init_mass_max;
    float adr_init_inertia_min;
    float adr_init_inertia_max;
    float adr_init_motor_thrust_min;
    float adr_init_motor_thrust_max;
    float adr_init_motor_tau_min;
    float adr_init_motor_tau_max;
    float adr_init_com_xy;
    float pal_probe_prob;
    int pal_probe_steps;
    float pal_probe_amp;
    float action_scale;
    int action_mode;
    float normalized_thrust_min;
    float normalized_thrust_max;
    float reset_pos_scale;
    float reset_yaw_range;
    float reset_vel_max;
    float action_latency;
    float sensor_noise;
    float camera_3x1_enabled;
    float camera_fov_x;
    float camera_fov_y;
    float camera_gate_gain;
    float camera_bg;
    float camera_noise;
    float race_gate_spacing;
    float race_lateral_range;
    float race_vertical_range;
    float race_spawn_dist;
    float race_spawn_jitter;
    float race_gate_reward;
    float race_gate_hit_penalty;
    float race_progress_scale;
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
                                        float r_omega_xy, float r_omega_z, float r_terminal,
                                        float action_delta_mean, float reset_action_jump) {
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
    agent->action_delta_sum += action_delta_mean;
    agent->reset_action_jump_sum += reset_action_jump;
    if (reset_action_jump > 0.0f) agent->reset_action_jump_count += 1.0f;
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
    for (int i = 0; i < 4; i++) agent->prev_action[i] = raw_actions[i];
    agent->has_prev_action = 1;
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
    env->log.mean_abs_delta_action += agent->action_delta_sum / steps;
    env->log.reset_action_jump_mean += agent->reset_action_jump_count > 0.0f
        ? agent->reset_action_jump_sum / agent->reset_action_jump_count
        : 0.0f;
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

static inline float smoothstep01(float x) {
    x = clampf(x, 0.0f, 1.0f);
    return x * x * (3.0f - 2.0f * x);
}

static inline float soft_band_weight(float err, float inner, float outer) {
    err = fabsf(err);
    outer = fmaxf(outer, inner + 1e-4f);
    if (err <= inner) return 1.0f;
    if (err >= outer) return 0.0f;
    return 1.0f - smoothstep01((err - inner) / (outer - inner));
}

static inline void append_camera_3x1_observations(DroneEnv* env, Drone* agent, float* obs) {
    int base = DRONE_BASE_OBS_SIZE;
    for (int i = 0; i < DRONE_CAMERA_3X1_RGB_SIZE; i++) obs[base + i] = 0.0f;
    if (env->camera_3x1_enabled <= 0.0f || agent->target == NULL) return;

    Quat q_inv = quat_inverse(agent->state.quat);
    Vec3 to_target_world = sub3(agent->target->pos, agent->state.pos);
    Vec3 to_target = quat_rotate(q_inv, to_target_world);
    float dist = fmaxf(norm3(to_target), 1e-3f);

    // Body +x is the forward/camera axis for this branch.
    if (to_target.x <= 0.0f) return;

    float fov_x = clampf(env->camera_fov_x, 10.0f, 170.0f) * ((float)M_PI / 180.0f);
    float fov_y = clampf(env->camera_fov_y, 10.0f, 170.0f) * ((float)M_PI / 180.0f);
    float gate_radius = agent->target->radius > 0.0f ? agent->target->radius : RING_RADIUS;
    float gate_ang = atan2f(gate_radius, dist);
    float az = atan2f(to_target.y, to_target.x);
    float el = atan2f(to_target.z, sqrtf(to_target.x * to_target.x + to_target.y * to_target.y));
    float v_weight = soft_band_weight(el, 0.5f * fov_y + gate_ang, 0.5f * fov_y + 2.0f * gate_ang + 0.02f);
    if (v_weight <= 0.0f) return;

    float half_pix = fov_x / 6.0f;
    float gain = fmaxf(env->camera_gate_gain, 0.0f);
    float bg = clampf(env->camera_bg, 0.0f, 1.0f);
    float noise = fminf(fabsf(env->camera_noise), 1.0f);
    float vertical_code = clampf(0.5f + 0.5f * el / fmaxf(0.5f * fov_y, 1e-4f), 0.0f, 1.0f);

    for (int px = 0; px < 3; px++) {
        float center = ((float)px - 1.0f) * (fov_x / 3.0f);
        float h_weight = soft_band_weight(az - center, half_pix + gate_ang, half_pix + 2.0f * gate_ang + 0.02f);
        float signal = clampf(gain * h_weight * v_weight, 0.0f, 1.0f);

        // Raw RGB-style active gate: left side reddish, right side greenish,
        // blue channel carries weak vertical color variation.
        float r = bg + signal * (px == 0 ? 1.00f : (px == 1 ? 0.70f : 0.25f));
        float g = bg + signal * (px == 2 ? 1.00f : (px == 1 ? 0.70f : 0.25f));
        float b = bg + signal * (0.25f + 0.75f * vertical_code);
        if (noise > 0.0f) {
            r += rndf(-noise, noise, &env->rng);
            g += rndf(-noise, noise, &env->rng);
            b += rndf(-noise, noise, &env->rng);
        }
        obs[base + px * 3 + 0] = clampf(r, 0.0f, 1.0f);
        obs[base + px * 3 + 1] = clampf(g, 0.0f, 1.0f);
        obs[base + px * 3 + 2] = clampf(b, 0.0f, 1.0f);
    }
}

void compute_observations(DroneEnv* env) {
    for (int i = 0; i < env->num_agents; i++) {
        float* obs = env->observations + i * DRONE_OBS_SIZE;
        compute_drone_observations(&env->agents[i], obs);
        append_camera_3x1_observations(env, &env->agents[i], obs);
        if (env->sensor_noise > 0.0f) {
            float noise = fminf(fabsf(env->sensor_noise), 1.0f);
            for (int j = 0; j < DRONE_BASE_OBS_SIZE; j++) {
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

static inline void apply_pal_probe(DroneEnv* env, Drone* agent, float actions[4]) {
    int steps = env->pal_probe_steps;
    float amp = fabsf(env->pal_probe_amp);
    if (!agent->pal_probe_active || steps <= 0 || amp <= 0.0f) return;
    if (agent->episode_length < 0 || agent->episode_length >= steps) return;

    float pattern[4] = {0};
    switch (agent->episode_length % 8) {
        case 0: pattern[0] =  1.0f; pattern[1] =  1.0f; pattern[2] =  1.0f; pattern[3] =  1.0f; break;
        case 1: pattern[0] = -1.0f; pattern[1] = -1.0f; pattern[2] = -1.0f; pattern[3] = -1.0f; break;
        case 2: pattern[0] =  1.0f; pattern[1] = -1.0f; pattern[2] =  1.0f; pattern[3] = -1.0f; break;
        case 3: pattern[0] = -1.0f; pattern[1] =  1.0f; pattern[2] = -1.0f; pattern[3] =  1.0f; break;
        case 4: pattern[0] =  1.0f; pattern[1] =  1.0f; pattern[2] = -1.0f; pattern[3] = -1.0f; break;
        case 5: pattern[0] = -1.0f; pattern[1] = -1.0f; pattern[2] =  1.0f; pattern[3] =  1.0f; break;
        case 6: pattern[0] =  1.0f; pattern[1] = -1.0f; pattern[2] = -1.0f; pattern[3] =  1.0f; break;
        default: pattern[0] = -1.0f; pattern[1] =  1.0f; pattern[2] =  1.0f; pattern[3] = -1.0f; break;
    }

    for (int m = 0; m < 4; m++) {
        actions[m] = clampf(actions[m] + amp * pattern[m], -1.0f, 1.0f);
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

static inline void reset_camera_race_track(DroneEnv* env) {
    float spacing = fmaxf(env->race_gate_spacing, 2.5f * RING_RADIUS);
    float lateral = fmaxf(env->race_lateral_range, 0.0f);
    float vertical = fmaxf(env->race_vertical_range, 0.0f);
    float y = 0.0f;
    float z = 0.0f;

    for (int i = 0; i < env->max_rings; i++) {
        if (i > 0) {
            y = clampf(y + rndf(-0.5f * lateral, 0.5f * lateral, &env->rng), -lateral, lateral);
            z = clampf(z + rndf(-0.5f * vertical, 0.5f * vertical, &env->rng), -vertical, vertical);
        }
        env->ring_buffer[i] = (Target){
            .pos = (Vec3){spacing * (float)(i + 1), y, z},
            .vel = (Vec3){0.0f, 0.0f, 0.0f},
            .orientation = (Quat){1.0f, 0.0f, 0.0f, 0.0f},
            .normal = (Vec3){1.0f, 0.0f, 0.0f},
            .radius = RING_RADIUS,
        };
    }
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
    agent->action_delta_sum = 0.0f;
    agent->reset_action_jump_sum = 0.0f;
    agent->reset_action_jump_count = 0.0f;
    agent->has_prev_action = 0;
    agent->pal_probe_active = rndf(0.0f, 1.0f, &env->rng) < clampf(env->pal_probe_prob, 0.0f, 1.0f);
    for (int i = 0; i < 4; i++) agent->prev_action[i] = 0.0f;
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
    if (!(env->domain_randomization > 0.0f && env->dr_authority_gated > 0.0f
          && env->dr_profile_mix >= 2.0f && env->dr_profile_mix < 3.0f)) {
        agent->params.normalized_thrust_max = env->normalized_thrust_max;
    }

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
        agent->buffer_idx = 0;
        Target* first_gate = &env->ring_buffer[0];
        float spawn_dist = fmaxf(env->race_spawn_dist, 2.0f * RING_RADIUS);
        float jitter = fmaxf(env->race_spawn_jitter, 0.0f);
        Vec3 spawn_offset = scalmul3(first_gate->normal, -spawn_dist);
        agent->state.pos = add3(first_gate->pos, spawn_offset);
        agent->state.pos.y += rndf(-jitter, jitter, &env->rng);
        agent->state.pos.z += rndf(-0.5f * jitter, 0.5f * jitter, &env->rng);
        agent->state.quat = (Quat){1.0f, 0.0f, 0.0f, 0.0f};
        agent->state.vel = (Vec3){0.0f, 0.0f, 0.0f};
        agent->state.omega = (Vec3){0.0f, 0.0f, 0.0f};
    }

    agent->prev_pos = agent->state.pos;
}

static inline void finalize_reset_potential(DroneEnv* env, Drone* agent) {
    agent->prev_pos = agent->state.pos;
    agent->prev_potential = hover_potential(agent, env->hover_dist, env->hover_omega, env->hover_vel);
}

void c_reset(DroneEnv* env) {
    if (env->task == RACE) {
        reset_camera_race_track(env);
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
        apply_pal_probe(env, agent, raw_actions);
        float action_delta_mean = 0.0f;
        if (agent->has_prev_action) {
            for (int m = 0; m < 4; m++) {
                action_delta_mean += fabsf(raw_actions[m] - agent->prev_action[m]);
            }
            action_delta_mean *= 0.25f;
        }
        bool reset_action_boundary = env->reset_action_interval > 0
            && ((env->tick - 1 + env->reset_action_interval) % env->reset_action_interval) == 0;
        float reset_action_jump = reset_action_boundary ? action_delta_mean : 0.0f;
        float delayed_actions[4];
        apply_action_latency(agent, raw_actions, env_action_latency_steps(env), delayed_actions);
        move_drone(agent, delayed_actions);
        agent->episode_length++;

        int ring_result = env->task == RACE ? check_ring(agent, agent->target) : 0;
        bool gate_passed = ring_result > 0;
        bool ring_collision = ring_result < 0;
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
        if (env->task == RACE) {
            float prev_plane = dot3(sub3(agent->prev_pos, agent->target->pos), agent->target->normal);
            float curr_plane = dot3(sub3(agent->state.pos, agent->target->pos), agent->target->normal);
            r_dist = 0.25f * env->alpha_dist * (prev_dist - curr_dist)
                   + env->race_progress_scale * (curr_plane - prev_plane);
            r_hover = 0.0f;
            r_shaping = 0.0f;
            if (gate_passed) {
                r_terminal += env->race_gate_reward;
            } else if (ring_collision) {
                r_terminal -= env->race_gate_hit_penalty;
            }
        }
        float r_action_delta = -env->alpha_action_delta * action_delta_mean
                             -env->alpha_reset_action_delta * reset_action_jump;
        float reward = r_dist + r_hover + r_shaping + r_omega + r_terminal + r_action_delta;
        
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
                            r_omega_xy, r_omega_z, r_terminal,
                            action_delta_mean, reset_action_jump);
        agent->episode_return += reward;
        env->rewards[i] = reward;

        if (gate_passed) {
            agent->rings_passed += 1;
            agent->buffer_idx = (agent->buffer_idx + 1) % agent->buffer_size;
            set_target(&env->rng, env->task, env->agents, i, env->num_agents, env->hover_target_dist);
            agent->prev_potential = hover_potential(agent, env->hover_dist, env->hover_omega, env->hover_vel);
        }
        if (ring_collision) {
            agent->collisions += 1.0f;
        }

        bool reset = oob || timeout || ring_collision;
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
