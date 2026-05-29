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
    float minimal_vision_enabled;
    float minimal_vision_only;
    float minimal_vision_mask_target;
    float minimal_vision_fov;
    float minimal_vision_vfov;
    float minimal_vision_sigma;
    float minimal_vision_depth_gain;
    float minimal_vision_noise;
    float minimal_vision_distractors;
    float minimal_vision_spawn_visible_target;
    float minimal_vision_gate_mask;
    float race_track_mode;
    float race_segment_mode;
    float race_isb_enabled;
    float race_isb_prob;
    float race_isb_margin;
    float race_isb_pos_xy;
    float race_isb_z;
    float race_isb_angle;
    float race_isb_vel;
    float race_isb_omega;
    float race_hard_gate_idx;
    float race_hard_gate_prob;
    float race_course_yaw_delta;
    float race_course_pitch_delta;
    float race_course_pitch_limit;
    float race_course_spacing_min;
    float race_course_spacing_max;
    float race_course_dz_max;
    float race_reset_start_prob;
    float race_reset_t_min;
    float race_reset_t_max;
    float race_reset_lateral;
    float race_reset_yaw_error_frac;
    float race_reset_speed_min;
    float race_reset_speed_max;
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

static inline int race_clamped_gate_idx(const Drone* agent) {
    if (agent->buffer_size <= 0) return 0;
    int idx = agent->buffer_idx;
    if (idx < 0) return 0;
    if (idx >= agent->buffer_size) return agent->buffer_size - 1;
    return idx;
}

static inline float point_segment_distance3(Vec3 p, Vec3 a, Vec3 b) {
    Vec3 ab = sub3(b, a);
    float denom = dot3(ab, ab);
    if (denom <= 1e-6f) return norm3(sub3(p, b));
    float t = dot3(sub3(p, a), ab) / denom;
    t = clampf(t, 0.0f, 1.0f);
    Vec3 closest = add3(a, scalmul3(ab, t));
    return norm3(sub3(p, closest));
}

static inline float race_track_centerline_distance(const Drone* agent) {
    if (agent->buffer == NULL || agent->buffer_size <= 0) return 0.0f;
    int idx = race_clamped_gate_idx(agent);
    int prev_idx = idx > 0 ? idx - 1 : idx;
    return point_segment_distance3(agent->state.pos,
                                   agent->buffer[prev_idx].pos,
                                   agent->buffer[idx].pos);
}

static inline Vec3 race_next_gate_pos(const Drone* agent) {
    if (agent->buffer == NULL || agent->buffer_size <= 0) return agent->target->pos;
    int idx = race_clamped_gate_idx(agent);
    int next_idx = idx + 1 < agent->buffer_size ? idx + 1 : idx;
    return agent->buffer[next_idx].pos;
}

void add_log(DroneEnv* env, int idx, bool oob, bool timeout, bool lap_complete) {
    Drone* agent = &env->agents[idx];
    float steps = fmaxf(agent->instrumentation_steps, 1.0f);

    env->log.episode_return += agent->episode_return;
    env->log.episode_length += agent->episode_length;
    env->log.collisions += agent->collisions;
    env->log.ring_collision += agent->ring_collision;

    if (oob) env->log.oob += 1.0f;
    if (timeout) env->log.timeout += 1.0f;
    if (lap_complete) env->log.lap_complete += 1.0f;
    if (env->task == RACE) {
        int gate_idx = race_clamped_gate_idx(agent);
        if (gate_idx >= 0 && gate_idx < DRONE_GATE_DEBUG_MAX) {
            if (oob) env->log.gate_oob_count[gate_idx] += 1.0f;
            if (timeout) env->log.gate_timeout_count[gate_idx] += 1.0f;
        }
    }
    if (oob && env->task == RACE) {
        Vec3 next_gate_pos = race_next_gate_pos(agent);
        env->log.oob_diag_count += 1.0f;
        env->log.gate_index_at_oob += (float)race_clamped_gate_idx(agent);
        env->log.position_norm_at_oob += norm3(agent->state.pos);
        env->log.target_gate_position_norm += norm3(agent->target->pos);
        env->log.next_gate_position_norm += norm3(next_gate_pos);
        env->log.distance_from_track_centerline += race_track_centerline_distance(agent);
    }

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
    env->log.target_in_fov_frac += agent->target_in_fov_sum / steps;
    env->log.retina_rgb_mean += agent->retina_rgb_mean_sum / steps;
    env->log.retina_rgb_std += agent->retina_rgb_std_sum / steps;
    env->log.retina_energy += agent->retina_energy_sum / steps;
    env->log.retina_left_center_right_argmax += agent->retina_argmax_sum / steps;
    env->log.retina_argmax_left_frac += agent->retina_argmax_left_count / steps;
    env->log.retina_argmax_center_frac += agent->retina_argmax_center_count / steps;
    env->log.retina_argmax_right_frac += agent->retina_argmax_right_count / steps;
    env->log.bearing_error_to_target += agent->bearing_error_sum / steps;
    env->log.distance_to_target += agent->distance_to_target_sum / steps;
    env->log.retina_signal_vs_distance += agent->retina_signal_vs_distance_sum / steps;
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
    if (env->task == RACE) {
        for (int gate = 0; gate < DRONE_GATE_DEBUG_MAX; gate++) {
            env->log.gate_time[gate] += agent->gate_time[gate];
            env->log.gate_target_in_fov[gate] += agent->gate_target_in_fov[gate];
            env->log.gate_bearing_error[gate] += agent->gate_bearing_error[gate];
            env->log.gate_distance_to_target[gate] += agent->gate_distance_to_target[gate];
            env->log.gate_pass_count[gate] += agent->gate_pass_count[gate];
            env->log.gate_collision_count[gate] += agent->gate_collision_count[gate];
        }
    }

    env->log.n += 1.0f;

    agent->episode_length = 0;
    agent->episode_return = 0.0f;
    agent->collisions = 0.0f;
    agent->ring_collision = 0.0f;
    agent->score = 0.0f;
    agent->rings_passed = 0.0f;
}

static inline float minimal_vision_blob_intensity(Quat q_inv, Vec3 pos, Vec3 target_pos,
                                                  float center_x, float center_y,
                                                  float sigma_x, float sigma_y,
                                                  float depth_gain) {
    Vec3 to_target_world = sub3(target_pos, pos);
    Vec3 to_target = quat_rotate(q_inv, to_target_world);
    float dist = norm3(to_target_world);
    float xy = sqrtf(to_target.x * to_target.x + to_target.y * to_target.y);
    float yaw = atan2f(to_target.y, fmaxf(to_target.x, 1e-3f));
    float pitch = atan2f(to_target.z, fmaxf(xy, 1e-3f));
    float front = to_target.x > 0.0f ? 1.0f : 0.0f;
    float depth = 1.0f / (1.0f + depth_gain * dist);
    float dx = yaw - center_x;
    float dy = pitch - center_y;
    float h = expf(-0.5f * (dx / sigma_x) * (dx / sigma_x));
    float v = expf(-0.5f * (dy / sigma_y) * (dy / sigma_y));
    return front * depth * h * v;
}

static inline float minimal_vision_gate_mask_intensity(Quat q_inv, Vec3 pos, Vec3 gate_pos,
                                                       float gate_radius, float center_x,
                                                       float center_y, float fov,
                                                       float vfov, float depth_gain) {
    Vec3 rel = quat_rotate(q_inv, sub3(gate_pos, pos));
    if (rel.x <= 0.0f) return 0.0f;

    float yaw = atan2f(rel.y, fmaxf(rel.x, 1e-5f));
    float xy = sqrtf(rel.x * rel.x + rel.y * rel.y);
    float pitch = atan2f(rel.z, fmaxf(xy, 1e-5f));
    float dist = fmaxf(norm3(rel), 1e-3f);
    float radius = atan2f(fmaxf(gate_radius, 0.05f), dist);
    float dx = center_x - yaw;
    float dy = center_y - pitch;
    float rho = sqrtf(dx * dx + dy * dy);
    float pixel = fminf(fov / (float)DRONE_MINIMAL_VISION_WIDTH,
                        vfov / (float)DRONE_MINIMAL_VISION_HEIGHT);
    float edge_sigma = fmaxf(0.35f * pixel, 0.18f * radius);
    float e = (rho - radius) / fmaxf(edge_sigma, 1e-4f);
    float depth = 1.0f / (1.0f + depth_gain * dist);
    return expf(-0.5f * e * e) * depth;
}

static inline void minimal_vision_pixel_rgb(DroneEnv* env, Drone* agent, int px, int py,
                                            float* red, float* green, float* blue) {
    Quat q_inv = quat_inverse(agent->state.quat);
    float fov = fmaxf(fabsf(env->minimal_vision_fov), 0.1f);
    float vfov = fmaxf(fabsf(env->minimal_vision_vfov), 0.1f);
    float sigma = fmaxf(fabsf(env->minimal_vision_sigma), 0.01f);
    float depth_gain = fmaxf(fabsf(env->minimal_vision_depth_gain), 0.0f);
    float sigma_y = DRONE_MINIMAL_VISION_HEIGHT <= 1
        ? fmaxf(0.35f * vfov, 1e-3f)
        : fmaxf(sigma * vfov / fov, 0.01f);
    float center_y = DRONE_MINIMAL_VISION_HEIGHT <= 1
        ? 0.0f
        : 0.5f * vfov - ((float)py + 0.5f) * (vfov / (float)DRONE_MINIMAL_VISION_HEIGHT);
    float center_x = -0.5f * fov + ((float)px + 0.5f) * (fov / (float)DRONE_MINIMAL_VISION_WIDTH);

    if (env->minimal_vision_gate_mask > 0.0f) {
        if (env->task == RACE && agent->buffer_size > 0) {
            int idx0 = agent->buffer_idx;
            if (idx0 < 0) idx0 = 0;
            if (idx0 >= agent->buffer_size) idx0 = agent->buffer_size - 1;
            int idx1 = agent->buffer_size > 1 ? (idx0 + 1) % agent->buffer_size : idx0;
            int idx2 = agent->buffer_size > 2 ? (idx0 + 2) % agent->buffer_size : idx1;
            *red = minimal_vision_gate_mask_intensity(
                q_inv, agent->state.pos, agent->buffer[idx0].pos, agent->buffer[idx0].radius,
                center_x, center_y, fov, vfov, depth_gain);
            *green = agent->buffer_size > 1
                ? 0.85f * minimal_vision_gate_mask_intensity(
                    q_inv, agent->state.pos, agent->buffer[idx1].pos, agent->buffer[idx1].radius,
                    center_x, center_y, fov, vfov, depth_gain)
                : 0.0f;
            *blue = agent->buffer_size > 2
                ? 0.70f * minimal_vision_gate_mask_intensity(
                    q_inv, agent->state.pos, agent->buffer[idx2].pos, agent->buffer[idx2].radius,
                    center_x, center_y, fov, vfov, depth_gain)
                : 0.0f;
            return;
        }

        *red = minimal_vision_gate_mask_intensity(
            q_inv, agent->state.pos, agent->target->pos, RING_RADIUS,
            center_x, center_y, fov, vfov, depth_gain);
        *green = 0.0f;
        *blue = 0.0f;
        return;
    }

    if (env->task == RACE && agent->buffer_size > 0) {
        int idx0 = agent->buffer_idx;
        if (idx0 < 0) idx0 = 0;
        if (idx0 >= agent->buffer_size) idx0 = agent->buffer_size - 1;
        int idx1 = agent->buffer_size > 1 ? (idx0 + 1) % agent->buffer_size : idx0;
        int idx2 = agent->buffer_size > 2 ? (idx0 + 2) % agent->buffer_size : idx1;
        *red = minimal_vision_blob_intensity(q_inv, agent->state.pos, agent->buffer[idx0].pos,
                                             center_x, center_y, sigma, sigma_y, depth_gain);
        *green = agent->buffer_size > 1
            ? 0.85f * minimal_vision_blob_intensity(q_inv, agent->state.pos, agent->buffer[idx1].pos,
                                                    center_x, center_y, sigma, sigma_y, depth_gain)
            : 0.0f;
        *blue = agent->buffer_size > 2
            ? 0.70f * minimal_vision_blob_intensity(q_inv, agent->state.pos, agent->buffer[idx2].pos,
                                                    center_x, center_y, sigma, sigma_y, depth_gain)
            : 0.0f;
        return;
    }

    Vec3 to_target_world = sub3(agent->target->pos, agent->state.pos);
    Vec3 to_target = quat_rotate(q_inv, to_target_world);
    float dist = norm3(to_target_world);
    float xy = sqrtf(to_target.x * to_target.x + to_target.y * to_target.y);
    float yaw = atan2f(to_target.y, fmaxf(to_target.x, 1e-3f));
    float pitch = atan2f(to_target.z, fmaxf(xy, 1e-3f));
    float yaw_norm = clampf(yaw / (0.5f * fov), -1.0f, 1.0f);
    float pitch_norm = clampf(pitch / (0.5f * vfov), -1.0f, 1.0f);
    float intensity = minimal_vision_blob_intensity(q_inv, agent->state.pos, agent->target->pos,
                                                    center_x, center_y, sigma, sigma_y, depth_gain);
    (void)dist;
    *red = intensity * (0.65f + 0.35f * clampf(-yaw_norm, 0.0f, 1.0f));
    *green = intensity * (0.65f + 0.35f * clampf(yaw_norm, 0.0f, 1.0f));
    *blue = intensity * (0.55f + 0.45f * (1.0f - fabsf(pitch_norm)));
}

static inline void compute_minimal_vision_observations(DroneEnv* env, Drone* agent, float* obs) {
    float noise = clampf(fabsf(env->minimal_vision_noise), 0.0f, 1.0f);
    float distractors = clampf(fabsf(env->minimal_vision_distractors), 0.0f, 1.0f);

    for (int py = 0; py < DRONE_MINIMAL_VISION_HEIGHT; py++) {
        for (int px = 0; px < DRONE_MINIMAL_VISION_WIDTH; px++) {
            float red = 0.0f;
            float green = 0.0f;
            float blue = 0.0f;
            minimal_vision_pixel_rgb(env, agent, px, py, &red, &green, &blue);

            if (distractors > 0.0f && rndf(0.0f, 1.0f, &env->rng) < 0.03f * distractors) {
                red += distractors * rndf(0.0f, 0.25f, &env->rng);
                green += distractors * rndf(0.0f, 0.25f, &env->rng);
                blue += distractors * rndf(0.0f, 0.25f, &env->rng);
            }
            if (noise > 0.0f) {
                red += rndf(-noise, noise, &env->rng);
                green += rndf(-noise, noise, &env->rng);
                blue += rndf(-noise, noise, &env->rng);
            }

            int out = 3 * (py * DRONE_MINIMAL_VISION_WIDTH + px);
            obs[out + 0] = clampf(red, 0.0f, 1.0f);
            obs[out + 1] = clampf(green, 0.0f, 1.0f);
            obs[out + 2] = clampf(blue, 0.0f, 1.0f);
        }
    }
}

static inline void record_retina_diagnostics(DroneEnv* env, Drone* agent) {
    Quat q_inv = quat_inverse(agent->state.quat);
    Vec3 to_target_world = sub3(agent->target->pos, agent->state.pos);
    Vec3 to_target = quat_rotate(q_inv, to_target_world);

    float fov = fmaxf(fabsf(env->minimal_vision_fov), 0.1f);
    float vfov = fmaxf(fabsf(env->minimal_vision_vfov), 0.1f);
    float sigma = fmaxf(fabsf(env->minimal_vision_sigma), 0.01f);
    float depth_gain = fmaxf(fabsf(env->minimal_vision_depth_gain), 0.0f);
    float dist = norm3(to_target_world);
    float xy = sqrtf(to_target.x * to_target.x + to_target.y * to_target.y);
    float yaw = atan2f(to_target.y, fmaxf(to_target.x, 1e-3f));
    float pitch = atan2f(to_target.z, fmaxf(xy, 1e-3f));
    float front = to_target.x > 0.0f ? 1.0f : 0.0f;
    float depth = 1.0f / (1.0f + depth_gain * dist);
    float yaw_norm = clampf(yaw / (0.5f * fov), -1.0f, 1.0f);
    float pitch_norm = clampf(pitch / (0.5f * vfov), -1.0f, 1.0f);
    float sigma_y = DRONE_MINIMAL_VISION_HEIGHT <= 1
        ? fmaxf(0.35f * vfov, 1e-3f)
        : fmaxf(sigma * vfov / fov, 0.01f);

    float rgb_sum = 0.0f;
    float rgb_sq_sum = 0.0f;
    float bucket_energy[3] = {0.0f, 0.0f, 0.0f};
    for (int py = 0; py < DRONE_MINIMAL_VISION_HEIGHT; py++) {
        for (int px = 0; px < DRONE_MINIMAL_VISION_WIDTH; px++) {
        float red = 0.0f;
        float green = 0.0f;
        float blue = 0.0f;
        minimal_vision_pixel_rgb(env, agent, px, py, &red, &green, &blue);
        float rgb[3] = {
            clampf(red, 0.0f, 1.0f),
            clampf(green, 0.0f, 1.0f),
            clampf(blue, 0.0f, 1.0f),
        };
        float px_energy = 0.0f;
        for (int c = 0; c < 3; c++) {
            rgb_sum += rgb[c];
            rgb_sq_sum += rgb[c] * rgb[c];
            px_energy += rgb[c] * rgb[c];
        }
        int bucket = (3 * px) / DRONE_MINIMAL_VISION_WIDTH;
        if (bucket < 0) bucket = 0;
        if (bucket > 2) bucket = 2;
        bucket_energy[bucket] += px_energy;
        }
    }

    float inv_channels = 1.0f / (float)DRONE_MINIMAL_VISION_OBS_SIZE;
    float mean = rgb_sum * inv_channels;
    float energy = rgb_sq_sum * inv_channels;
    float var = fmaxf(0.0f, energy - mean * mean);
    float argmax = -1.0f;
    float best = bucket_energy[0];
    int best_idx = 0;
    for (int bucket = 1; bucket < 3; bucket++) {
        if (bucket_energy[bucket] > best) {
            best = bucket_energy[bucket];
            best_idx = bucket;
        }
    }
    if (best > 1e-8f) argmax = (float)best_idx;

    agent->target_in_fov_sum += (front > 0.0f && fabsf(yaw) <= 0.5f * fov
        && fabsf(pitch) <= 0.5f * vfov) ? 1.0f : 0.0f;
    agent->retina_rgb_mean_sum += mean;
    agent->retina_rgb_std_sum += sqrtf(var);
    agent->retina_energy_sum += energy;
    agent->retina_argmax_sum += argmax;
    if (argmax == 0.0f) agent->retina_argmax_left_count += 1.0f;
    else if (argmax == 1.0f) agent->retina_argmax_center_count += 1.0f;
    else if (argmax == 2.0f) agent->retina_argmax_right_count += 1.0f;
    agent->bearing_error_sum += sqrtf(yaw * yaw + pitch * pitch);
    agent->distance_to_target_sum += dist;
    agent->retina_signal_vs_distance_sum += energy * fmaxf(dist, 1e-3f);
    if (env->task == RACE) {
        int gate_idx = race_clamped_gate_idx(agent);
        if (gate_idx >= 0 && gate_idx < DRONE_GATE_DEBUG_MAX) {
            float in_fov = (front > 0.0f && fabsf(yaw) <= 0.5f * fov
                && fabsf(pitch) <= 0.5f * vfov) ? 1.0f : 0.0f;
            agent->gate_time[gate_idx] += 1.0f;
            agent->gate_target_in_fov[gate_idx] += in_fov;
            agent->gate_bearing_error[gate_idx] += sqrtf(yaw * yaw + pitch * pitch);
            agent->gate_distance_to_target[gate_idx] += dist;
        }
    }
}

void compute_observations(DroneEnv* env) {
    for (int i = 0; i < env->num_agents; i++) {
        float* obs = env->observations + i * DRONE_OBS_SIZE;
        for (int j = 0; j < DRONE_OBS_SIZE; j++) obs[j] = 0.0f;

        if (!(env->minimal_vision_only > 0.0f)) {
            compute_drone_observations(&env->agents[i], obs);
        }

        if (env->sensor_noise > 0.0f && !(env->minimal_vision_only > 0.0f)) {
            float noise = fminf(fabsf(env->sensor_noise), 1.0f);
            for (int j = 0; j < DRONE_STATE_OBS_SIZE; j++) {
                obs[j] = clampf(obs[j] + rndf(-noise, noise, &env->rng), -2.0f, 2.0f);
            }
        }
        if (!(env->minimal_vision_only > 0.0f) && env->minimal_vision_mask_target > 0.0f) {
            for (int j = 10; j < 19; j++) obs[j] = 0.0f;
        }

        if (env->minimal_vision_enabled > 0.0f) {
            int offset = env->minimal_vision_only > 0.0f ? 0 : DRONE_STATE_OBS_SIZE;
            compute_minimal_vision_observations(env, &env->agents[i], obs + offset);
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

static inline void set_minimal_vision_visible_target(DroneEnv* env, Drone* agent) {
    if (!(env->minimal_vision_spawn_visible_target > 0.0f)) return;
    if (!(env->minimal_vision_enabled > 0.0f)) return;
    if (env->task != HOVER) return;

    float fov = fmaxf(fabsf(env->minimal_vision_fov), 0.1f);
    float vfov = fmaxf(fabsf(env->minimal_vision_vfov), 0.1f);
    float dist = rndf(0.45f * env->hover_target_dist, env->hover_target_dist, &env->rng);
    float yaw = rndf(-0.35f * fov, 0.35f * fov, &env->rng);
    float pitch = rndf(-0.25f * vfov, 0.25f * vfov, &env->rng);
    float cp = cosf(pitch);
    Vec3 body = (Vec3){dist * cp * cosf(yaw), dist * cp * sinf(yaw), dist * sinf(pitch)};
    Vec3 world = quat_rotate(agent->state.quat, body);
    Vec3 p = add3(agent->state.pos, world);
    agent->target->pos = (Vec3){
        clampf(p.x, -MARGIN_X, MARGIN_X),
        clampf(p.y, -MARGIN_Y, MARGIN_Y),
        clampf(p.z, -MARGIN_Z, MARGIN_Z)
    };
    agent->target->vel = (Vec3){0.0f, 0.0f, 0.0f};
    agent->target->normal = (Vec3){0.0f, 0.0f, 1.0f};
    agent->target->orientation = (Quat){1.0f, 0.0f, 0.0f, 0.0f};
    agent->target->radius = 0.0f;
}

static inline Vec3 random_unit_vec3(unsigned int* rng) {
    float u = rndf(0.0f, 1.0f, rng);
    float v = rndf(0.0f, 1.0f, rng);
    float z = 2.0f * v - 1.0f;
    float a = 2.0f * (float)M_PI * u;
    float r_xy = sqrtf(fmaxf(0.0f, 1.0f - z * z));
    return (Vec3){r_xy * cosf(a), r_xy * sinf(a), z};
}

static inline Vec3 clamp_world_vec3(Vec3 p) {
    return (Vec3){
        clampf(p.x, -MARGIN_X, MARGIN_X),
        clampf(p.y, -MARGIN_Y, MARGIN_Y),
        clampf(p.z, -MARGIN_Z, MARGIN_Z)
    };
}

static inline Vec3 race_course_dir_from_yaw_pitch(float yaw, float pitch) {
    float cp = cosf(pitch);
    return normalize3_or((Vec3){
        cp * cosf(yaw),
        cp * sinf(yaw),
        sinf(pitch)
    }, (Vec3){1.0f, 0.0f, 0.0f});
}

static inline Vec3 race_course_next_visible_dir(DroneEnv* env, Vec3 prev_dir,
                                                unsigned int* rng) {
    prev_dir = normalize3_or(prev_dir, (Vec3){1.0f, 0.0f, 0.0f});
    float base_yaw = atan2f(prev_dir.y, prev_dir.x);
    float horiz = sqrtf(prev_dir.x * prev_dir.x + prev_dir.y * prev_dir.y);
    float base_pitch = atan2f(prev_dir.z, fmaxf(horiz, 1e-3f));

    float max_yaw_delta = env->race_course_yaw_delta > 0.0f
        ? env->race_course_yaw_delta
        : fminf(0.38f, 0.18f * fmaxf(fabsf(env->minimal_vision_fov), 0.1f));
    float max_pitch_delta = env->race_course_pitch_delta > 0.0f
        ? env->race_course_pitch_delta
        : fminf(0.20f, 0.12f * fmaxf(fabsf(env->minimal_vision_vfov), 0.1f));
    float pitch_limit = env->race_course_pitch_limit > 0.0f
        ? env->race_course_pitch_limit
        : 0.30f;
    float yaw = base_yaw + rndf(-max_yaw_delta, max_yaw_delta, rng);
    float pitch = clampf(base_pitch + rndf(-max_pitch_delta, max_pitch_delta, rng),
                         -pitch_limit, pitch_limit);
    return race_course_dir_from_yaw_pitch(yaw, pitch);
}

static inline Vec3 race_course_side(Vec3 dir) {
    Vec3 horiz = normalize3_or((Vec3){dir.x, dir.y, 0.0f},
                               (Vec3){1.0f, 0.0f, 0.0f});
    return normalize3_or((Vec3){-horiz.y, horiz.x, 0.0f},
                         (Vec3){0.0f, 1.0f, 0.0f});
}

static inline Quat race_reset_quat(float yaw, float pitch, float roll) {
    Quat q_yaw = quat_from_axis_angle((Vec3){0.0f, 0.0f, 1.0f}, yaw);
    Quat q_pitch = quat_from_axis_angle((Vec3){0.0f, 1.0f, 0.0f}, pitch);
    Quat q_roll = quat_from_axis_angle((Vec3){1.0f, 0.0f, 0.0f}, roll);
    Quat q = quat_mul(q_yaw, quat_mul(q_pitch, q_roll));
    quat_normalize(&q);
    return q;
}

static inline int race_swift_like_ring_count(DroneEnv* env) {
    int n = env->max_rings < 7 ? env->max_rings : 7;
    return n < 1 ? 1 : n;
}

static inline Vec3 race_swift_like_base_pos(int idx) {
    switch (idx) {
        case 0: return (Vec3){-0.60f, -0.86f, 3.68f};
        case 1: return (Vec3){ 9.00f,  6.45f, 1.05f};
        case 2: return (Vec3){ 8.85f, -3.80f, 1.05f};
        case 3: return (Vec3){-4.30f, -5.60f, 3.40f};
        case 4: return (Vec3){-4.30f, -5.60f, 1.42f};
        case 5: return (Vec3){ 4.50f, -0.45f, 1.05f};
        default: return (Vec3){-1.95f, 6.81f, 1.05f};
    }
}

static inline float race_swift_like_base_yaw(int idx) {
    switch (idx) {
        case 0: return -0.34906585f;  // -20 deg
        case 1: return  0.0f;
        case 2: return -2.26892803f;  // -130 deg
        case 3: return -(float)M_PI;
        case 4: return  0.0f;
        case 5: return  1.39626340f;  // 80 deg
        default: return -2.61799388f; // -150 deg
    }
}

static inline Vec3 rotate_yaw_vec3(Vec3 p, float yaw) {
    float c = cosf(yaw);
    float s = sinf(yaw);
    return (Vec3){
        c * p.x - s * p.y,
        s * p.x + c * p.y,
        p.z
    };
}

static inline void set_swift_like_race_course(DroneEnv* env, Drone* agent) {
    if (env->task != RACE) return;

    int n = race_swift_like_ring_count(env);
    agent->buffer_size = n;
    if (agent->buffer_idx >= n) agent->buffer_idx = n - 1;
    if (agent->buffer_idx < 0) agent->buffer_idx = 0;

    bool randomized = env->race_track_mode >= 2.0f;
    float global_yaw = randomized ? rndf(-(float)M_PI, (float)M_PI, &env->rng) : 0.0f;
    float scale = randomized ? rndf(0.90f, 1.10f, &env->rng) : 1.0f;
    float mirror = randomized && rndf(0.0f, 1.0f, &env->rng) < 0.5f ? -1.0f : 1.0f;
    Vec3 offset = randomized
        ? (Vec3){rndf(-3.0f, 3.0f, &env->rng), rndf(-3.0f, 3.0f, &env->rng),
                 rndf(-0.15f, 0.15f, &env->rng)}
        : (Vec3){0.0f, 0.0f, 0.0f};

    for (int i = 0; i < n; i++) {
        Vec3 p = race_swift_like_base_pos(i);
        p.y *= mirror;
        p = scalmul3(p, scale);
        p = rotate_yaw_vec3(p, global_yaw);
        p = clamp_world_vec3(add3(p, offset));

        float yaw = global_yaw + mirror * race_swift_like_base_yaw(i);
        Vec3 normal = normalize3_or((Vec3){cosf(yaw), sinf(yaw), 0.0f},
                                    (Vec3){1.0f, 0.0f, 0.0f});

        env->ring_buffer[i].pos = p;
        env->ring_buffer[i].normal = normal;
        env->ring_buffer[i].radius = RING_RADIUS;
        env->ring_buffer[i].orientation = quat_from_axis_angle((Vec3){0.0f, 0.0f, 1.0f}, yaw);
        env->ring_buffer[i].vel = (Vec3){0.0f, 0.0f, 0.0f};
    }
}

static inline void set_visible_race_course(DroneEnv* env, Drone* agent) {
    if (env->task != RACE) return;
    if (env->race_track_mode >= 1.0f) {
        set_swift_like_race_course(env, agent);
        return;
    }
    if (!(env->minimal_vision_spawn_visible_target > 0.0f)) return;
    if (!(env->minimal_vision_enabled > 0.0f)) return;
    if (env->num_agents != 1) return;

    int n = env->max_rings;
    if (n < 1) return;
    float segment = fmaxf(env->hover_target_dist, 4.0f);
    float fov = fmaxf(fabsf(env->minimal_vision_fov), 0.1f);
    float vfov = fmaxf(fabsf(env->minimal_vision_vfov), 0.1f);
    Vec3 prev_pos = agent->state.pos;
    Vec3 prev_dir = normalize3_or(
        quat_rotate(agent->state.quat, (Vec3){1.0f, 0.0f, 0.0f}),
        (Vec3){1.0f, 0.0f, 0.0f});

    for (int i = 0; i < n; i++) {
        Vec3 ring_pos;
        Vec3 normal;
        if (i == 0) {
            float dist = rndf(0.55f * segment, 0.90f * segment, &env->rng);
            float yaw = rndf(-0.18f * fov, 0.18f * fov, &env->rng);
            float pitch = rndf(-0.10f * vfov, 0.10f * vfov, &env->rng);
            float cp = cosf(pitch);
            Vec3 body = (Vec3){dist * cp * cosf(yaw),
                               dist * cp * sinf(yaw),
                               dist * sinf(pitch)};
            Vec3 world = quat_rotate(agent->state.quat, body);
            ring_pos = clamp_world_vec3(add3(agent->state.pos, world));
            normal = normalize3_or(world, prev_dir);
        } else {
            normal = race_course_next_visible_dir(env, prev_dir, &env->rng);
            float spacing_lo = env->race_course_spacing_min > 0.0f
                ? env->race_course_spacing_min
                : fmaxf(3.0f, 0.75f * clampf(segment, 3.0f, 5.0f));
            float spacing_hi = env->race_course_spacing_max > 0.0f
                ? env->race_course_spacing_max
                : clampf(segment, 3.0f, 5.0f);
            spacing_hi = fmaxf(spacing_hi, spacing_lo);
            float dist = rndf(spacing_lo, spacing_hi, &env->rng);
            float dz_max = env->race_course_dz_max > 0.0f ? env->race_course_dz_max : 0.35f;
            float dz = clampf(dist * normal.z, -dz_max, dz_max);
            float xy = sqrtf(fmaxf(0.0f, dist * dist - dz * dz));
            Vec3 prev_horiz = normalize3_or((Vec3){prev_dir.x, prev_dir.y, 0.0f},
                                            (Vec3){1.0f, 0.0f, 0.0f});
            Vec3 horiz = normalize3_or((Vec3){normal.x, normal.y, 0.0f}, prev_horiz);
            ring_pos = clamp_world_vec3(add3(prev_pos, (Vec3){
                horiz.x * xy,
                horiz.y * xy,
                dz
            }));
            normal = normalize3_or(sub3(ring_pos, prev_pos), normal);
        }
        env->ring_buffer[i].pos = ring_pos;
        env->ring_buffer[i].normal = normal;
        env->ring_buffer[i].radius = RING_RADIUS;
        env->ring_buffer[i].orientation = (Quat){1.0f, 0.0f, 0.0f, 0.0f};
        env->ring_buffer[i].vel = (Vec3){0.0f, 0.0f, 0.0f};
        prev_pos = ring_pos;
        prev_dir = normal;
    }
}

static inline int race_random_target_idx(DroneEnv* env, int n) {
    if (n <= 1) return 0;
    int idx = 1 + (int)floorf(rndf(0.0f, (float)(n - 1), &env->rng));
    if (idx < 1) idx = 1;
    if (idx >= n) idx = n - 1;
    return idx;
}

static inline int race_segment_target_idx(DroneEnv* env, int n) {
    if (n <= 1) return 0;
    int hard_idx = (int)floorf(env->race_hard_gate_idx + 0.5f);
    float hard_prob = clampf(env->race_hard_gate_prob, 0.0f, 1.0f);
    if (hard_idx >= 0 && hard_idx < n && rndf(0.0f, 1.0f, &env->rng) < hard_prob) {
        return hard_idx;
    }
    int idx = (int)floorf(rndf(0.0f, (float)n, &env->rng));
    if (idx < 0) idx = 0;
    if (idx >= n) idx = n - 1;
    return idx;
}

static inline void race_isb_push(DroneEnv* env, Drone* agent, int target_idx, float pass_margin) {
    if (!(env->race_isb_enabled > 0.0f)) return;
    if (env->race_track_mode >= 2.0f) return;
    if (target_idx < 0 || target_idx >= DRONE_GATE_DEBUG_MAX) return;
    float margin_min = env->race_isb_margin > 0.0f ? env->race_isb_margin : 0.8f;
    if (pass_margin < margin_min) return;
    int slot = agent->race_isb_cursor[target_idx];
    if (slot < 0 || slot >= DRONE_ISB_CAPACITY) slot = 0;
    agent->race_isb_states[target_idx][slot] = agent->state;
    agent->race_isb_cursor[target_idx] = (slot + 1) % DRONE_ISB_CAPACITY;
    if (agent->race_isb_count[target_idx] < DRONE_ISB_CAPACITY) {
        agent->race_isb_count[target_idx] += 1;
    }
}

static inline float race_pass_margin(Drone* agent, int gate_idx) {
    if (gate_idx < 0 || gate_idx >= agent->buffer_size) return -FLT_MAX;
    Target* gate = &agent->buffer[gate_idx];
    Vec3 ring_pos = gate->pos;
    Vec3 ring_normal = gate->normal;
    float prev_dot = dot3(sub3(agent->prev_pos, ring_pos), ring_normal);
    Vec3 dir = sub3(agent->state.pos, agent->prev_pos);
    float denom = dot3(ring_normal, dir);
    if (fabsf(denom) < 1e-9f) return -FLT_MAX;
    float t = -prev_dot / denom;
    Vec3 intersection = add3(agent->prev_pos, scalmul3(dir, t));
    float dist = norm3(sub3(intersection, ring_pos));
    return gate->radius - dist;
}

static inline void race_isb_perturb_state(DroneEnv* env, Drone* agent) {
    float pos_xy = env->race_isb_pos_xy > 0.0f ? env->race_isb_pos_xy : 0.45f;
    float pos_z = env->race_isb_z > 0.0f ? env->race_isb_z : 0.25f;
    float angle = env->race_isb_angle > 0.0f ? env->race_isb_angle : 0.18f;
    float vel = env->race_isb_vel > 0.0f ? env->race_isb_vel : 0.60f;
    float omega = env->race_isb_omega > 0.0f ? env->race_isb_omega : 0.60f;
    agent->state.pos.x += rndf(-pos_xy, pos_xy, &env->rng);
    agent->state.pos.y += rndf(-pos_xy, pos_xy, &env->rng);
    agent->state.pos.z += rndf(-pos_z, pos_z, &env->rng);
    agent->state.pos = clamp_world_vec3(agent->state.pos);
    Quat dq = race_reset_quat(rndf(-angle, angle, &env->rng),
                              rndf(-angle, angle, &env->rng),
                              rndf(-angle, angle, &env->rng));
    agent->state.quat = quat_mul(dq, agent->state.quat);
    quat_normalize(&agent->state.quat);
    agent->state.vel.x += rndf(-vel, vel, &env->rng);
    agent->state.vel.y += rndf(-vel, vel, &env->rng);
    agent->state.vel.z += rndf(-vel, vel, &env->rng);
    agent->state.omega.x += rndf(-omega, omega, &env->rng);
    agent->state.omega.y += rndf(-omega, omega, &env->rng);
    agent->state.omega.z += rndf(-omega, omega, &env->rng);
}

static inline bool race_try_isb_reset(DroneEnv* env, Drone* agent, int target_idx) {
    if (!(env->race_isb_enabled > 0.0f)) return false;
    if (env->race_track_mode >= 2.0f) return false;
    if (target_idx < 0 || target_idx >= DRONE_GATE_DEBUG_MAX) return false;
    if (rndf(0.0f, 1.0f, &env->rng) >= clampf(env->race_isb_prob, 0.0f, 1.0f)) return false;
    int count = agent->race_isb_count[target_idx];
    if (count <= 0) return false;
    if (count > DRONE_ISB_CAPACITY) count = DRONE_ISB_CAPACITY;
    int slot = (int)floorf(rndf(0.0f, (float)count, &env->rng));
    if (slot >= count) slot = count - 1;
    agent->state = agent->race_isb_states[target_idx][slot];
    agent->buffer_idx = target_idx;
    race_isb_perturb_state(env, agent);
    return true;
}

static inline void race_set_state_between(DroneEnv* env, Drone* agent, int target_idx,
                                          float t_min, float t_max, float lateral,
                                          float yaw_error, float forward_speed) {
    Target* prev = &agent->buffer[target_idx - 1];
    Target* target = &agent->buffer[target_idx];
    Vec3 segment = sub3(target->pos, prev->pos);
    float len = fmaxf(norm3(segment), 1e-3f);
    Vec3 dir = normalize3_or(segment, target->normal);
    Vec3 side = race_course_side(dir);
    float t = rndf(t_min, t_max, &env->rng);
    float side_offset = rndf(-lateral, lateral, &env->rng);
    float z_offset = rndf(-0.15f, 0.15f, &env->rng);
    agent->state.pos = clamp_world_vec3(add3(prev->pos, add3(
        scalmul3(dir, len * t),
        add3(scalmul3(side, side_offset), (Vec3){0.0f, 0.0f, z_offset}))));

    Vec3 to_target = sub3(target->pos, agent->state.pos);
    float yaw_to_target = atan2f(to_target.y, to_target.x);
    float roll = rndf(-0.04f, 0.04f, &env->rng);
    float pitch = rndf(-0.04f, 0.04f, &env->rng);
    agent->state.quat = race_reset_quat(yaw_to_target - yaw_error, pitch, roll);
    agent->state.vel = add3(scalmul3(dir, forward_speed),
                            scalmul3(side, rndf(-0.4f, 0.4f, &env->rng)));
    agent->state.vel.z += rndf(-0.12f, 0.12f, &env->rng);
    agent->state.omega = (Vec3){rndf(-0.15f, 0.15f, &env->rng),
                                rndf(-0.15f, 0.15f, &env->rng),
                                rndf(-0.20f, 0.20f, &env->rng)};
    agent->buffer_idx = target_idx;
}

static inline void race_set_state_segment_start(DroneEnv* env, Drone* agent, int start_idx,
                                                float t_min, float t_max, float lateral,
                                                float yaw_error, float forward_speed) {
    int n = agent->buffer_size;
    if (n <= 1) return;
    start_idx = ((start_idx % n) + n) % n;
    int target_idx = (start_idx + 1) % n;
    Target* start = &agent->buffer[start_idx];
    Target* target = &agent->buffer[target_idx];
    Vec3 segment = sub3(target->pos, start->pos);
    float len = fmaxf(norm3(segment), 1e-3f);
    Vec3 dir = normalize3_or(segment, target->normal);
    Vec3 side = race_course_side(dir);
    float t = rndf(t_min, t_max, &env->rng);
    float side_offset = rndf(-lateral, lateral, &env->rng);
    float z_offset = rndf(-0.12f, 0.12f, &env->rng);
    agent->state.pos = clamp_world_vec3(add3(start->pos, add3(
        scalmul3(dir, len * t),
        add3(scalmul3(side, side_offset), (Vec3){0.0f, 0.0f, z_offset}))));

    Vec3 to_target = sub3(target->pos, agent->state.pos);
    float yaw_to_target = atan2f(to_target.y, to_target.x);
    float roll = rndf(-0.04f, 0.04f, &env->rng);
    float pitch = rndf(-0.04f, 0.04f, &env->rng);
    agent->state.quat = race_reset_quat(yaw_to_target - yaw_error, pitch, roll);
    agent->state.vel = add3(scalmul3(dir, forward_speed),
                            scalmul3(side, rndf(-0.35f, 0.35f, &env->rng)));
    agent->state.vel.z += rndf(-0.10f, 0.10f, &env->rng);
    agent->state.omega = (Vec3){rndf(-0.15f, 0.15f, &env->rng),
                                rndf(-0.15f, 0.15f, &env->rng),
                                rndf(-0.20f, 0.20f, &env->rng)};
    agent->buffer_idx = target_idx;
}

static inline void race_set_state_before_first(DroneEnv* env, Drone* agent,
                                               float lateral, float yaw_error,
                                               float forward_speed) {
    Target* target = &agent->buffer[0];
    Vec3 dir = normalize3_or(target->normal, (Vec3){1.0f, 0.0f, 0.0f});
    Vec3 side = race_course_side(dir);
    float dist = rndf(2.5f, 4.5f, &env->rng);
    float side_offset = rndf(-lateral, lateral, &env->rng);
    float z_offset = rndf(-0.15f, 0.15f, &env->rng);
    agent->state.pos = clamp_world_vec3(add3(target->pos, add3(
        scalmul3(dir, -dist),
        add3(scalmul3(side, side_offset), (Vec3){0.0f, 0.0f, z_offset}))));

    Vec3 to_target = sub3(target->pos, agent->state.pos);
    float yaw_to_target = atan2f(to_target.y, to_target.x);
    float roll = rndf(-0.04f, 0.04f, &env->rng);
    float pitch = rndf(-0.04f, 0.04f, &env->rng);
    agent->state.quat = race_reset_quat(yaw_to_target - yaw_error, pitch, roll);
    agent->state.vel = add3(scalmul3(dir, forward_speed),
                            scalmul3(side, rndf(-0.25f, 0.25f, &env->rng)));
    agent->state.vel.z += rndf(-0.08f, 0.08f, &env->rng);
    agent->state.omega = (Vec3){rndf(-0.12f, 0.12f, &env->rng),
                                rndf(-0.12f, 0.12f, &env->rng),
                                rndf(-0.16f, 0.16f, &env->rng)};
    agent->buffer_idx = 0;
}

static inline void apply_race_reset_curriculum(DroneEnv* env, Drone* agent) {
    if (env->task != RACE) return;
    if (!(env->minimal_vision_spawn_visible_target > 0.0f)) return;
    if (!(env->minimal_vision_enabled > 0.0f)) return;
    if (env->num_agents != 1) return;
    if (agent->buffer_size <= 1) return;

    if (env->race_segment_mode >= 1.0f) {
        float yaw_error_frac = env->race_reset_yaw_error_frac > 0.0f
            ? env->race_reset_yaw_error_frac
            : 0.10f;
        float yaw_error = rndf(-yaw_error_frac * env->minimal_vision_fov,
                               yaw_error_frac * env->minimal_vision_fov, &env->rng);
        float t_min = env->race_reset_t_min > 0.0f ? env->race_reset_t_min : 0.02f;
        float t_max = env->race_reset_t_max > 0.0f ? env->race_reset_t_max : 0.30f;
        float lateral = env->race_reset_lateral > 0.0f ? env->race_reset_lateral : 0.35f;
        float speed_min = env->race_reset_speed_min > 0.0f ? env->race_reset_speed_min : 0.3f;
        float speed_max = env->race_reset_speed_max > 0.0f ? env->race_reset_speed_max : 1.4f;
        int target_idx = race_segment_target_idx(env, agent->buffer_size);
        if (race_try_isb_reset(env, agent, target_idx)) return;
        int start_idx = (target_idx - 1 + agent->buffer_size) % agent->buffer_size;
        race_set_state_segment_start(env, agent, start_idx, t_min, t_max,
                                     lateral, yaw_error, rndf(speed_min, speed_max, &env->rng));
        return;
    }

    float start_prob = env->race_reset_start_prob > 0.0f ? env->race_reset_start_prob : 0.90f;
    float mix = rndf(0.0f, 1.0f, &env->rng);
    if (mix < clampf(start_prob, 0.0f, 1.0f)) {
        if (env->race_track_mode >= 1.0f) {
            float yaw_error_frac = env->race_reset_yaw_error_frac > 0.0f
                ? env->race_reset_yaw_error_frac
                : 0.10f;
            float yaw_error = rndf(-yaw_error_frac * env->minimal_vision_fov,
                                   yaw_error_frac * env->minimal_vision_fov, &env->rng);
            float lateral = env->race_reset_lateral > 0.0f ? env->race_reset_lateral : 0.25f;
            float speed_min = env->race_reset_speed_min > 0.0f ? env->race_reset_speed_min : 0.3f;
            float speed_max = env->race_reset_speed_max > 0.0f ? env->race_reset_speed_max : 1.2f;
            race_set_state_before_first(env, agent, lateral, yaw_error,
                                        rndf(speed_min, speed_max, &env->rng));
        }
        agent->buffer_idx = 0;
        return;
    }

    int target_idx = race_random_target_idx(env, agent->buffer_size);
    float yaw_error_frac = env->race_reset_yaw_error_frac > 0.0f
        ? env->race_reset_yaw_error_frac
        : 0.10f;
    float yaw_error = rndf(-yaw_error_frac * env->minimal_vision_fov,
                           yaw_error_frac * env->minimal_vision_fov, &env->rng);
    float t_min = env->race_reset_t_min > 0.0f ? env->race_reset_t_min : 0.15f;
    float t_max = env->race_reset_t_max > 0.0f ? env->race_reset_t_max : 0.45f;
    float lateral = env->race_reset_lateral > 0.0f ? env->race_reset_lateral : 0.25f;
    float speed_min = env->race_reset_speed_min > 0.0f ? env->race_reset_speed_min : 0.3f;
    float speed_max = env->race_reset_speed_max > 0.0f ? env->race_reset_speed_max : 1.2f;
    race_set_state_between(env, agent, target_idx, t_min, t_max,
                           lateral, yaw_error, rndf(speed_min, speed_max, &env->rng));
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
    agent->ring_collision = 0.0f;
    agent->rings_passed = 0;
    agent->race_gate_bank = 0.0f;
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
    agent->target_in_fov_sum = 0.0f;
    agent->retina_rgb_mean_sum = 0.0f;
    agent->retina_rgb_std_sum = 0.0f;
    agent->retina_energy_sum = 0.0f;
    agent->retina_argmax_sum = 0.0f;
    agent->retina_argmax_left_count = 0.0f;
    agent->retina_argmax_center_count = 0.0f;
    agent->retina_argmax_right_count = 0.0f;
    agent->bearing_error_sum = 0.0f;
    agent->distance_to_target_sum = 0.0f;
    agent->retina_signal_vs_distance_sum = 0.0f;
    for (int gate = 0; gate < DRONE_GATE_DEBUG_MAX; gate++) {
        agent->gate_time[gate] = 0.0f;
        agent->gate_target_in_fov[gate] = 0.0f;
        agent->gate_bearing_error[gate] = 0.0f;
        agent->gate_distance_to_target[gate] = 0.0f;
        agent->gate_pass_count[gate] = 0.0f;
        agent->gate_collision_count[gate] = 0.0f;
        agent->gate_timeout_count[gate] = 0.0f;
        agent->gate_oob_count[gate] = 0.0f;
    }
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
        set_visible_race_course(env, agent);
        apply_race_reset_curriculum(env, agent);
        set_target(&env->rng, env->task, env->agents, i, env->num_agents, env->hover_target_dist);
        set_minimal_vision_visible_target(env, agent);
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

        bool oob = norm3(sub3(agent->target->pos, agent->state.pos)) > env->oob_radius;
        bool timeout = (agent->episode_length >= HORIZON);

        int ring_result = 0;
        bool lap_complete = false;
        if (env->task == RACE) {
            int current_gate_idx = race_clamped_gate_idx(agent);
            ring_result = check_ring(agent, agent->target);
            if (ring_result == 1) {
                agent->rings_passed += 1;
                agent->race_gate_bank += 1.0f;
                if (current_gate_idx >= 0 && current_gate_idx < DRONE_GATE_DEBUG_MAX) {
                    agent->gate_pass_count[current_gate_idx] += 1.0f;
                }
                if (agent->buffer_size > 0) {
                    int next_target = (current_gate_idx + 1) % agent->buffer_size;
                    race_isb_push(env, agent, next_target, race_pass_margin(agent, current_gate_idx));
                }
                lap_complete = env->race_segment_mode >= 1.0f
                    || (agent->buffer_size > 0 && agent->buffer_idx == agent->buffer_size - 1);
            } else if (ring_result == -1) {
                agent->ring_collision += 1.0f;
                agent->collisions += 1.0f;
                if (current_gate_idx >= 0 && current_gate_idx < DRONE_GATE_DEBUG_MAX) {
                    agent->gate_collision_count[current_gate_idx] += 1.0f;
                }
            }
        }

        float curr = hover_potential(agent, env->hover_dist, env->hover_omega, env->hover_vel);
        float prev_dist = norm3(sub3(agent->target->pos, agent->prev_pos));
        float curr_dist = norm3(sub3(agent->target->pos, agent->state.pos));
        float omega = norm3(agent->state.omega);
        float speed = norm3(agent->state.vel);
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
            if (ring_result == 1) r_terminal += 0.2f;
            else if (ring_result == -1) r_terminal -= 2.0f;
            if (oob) {
                r_terminal += -10.0f
                            - agent->race_gate_bank
                            - 0.05f * speed * speed;
                agent->race_gate_bank = 0.0f;
            } else if (lap_complete || timeout) {
                r_terminal += agent->race_gate_bank;
                agent->race_gate_bank = 0.0f;
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
        agent->ema_vel = 0.99f * agent->ema_vel + 0.01f * speed;
        agent->ema_omega = 0.99f * agent->ema_omega + 0.01f * omega;
        agent->ema_omega_x = 0.99f * agent->ema_omega_x + 0.01f * fabsf(agent->state.omega.x);
        agent->ema_omega_y = 0.99f * agent->ema_omega_y + 0.01f * fabsf(agent->state.omega.y);
        agent->ema_omega_z = 0.99f * agent->ema_omega_z + 0.01f * fabsf(agent->state.omega.z);
        record_retina_diagnostics(env, agent);
        record_step_metrics(agent, raw_actions, r_dist, r_hover, r_shaping, r_omega,
                            r_omega_xy, r_omega_z, r_terminal,
                            action_delta_mean, reset_action_jump);
        agent->episode_return += reward;
        env->rewards[i] = reward;

        if (env->task == RACE && ring_result == 1 && !lap_complete) {
            agent->buffer_idx = agent->buffer_idx + 1;
            set_target(&env->rng, env->task, env->agents, i, env->num_agents, env->hover_target_dist);
            finalize_reset_potential(env, agent);
        }

        bool reset = oob || timeout || lap_complete;
        env->terminals[i] = reset ? 1.0f : 0.0f;

        if (reset) {
            add_log(env, i, oob, timeout, lap_complete);
            reset_agent(env, agent, i);
            set_visible_race_course(env, agent);
            apply_race_reset_curriculum(env, agent);
            set_target(&env->rng, env->task, env->agents, i, env->num_agents, env->hover_target_dist);
            set_minimal_vision_visible_target(env, agent);
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
