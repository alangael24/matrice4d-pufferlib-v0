#include "drone.h"
#include "render.h"

#include <stdio.h>

#define OBS_SIZE DRONE_OBS_SIZE
#define NUM_ATNS 4
#define ACT_SIZES {1, 1, 1, 1}
#define OBS_TENSOR_T FloatTensor

#define Env DroneEnv
#include "vecenv.h"

static inline float dict_get_default(Dict* dict, const char* key, float fallback) {
    DictItem* item = dict_get_unsafe(dict, key);
    return item == NULL ? fallback : item->value;
}

void my_init(Env* env, Dict* kwargs) {
    env->num_agents = (int)dict_get(kwargs, "num_drones")->value;
    env->task = (int)dict_get(kwargs, "task")->value;
    env->max_rings = (int)dict_get(kwargs, "max_rings")->value;
    env->alpha_dist = dict_get(kwargs, "alpha_dist")->value;
    env->alpha_hover = dict_get(kwargs, "alpha_hover")->value;
    env->alpha_shaping = dict_get(kwargs, "alpha_shaping")->value;
    env->alpha_omega = dict_get(kwargs, "alpha_omega")->value;
    env->alpha_omega_xy = dict_get(kwargs, "alpha_omega_xy")->value;
    env->alpha_omega_z = dict_get(kwargs, "alpha_omega_z")->value;
    env->alpha_omega_z_sq = dict_get(kwargs, "alpha_omega_z_sq")->value;
    env->alpha_omega_z_mult = dict_get(kwargs, "alpha_omega_z_mult")->value;
    env->alpha_action_delta = dict_get_default(kwargs, "alpha_action_delta", 0.0f);
    env->alpha_reset_action_delta = dict_get_default(kwargs, "alpha_reset_action_delta", 0.0f);
    env->reset_action_interval = (int)dict_get_default(kwargs, "reset_action_interval", 0.0f);
    env->hover_target_dist = dict_get(kwargs, "hover_target_dist")->value;
    env->oob_radius = dict_get(kwargs, "oob_radius")->value;
    env->hover_dist = dict_get(kwargs, "hover_dist")->value;
    env->hover_omega = dict_get(kwargs, "hover_omega")->value;
    env->hover_vel = dict_get(kwargs, "hover_vel")->value;
    env->domain_randomization = dict_get(kwargs, "domain_randomization")->value;
    env->dr_mass = dict_get(kwargs, "dr_mass")->value;
    env->dr_inertia = dict_get(kwargs, "dr_inertia")->value;
    env->dr_k_thrust = dict_get(kwargs, "dr_k_thrust")->value;
    env->dr_linear_drag = dict_get(kwargs, "dr_linear_drag")->value;
    env->dr_yaw_drag = dict_get(kwargs, "dr_yaw_drag")->value;
    env->dr_motor_lag = dict_get(kwargs, "dr_motor_lag")->value;
    env->dr_com_xy = dict_get(kwargs, "dr_com_xy")->value;
    env->dr_com_z = dict_get(kwargs, "dr_com_z")->value;
    env->dr_authority_gated = dict_get_default(kwargs, "dr_authority_gated", 0.0f);
    env->dr_usable_t2w_min = dict_get_default(kwargs, "dr_usable_t2w_min", 0.0f);
    env->dr_usable_t2w_max = dict_get_default(kwargs, "dr_usable_t2w_max", 0.0f);
    env->dr_mass_min = dict_get_default(kwargs, "dr_mass_min", 1.0f);
    env->dr_mass_max = dict_get_default(kwargs, "dr_mass_max", 1.0f);
    env->dr_inertia_min = dict_get_default(kwargs, "dr_inertia_min", 1.0f);
    env->dr_inertia_max = dict_get_default(kwargs, "dr_inertia_max", 1.0f);
    env->dr_motor_thrust_min = dict_get_default(kwargs, "dr_motor_thrust_min", 1.0f);
    env->dr_motor_thrust_max = dict_get_default(kwargs, "dr_motor_thrust_max", 1.0f);
    env->dr_motor_tau_min = dict_get_default(kwargs, "dr_motor_tau_min", BASE_K_MOT);
    env->dr_motor_tau_max = dict_get_default(kwargs, "dr_motor_tau_max", BASE_K_MOT);
    env->dr_yaw_torque_min = dict_get_default(kwargs, "dr_yaw_torque_min", 1.0f);
    env->dr_yaw_torque_max = dict_get_default(kwargs, "dr_yaw_torque_max", 1.0f);
    env->dr_linear_drag_min = dict_get_default(kwargs, "dr_linear_drag_min", 1.0f);
    env->dr_linear_drag_max = dict_get_default(kwargs, "dr_linear_drag_max", 1.0f);
    env->dr_angular_damping_min = dict_get_default(kwargs, "dr_angular_damping_min", 1.0f);
    env->dr_angular_damping_max = dict_get_default(kwargs, "dr_angular_damping_max", 1.0f);
    env->dr_profile_mix = dict_get_default(kwargs, "dr_profile_mix", 0.0f);
    env->adr_enabled = dict_get_default(kwargs, "adr_enabled", 0.0f);
    env->adr_mode = dict_get_default(kwargs, "adr_mode", 1.0f);
    env->adr_probe_prob = dict_get_default(kwargs, "adr_probe_prob", 0.02f);
    env->adr_success_threshold = dict_get_default(kwargs, "adr_success_threshold", 0.90f);
    env->adr_contract_threshold = dict_get_default(kwargs, "adr_contract_threshold", 0.50f);
    env->adr_step = dict_get_default(kwargs, "adr_step", 0.02f);
    env->adr_eval_episodes = dict_get_default(kwargs, "adr_eval_episodes", 64.0f);
    env->adr_init_usable_t2w_min = dict_get_default(kwargs, "adr_init_usable_t2w_min", 2.40f);
    env->adr_init_usable_t2w_max = dict_get_default(kwargs, "adr_init_usable_t2w_max", 3.40f);
    env->adr_init_mass_min = dict_get_default(kwargs, "adr_init_mass_min", 0.90f);
    env->adr_init_mass_max = dict_get_default(kwargs, "adr_init_mass_max", 1.10f);
    env->adr_init_inertia_min = dict_get_default(kwargs, "adr_init_inertia_min", 0.80f);
    env->adr_init_inertia_max = dict_get_default(kwargs, "adr_init_inertia_max", 1.30f);
    env->adr_init_motor_thrust_min = dict_get_default(kwargs, "adr_init_motor_thrust_min", 0.92f);
    env->adr_init_motor_thrust_max = dict_get_default(kwargs, "adr_init_motor_thrust_max", 1.08f);
    env->adr_init_motor_tau_min = dict_get_default(kwargs, "adr_init_motor_tau_min", 0.08f);
    env->adr_init_motor_tau_max = dict_get_default(kwargs, "adr_init_motor_tau_max", 0.18f);
    env->adr_init_com_xy = dict_get_default(kwargs, "adr_init_com_xy", 0.012f);
    env->pal_probe_prob = dict_get_default(kwargs, "pal_probe_prob", 0.0f);
    env->pal_probe_steps = (int)dict_get_default(kwargs, "pal_probe_steps", 0.0f);
    env->pal_probe_amp = dict_get_default(kwargs, "pal_probe_amp", 0.0f);
    env->action_scale = dict_get(kwargs, "action_scale")->value;
    env->action_mode = (int)dict_get_default(kwargs, "action_mode", (float)M4D_ACTION_HOVER_TRIM);
    env->normalized_thrust_min = dict_get_default(kwargs, "normalized_thrust_min", 0.0f);
    env->normalized_thrust_max = dict_get_default(kwargs, "normalized_thrust_max", 1.0f);
    env->reset_pos_scale = dict_get(kwargs, "reset_pos_scale")->value;
    env->reset_yaw_range = dict_get(kwargs, "reset_yaw_range")->value;
    env->reset_vel_max = dict_get(kwargs, "reset_vel_max")->value;
    env->action_latency = dict_get(kwargs, "action_latency")->value;
    env->sensor_noise = dict_get(kwargs, "sensor_noise")->value;
    env->minimal_vision_enabled = dict_get_default(kwargs, "minimal_vision_enabled", 0.0f);
    env->minimal_vision_only = dict_get_default(kwargs, "minimal_vision_only", 0.0f);
    env->minimal_vision_mask_target = dict_get_default(kwargs, "minimal_vision_mask_target", 0.0f);
    env->minimal_vision_fov = dict_get_default(kwargs, "minimal_vision_fov", 2.0943951f);
    env->minimal_vision_vfov = dict_get_default(kwargs, "minimal_vision_vfov", 1.3962634f);
    env->minimal_vision_sigma = dict_get_default(kwargs, "minimal_vision_sigma", 0.45f);
    env->minimal_vision_depth_gain = dict_get_default(kwargs, "minimal_vision_depth_gain", 0.08f);
    env->minimal_vision_noise = dict_get_default(kwargs, "minimal_vision_noise", 0.0f);
    env->minimal_vision_distractors = dict_get_default(kwargs, "minimal_vision_distractors", 0.0f);
    env->minimal_vision_spawn_visible_target = dict_get_default(kwargs, "minimal_vision_spawn_visible_target", 0.0f);
    env->minimal_vision_gate_mask = dict_get_default(kwargs, "minimal_vision_gate_mask", 0.0f);
    env->race_track_mode = dict_get_default(kwargs, "race_track_mode", 0.0f);
    env->race_segment_mode = dict_get_default(kwargs, "race_segment_mode", 0.0f);
    env->race_isb_enabled = dict_get_default(kwargs, "race_isb_enabled", 0.0f);
    env->race_isb_prob = dict_get_default(kwargs, "race_isb_prob", 0.0f);
    env->race_isb_margin = dict_get_default(kwargs, "race_isb_margin", 0.8f);
    env->race_isb_pos_xy = dict_get_default(kwargs, "race_isb_pos_xy", 0.45f);
    env->race_isb_z = dict_get_default(kwargs, "race_isb_z", 0.25f);
    env->race_isb_angle = dict_get_default(kwargs, "race_isb_angle", 0.18f);
    env->race_isb_vel = dict_get_default(kwargs, "race_isb_vel", 0.60f);
    env->race_isb_omega = dict_get_default(kwargs, "race_isb_omega", 0.60f);
    env->race_hard_gate_idx = dict_get_default(kwargs, "race_hard_gate_idx", -1.0f);
    env->race_hard_gate_prob = dict_get_default(kwargs, "race_hard_gate_prob", 0.0f);
    env->race_course_yaw_delta = dict_get_default(kwargs, "race_course_yaw_delta", 0.0f);
    env->race_course_pitch_delta = dict_get_default(kwargs, "race_course_pitch_delta", 0.0f);
    env->race_course_pitch_limit = dict_get_default(kwargs, "race_course_pitch_limit", 0.0f);
    env->race_course_spacing_min = dict_get_default(kwargs, "race_course_spacing_min", 0.0f);
    env->race_course_spacing_max = dict_get_default(kwargs, "race_course_spacing_max", 0.0f);
    env->race_course_dz_max = dict_get_default(kwargs, "race_course_dz_max", 0.0f);
    env->race_reset_start_prob = dict_get_default(kwargs, "race_reset_start_prob", 0.0f);
    env->race_reset_t_min = dict_get_default(kwargs, "race_reset_t_min", 0.0f);
    env->race_reset_t_max = dict_get_default(kwargs, "race_reset_t_max", 0.0f);
    env->race_reset_lateral = dict_get_default(kwargs, "race_reset_lateral", 0.0f);
    env->race_reset_yaw_error_frac = dict_get_default(kwargs, "race_reset_yaw_error_frac", 0.0f);
    env->race_reset_speed_min = dict_get_default(kwargs, "race_reset_speed_min", 0.0f);
    env->race_reset_speed_max = dict_get_default(kwargs, "race_reset_speed_max", 0.0f);
    init(env);
}

static inline void set_gate_debug_log(Dict* out, const Log* log, int gate,
                                      const char* time_key, const char* fov_key,
                                      const char* bearing_key, const char* dist_key,
                                      const char* pass_key, const char* collision_key,
                                      const char* timeout_key, const char* oob_key,
                                      float total_gate_time) {
    float gate_time = log->gate_time[gate];
    float gate_inv = gate_time > 1e-6f ? 1.0f / gate_time : 0.0f;
    float total_inv = total_gate_time > 1e-6f ? 1.0f / total_gate_time : 0.0f;
    dict_set(out, time_key, gate_time * total_inv);
    dict_set(out, fov_key, log->gate_target_in_fov[gate] * gate_inv);
    dict_set(out, bearing_key, log->gate_bearing_error[gate] * gate_inv);
    dict_set(out, dist_key, log->gate_distance_to_target[gate] * gate_inv);
    dict_set(out, pass_key, log->gate_pass_count[gate]);
    dict_set(out, collision_key, log->gate_collision_count[gate]);
    dict_set(out, timeout_key, log->gate_timeout_count[gate]);
    dict_set(out, oob_key, log->gate_oob_count[gate]);
}

#define SET_GATE_DEBUG_LOG(G) \
    set_gate_debug_log(out, log, G, \
        "g" #G "_time_frac", "g" #G "_fov", "g" #G "_bearing", "g" #G "_dist", \
        "g" #G "_pass", "g" #G "_collision", "g" #G "_timeout", "g" #G "_oob", \
        total_gate_time)

void my_log(Log* log, Dict* out) {
    dict_set(out, "perf", log->perf);
    dict_set(out, "score", log->score);
    dict_set(out, "rings_passed", log->rings_passed);
    dict_set(out, "ring_collisions", log->ring_collision);
    dict_set(out, "collisions", log->collisions);
    dict_set(out, "oob", log->oob);
    dict_set(out, "timeout", log->timeout);
    dict_set(out, "lap_complete", log->lap_complete);
    dict_set(out, "episode_return", log->episode_return);
    dict_set(out, "episode_length", log->episode_length);
    dict_set(out, "ema_dist", log->ema_dist);
    dict_set(out, "ema_vel", log->ema_vel);
    dict_set(out, "ema_omega", log->ema_omega);
    dict_set(out, "action_saturation_frac", log->action_saturation_frac);
    dict_set(out, "mean_abs_delta_action", log->mean_abs_delta_action);
    dict_set(out, "target_in_fov_frac", log->target_in_fov_frac);
    dict_set(out, "retina_energy", log->retina_energy);
    dict_set(out, "bearing_error_to_target", log->bearing_error_to_target);
    dict_set(out, "distance_to_target", log->distance_to_target);
    {
        float oob_diag_inv = log->oob_diag_count > 1e-6f ? 1.0f / log->oob_diag_count : 0.0f;
        dict_set(out, "oob_diag_count", log->oob_diag_count);
        dict_set(out, "gate_index_at_oob", log->gate_index_at_oob * oob_diag_inv);
        dict_set(out, "distance_from_track_centerline", log->distance_from_track_centerline * oob_diag_inv);
        float total_gate_time = 0.0f;
        for (int gate = 0; gate < DRONE_GATE_DEBUG_MAX; gate++) {
            total_gate_time += log->gate_time[gate];
        }
        SET_GATE_DEBUG_LOG(0);
        SET_GATE_DEBUG_LOG(1);
        SET_GATE_DEBUG_LOG(2);
        SET_GATE_DEBUG_LOG(3);
        SET_GATE_DEBUG_LOG(4);
        SET_GATE_DEBUG_LOG(5);
        SET_GATE_DEBUG_LOG(6);
        SET_GATE_DEBUG_LOG(7);
    }
    return;

    dict_set(out, "perf", log->perf);
    dict_set(out, "score", log->score);
    dict_set(out, "rings_passed", log->rings_passed);
    dict_set(out, "ring_collisions", log->ring_collision);
    dict_set(out, "collisions", log->collisions);
    dict_set(out, "oob", log->oob);
    dict_set(out, "timeout", log->timeout);
    dict_set(out, "lap_complete", log->lap_complete);
    dict_set(out, "episode_return", log->episode_return);
    dict_set(out, "episode_length", log->episode_length);
    dict_set(out, "ema_dist", log->ema_dist);
    dict_set(out, "ema_vel", log->ema_vel);
    dict_set(out, "ema_omega", log->ema_omega);
    dict_set(out, "ema_omega_x", log->ema_omega_x);
    dict_set(out, "ema_omega_y", log->ema_omega_y);
    dict_set(out, "ema_omega_z", log->ema_omega_z);
    dict_set(out, "mean_abs_action", log->mean_abs_action);
    dict_set(out, "mean_abs_action_clipped", log->mean_abs_action_clipped);
    dict_set(out, "max_abs_action", log->max_abs_action);
    dict_set(out, "action_saturation_frac", log->action_saturation_frac);
    dict_set(out, "mean_abs_delta_action", log->mean_abs_delta_action);
    dict_set(out, "reset_action_jump_mean", log->reset_action_jump_mean);
    dict_set(out, "motor_clip_low_frac", log->motor_clip_low_frac);
    dict_set(out, "motor_clip_high_frac", log->motor_clip_high_frac);
    dict_set(out, "hover_trim_rpm_mean", log->hover_trim_rpm_mean);
    dict_set(out, "hover_trim_rpm_max", log->hover_trim_rpm_max);
    dict_set(out, "hover_trim_rpm_frac_of_max", log->hover_trim_rpm_frac_of_max);
    dict_set(out, "mean_rpm_FL", log->mean_rpm_FL);
    dict_set(out, "mean_rpm_FR", log->mean_rpm_FR);
    dict_set(out, "mean_rpm_RL", log->mean_rpm_RL);
    dict_set(out, "mean_rpm_RR", log->mean_rpm_RR);
    dict_set(out, "target_in_fov_frac", log->target_in_fov_frac);
    dict_set(out, "retina_rgb_mean", log->retina_rgb_mean);
    dict_set(out, "retina_rgb_std", log->retina_rgb_std);
    dict_set(out, "retina_energy", log->retina_energy);
    dict_set(out, "retina_left_center_right_argmax", log->retina_left_center_right_argmax);
    dict_set(out, "retina_argmax_left_frac", log->retina_argmax_left_frac);
    dict_set(out, "retina_argmax_center_frac", log->retina_argmax_center_frac);
    dict_set(out, "retina_argmax_right_frac", log->retina_argmax_right_frac);
    dict_set(out, "bearing_error_to_target", log->bearing_error_to_target);
    dict_set(out, "distance_to_target", log->distance_to_target);
    dict_set(out, "retina_signal_vs_distance", log->retina_signal_vs_distance);
    dict_set(out, "r_dist", log->r_dist);
    dict_set(out, "r_hover", log->r_hover);
    dict_set(out, "r_shaping", log->r_shaping);
    dict_set(out, "r_omega", log->r_omega);
    dict_set(out, "r_omega_xy", log->r_omega_xy);
    dict_set(out, "r_omega_z", log->r_omega_z);
    dict_set(out, "r_terminal", log->r_terminal);
    dict_set(out, "mass_mult_mean", log->mass_mult_mean);
    dict_set(out, "ixx_mult_mean", log->ixx_mult_mean);
    dict_set(out, "iyy_mult_mean", log->iyy_mult_mean);
    dict_set(out, "izz_mult_mean", log->izz_mult_mean);
    dict_set(out, "k_thrust_mult_mean", log->k_thrust_mult_mean);
    dict_set(out, "k_thrust_mult_min", log->k_thrust_mult_min);
    dict_set(out, "k_thrust_mult_max", log->k_thrust_mult_max);
    dict_set(out, "linear_drag_mult_mean", log->linear_drag_mult_mean);
    dict_set(out, "yaw_drag_mult_mean", log->yaw_drag_mult_mean);
    dict_set(out, "motor_lag_mult_mean", log->motor_lag_mult_mean);
    dict_set(out, "com_x_mean", log->com_x_mean);
    dict_set(out, "com_y_mean", log->com_y_mean);
    dict_set(out, "com_z_mean", log->com_z_mean);
    float oob_diag_inv = log->oob_diag_count > 1e-6f ? 1.0f / log->oob_diag_count : 0.0f;
    dict_set(out, "oob_diag_count", log->oob_diag_count);
    dict_set(out, "gate_index_at_oob", log->gate_index_at_oob * oob_diag_inv);
    dict_set(out, "position_norm_at_oob", log->position_norm_at_oob * oob_diag_inv);
    dict_set(out, "target_gate_position_norm", log->target_gate_position_norm * oob_diag_inv);
    dict_set(out, "next_gate_position_norm", log->next_gate_position_norm * oob_diag_inv);
    dict_set(out, "distance_from_track_centerline", log->distance_from_track_centerline * oob_diag_inv);
    float total_gate_time = 0.0f;
    for (int gate = 0; gate < DRONE_GATE_DEBUG_MAX; gate++) {
        total_gate_time += log->gate_time[gate];
    }
    SET_GATE_DEBUG_LOG(0);
    SET_GATE_DEBUG_LOG(1);
    SET_GATE_DEBUG_LOG(2);
    SET_GATE_DEBUG_LOG(3);
    SET_GATE_DEBUG_LOG(4);
    SET_GATE_DEBUG_LOG(5);
    SET_GATE_DEBUG_LOG(6);
    SET_GATE_DEBUG_LOG(7);
}

typedef struct DroneDebugState {
    float pos[3];
    float vel[3];
    float quat[4];
    float omega[3];
    float rpms[4];
    float target_pos[3];
    float target_normal[3];
    float prev_pos[3];
    float prev_potential;
    float episode_return;
    int episode_length;
    float mass;
    float ixx;
    float iyy;
    float izz;
    float k_thrust;
    float k_drag;
    float b_drag;
    float k_mot;
    float action_scale;
    float motor_x[4];
    float motor_y[4];
    float yaw_sign[4];
    float hover_trim[4];
} DroneDebugState;

static int drone_debug_find_agent(StaticVec* vec, int agent_idx, DroneEnv** out_env, int* out_local) {
    if (vec == NULL || vec->envs == NULL || agent_idx < 0 || agent_idx >= vec->total_agents) {
        return 0;
    }

    DroneEnv* envs = (DroneEnv*)vec->envs;
    int base = 0;
    for (int e = 0; e < vec->size; e++) {
        DroneEnv* env = &envs[e];
        if (agent_idx < base + env->num_agents) {
            *out_env = env;
            *out_local = agent_idx - base;
            return 1;
        }
        base += env->num_agents;
    }
    return 0;
}

int drone_debug_cpu_state(StaticVec* vec, int agent_idx, DroneDebugState* out) {
    DroneEnv* env = NULL;
    int local = 0;
    if (out == NULL || !drone_debug_find_agent(vec, agent_idx, &env, &local)) {
        return 0;
    }

    Drone* d = &env->agents[local];
    memset(out, 0, sizeof(*out));
    out->pos[0] = d->state.pos.x;
    out->pos[1] = d->state.pos.y;
    out->pos[2] = d->state.pos.z;
    out->vel[0] = d->state.vel.x;
    out->vel[1] = d->state.vel.y;
    out->vel[2] = d->state.vel.z;
    out->quat[0] = d->state.quat.w;
    out->quat[1] = d->state.quat.x;
    out->quat[2] = d->state.quat.y;
    out->quat[3] = d->state.quat.z;
    out->omega[0] = d->state.omega.x;
    out->omega[1] = d->state.omega.y;
    out->omega[2] = d->state.omega.z;
    for (int i = 0; i < 4; i++) out->rpms[i] = d->state.rpms[i];
    if (d->target != NULL) {
        out->target_pos[0] = d->target->pos.x;
        out->target_pos[1] = d->target->pos.y;
        out->target_pos[2] = d->target->pos.z;
        out->target_normal[0] = d->target->normal.x;
        out->target_normal[1] = d->target->normal.y;
        out->target_normal[2] = d->target->normal.z;
    }
    out->prev_pos[0] = d->prev_pos.x;
    out->prev_pos[1] = d->prev_pos.y;
    out->prev_pos[2] = d->prev_pos.z;
    out->prev_potential = d->prev_potential;
    out->episode_return = d->episode_return;
    out->episode_length = d->episode_length;
    out->mass = d->params.mass;
    out->ixx = d->params.ixx;
    out->iyy = d->params.iyy;
    out->izz = d->params.izz;
    out->k_thrust = d->params.k_thrust;
    out->k_drag = d->params.k_drag;
    out->b_drag = d->params.b_drag;
    out->k_mot = d->params.k_mot;
    out->action_scale = d->params.action_scale;
    for (int i = 0; i < 4; i++) {
        out->motor_x[i] = d->params.motor_x[i];
        out->motor_y[i] = d->params.motor_y[i];
        out->yaw_sign[i] = d->params.yaw_sign[i];
    }
    hover_trim_thrusts(&d->params, out->hover_trim);
    return 1;
}
