#include "drone.h"
#include "render.h"

#define OBS_SIZE 23
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
    init(env);
}

void my_log(Log* log, Dict* out) {
    dict_set(out, "perf", log->perf);
    dict_set(out, "score", log->score);
    dict_set(out, "rings_passed", log->rings_passed);
    dict_set(out, "ring_collisions", log->ring_collision);
    dict_set(out, "collisions", log->collisions);
    dict_set(out, "oob", log->oob);
    dict_set(out, "timeout", log->timeout);
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
