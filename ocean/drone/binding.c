#include "drone.h"
#include "render.h"

#define OBS_SIZE 23
#define NUM_ATNS 4
#define ACT_SIZES {1, 1, 1, 1}
#define OBS_TENSOR_T FloatTensor

#define Env DroneEnv
#include "vecenv.h"

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
    env->action_scale = dict_get(kwargs, "action_scale")->value;
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
