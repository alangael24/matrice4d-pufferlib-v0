#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cuda_bf16.h>

#ifdef PRECISION_FLOAT
typedef float precision_t;
#else
typedef __nv_bfloat16 precision_t;
#endif

#include <math.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "vecenv.h"

#define DRONE_STATE_OBS_SIZE 23
#ifndef DRONE_MINIMAL_VISION_WIDTH
#define DRONE_MINIMAL_VISION_WIDTH 3
#endif
#ifndef DRONE_MINIMAL_VISION_HEIGHT
#define DRONE_MINIMAL_VISION_HEIGHT 1
#endif
#define DRONE_MINIMAL_VISION_PIXELS (DRONE_MINIMAL_VISION_WIDTH * DRONE_MINIMAL_VISION_HEIGHT)
#define DRONE_MINIMAL_VISION_CHANNELS 3
#define DRONE_MINIMAL_VISION_OBS_SIZE (DRONE_MINIMAL_VISION_PIXELS * DRONE_MINIMAL_VISION_CHANNELS)
#define DRONE_OBS_SIZE (DRONE_STATE_OBS_SIZE + DRONE_MINIMAL_VISION_OBS_SIZE)
#define DRONE_NUM_ATNS 4
#define DRONE_HORIZON 1024

#define DRONE_DT 0.002f
#define DRONE_ACTION_SUBSTEPS 5
#define DRONE_ACTION_DT (DRONE_DT * (float)DRONE_ACTION_SUBSTEPS)
#define DRONE_MAX_ACTION_LATENCY_STEPS 8

#define DRONE_GRID_X 120.0f
#define DRONE_GRID_Y 120.0f
#define DRONE_GRID_Z 60.0f
#define DRONE_MARGIN_X (DRONE_GRID_X - 1.0f)
#define DRONE_MARGIN_Y (DRONE_GRID_Y - 1.0f)
#define DRONE_MARGIN_Z (DRONE_GRID_Z - 1.0f)
#define DRONE_PI 3.14159265358979323846f

#define BASE_MASS 1.850f
#define BASE_IXX 4.0250e-2f
#define BASE_IYY 3.9390e-2f
#define BASE_IZZ 7.0230e-2f
#define BASE_K_THRUST 1.4863332e-7f
#define BASE_K_DRAG 0.020f
#define BASE_GRAVITY 9.81f
#define BASE_MAX_RPM 8500.0f
#define BASE_K_MOT 0.20f
#define BASE_K_ANG_DAMP 0.010f
#define BASE_B_DRAG 0.150f
#define BASE_MAX_VEL 21.0f
#define BASE_MAX_OMEGA 3.4906585f

#define BASE_MOTOR_FL_X -0.1915f
#define BASE_MOTOR_FL_Y  0.1708f
#define BASE_MOTOR_FR_X  0.1915f
#define BASE_MOTOR_FR_Y  0.1708f
#define BASE_MOTOR_RL_X -0.1715f
#define BASE_MOTOR_RL_Y -0.1708f
#define BASE_MOTOR_RR_X  0.1715f
#define BASE_MOTOR_RR_Y -0.1708f
#define BASE_YAW_SIGN_FL  1.0f
#define BASE_YAW_SIGN_FR -1.0f
#define BASE_YAW_SIGN_RL -1.0f
#define BASE_YAW_SIGN_RR  1.0f

#define M4D_ACTION_HOVER_TRIM 0
#define M4D_ACTION_NORMALIZED_THRUST 1

#define DRONE_ADR_MODE_AUTHORITY 1
#define DRONE_ADR_MODE_LEGACY_PAPER 2

#define DRONE_ADR_PARAM_COUNT 6
#define DRONE_ADR_EDGE_COUNT (DRONE_ADR_PARAM_COUNT * 2)
#define DRONE_ADR_LEGACY_PARAM_COUNT 8
#define DRONE_ADR_LEGACY_EDGE_COUNT (DRONE_ADR_LEGACY_PARAM_COUNT * 2)
#define DRONE_ADR_SIDE_LOW 0
#define DRONE_ADR_SIDE_HIGH 1
#define DRONE_TASK_HOVER 1
#define DRONE_TASK_RACE 7
#define DRONE_CUDA_MAX_RINGS 16
#define DRONE_GATE_DEBUG_MAX 8
#define DRONE_ISB_CAPACITY 10
#define DRONE_RING_RADIUS 2.0f
#define DRONE_ADR_USABLE_T2W 0
#define DRONE_ADR_MASS 1
#define DRONE_ADR_INERTIA 2
#define DRONE_ADR_MOTOR_THRUST 3
#define DRONE_ADR_MOTOR_TAU 4
#define DRONE_ADR_COM_XY 5
#define DRONE_ADR_LEGACY_MASS 0
#define DRONE_ADR_LEGACY_INERTIA 1
#define DRONE_ADR_LEGACY_K_THRUST 2
#define DRONE_ADR_LEGACY_LINEAR_DRAG 3
#define DRONE_ADR_LEGACY_YAW_DRAG 4
#define DRONE_ADR_LEGACY_MOTOR_LAG 5
#define DRONE_ADR_LEGACY_COM_XY 6
#define DRONE_ADR_LEGACY_COM_Z 7

struct DroneCudaAdrState {
    int enabled;
    int mode;
    float usable_t2w_min, usable_t2w_max;
    float mass_min, mass_max;
    float inertia_min, inertia_max;
    float motor_thrust_min, motor_thrust_max;
    float motor_tau_min, motor_tau_max;
    float com_xy;
    float legacy_mass_min, legacy_mass_max;
    float legacy_inertia_min, legacy_inertia_max;
    float legacy_k_thrust_min, legacy_k_thrust_max;
    float legacy_linear_drag_min, legacy_linear_drag_max;
    float legacy_yaw_drag_min, legacy_yaw_drag_max;
    float legacy_motor_lag_min, legacy_motor_lag_max;
    float legacy_com_xy;
    float legacy_com_z;
    unsigned int counts[DRONE_ADR_EDGE_COUNT];
    unsigned int successes[DRONE_ADR_EDGE_COUNT];
    unsigned int updates[DRONE_ADR_EDGE_COUNT];
    unsigned int updating[DRONE_ADR_EDGE_COUNT];
    unsigned int legacy_counts[DRONE_ADR_LEGACY_EDGE_COUNT];
    float legacy_perf_sum[DRONE_ADR_LEGACY_EDGE_COUNT];
    unsigned int legacy_updates[DRONE_ADR_LEGACY_EDGE_COUNT];
    unsigned int legacy_updating[DRONE_ADR_LEGACY_EDGE_COUNT];
};

struct DroneCudaParams {
    float mass, ixx, iyy, izz;
    float motor_x[4], motor_y[4], yaw_sign[4];
    float motor_thrust_scale[4], motor_tau[4], yaw_torque_scale[4];
    float k_thrust, k_ang_damp, k_drag, b_drag, gravity;
    float max_rpm, max_vel, max_omega, k_mot, action_scale;
    int action_mode;
    float normalized_thrust_min, normalized_thrust_max;
    float com_x, com_y, com_z;
    float mass_mult, ixx_mult, iyy_mult, izz_mult;
    float k_thrust_mult, linear_drag_mult, yaw_drag_mult, motor_lag_mult;
};

struct DroneCudaState {
    float3 pos;
    float3 vel;
    float4 quat;  // (w, x, y, z)
    float3 omega;
    float rpms[4];

    float3 target_pos;
    float3 target_normal;
    float3 ring_pos[DRONE_CUDA_MAX_RINGS];
    float3 ring_normal[DRONE_CUDA_MAX_RINGS];
    float ring_radius[DRONE_CUDA_MAX_RINGS];
    int buffer_idx;
    int buffer_size;
    float rings_passed;
    float race_gate_bank;
    float ring_collision;
    float collisions;
    float3 prev_pos;
    float prev_potential;
    float episode_return;
    int episode_length;

    float hover_score;
    float hover_ema;
    float ema_dist;
    float ema_vel;
    float ema_omega;
    float ema_omega_x;
    float ema_omega_y;
    float ema_omega_z;

    float action_abs_sum;
    float action_clipped_abs_sum;
    float action_max_abs;
    float action_saturation_count;
    float action_delta_sum;
    float reset_action_jump_sum;
    float reset_action_jump_count;
    float motor_clip_low_count;
    float motor_clip_high_count;
    float rpm_sum[4];
    float instrumentation_steps;
    float target_in_fov_sum;
    float retina_rgb_mean_sum;
    float retina_rgb_std_sum;
    float retina_energy_sum;
    float retina_argmax_sum;
    float retina_argmax_left_count;
    float retina_argmax_center_count;
    float retina_argmax_right_count;
    float bearing_error_sum;
    float distance_to_target_sum;
    float retina_signal_vs_distance_sum;
    float gate_time[DRONE_GATE_DEBUG_MAX];
    float gate_target_in_fov[DRONE_GATE_DEBUG_MAX];
    float gate_bearing_error[DRONE_GATE_DEBUG_MAX];
    float gate_distance_to_target[DRONE_GATE_DEBUG_MAX];
    float gate_pass_count[DRONE_GATE_DEBUG_MAX];
    float gate_collision_count[DRONE_GATE_DEBUG_MAX];

    float r_dist_sum;
    float r_hover_sum;
    float r_shaping_sum;
    float r_omega_sum;
    float r_omega_xy_sum;
    float r_omega_z_sum;
    float r_terminal_sum;

    float action_history[DRONE_MAX_ACTION_LATENCY_STEPS + 1][4];
    int action_history_idx;
    float prev_action[4];
    int has_prev_action;
    int pal_probe_active;
    int adr_probe_param;
    int adr_probe_side;
};

struct DroneCudaIsbState {
    float3 pos;
    float3 vel;
    float4 quat;
    float3 omega;
    float rpms[4];
};

struct DroneCudaDerivative {
    float3 vel;
    float3 v_dot;
    float4 q_dot;
    float3 w_dot;
    float rpm_dot[4];
};

struct DroneCudaLog {
    float episode_return;
    float episode_length;
    float rings_passed;
    float collisions;
    float oob;
    float ring_collision;
    float timeout;
    float score;
    float perf;
    float ema_dist;
    float ema_vel;
    float ema_omega;
    float ema_omega_x;
    float ema_omega_y;
    float ema_omega_z;
    float mean_abs_action;
    float mean_abs_action_clipped;
    float max_abs_action;
    float action_saturation_frac;
    float mean_abs_delta_action;
    float reset_action_jump_mean;
    float motor_clip_low_frac;
    float motor_clip_high_frac;
    float hover_trim_rpm_mean;
    float hover_trim_rpm_max;
    float hover_trim_rpm_frac_of_max;
    float mean_rpm_FL;
    float mean_rpm_FR;
    float mean_rpm_RL;
    float mean_rpm_RR;
    float target_in_fov_frac;
    float retina_rgb_mean;
    float retina_rgb_std;
    float retina_energy;
    float retina_left_center_right_argmax;
    float retina_argmax_left_frac;
    float retina_argmax_center_frac;
    float retina_argmax_right_frac;
    float bearing_error_to_target;
    float distance_to_target;
    float retina_signal_vs_distance;
    float r_dist;
    float r_hover;
    float r_shaping;
    float r_omega;
    float r_omega_xy;
    float r_omega_z;
    float r_terminal;
    float mass_mult_mean;
    float ixx_mult_mean;
    float iyy_mult_mean;
    float izz_mult_mean;
    float k_thrust_mult_mean;
    float k_thrust_mult_min;
    float k_thrust_mult_max;
    float linear_drag_mult_mean;
    float yaw_drag_mult_mean;
    float motor_lag_mult_mean;
    float com_x_mean;
    float com_y_mean;
    float com_z_mean;
    float lap_complete;
    float oob_diag_count;
    float gate_index_at_oob;
    float position_norm_at_oob;
    float target_gate_position_norm;
    float next_gate_position_norm;
    float distance_from_track_centerline;
    float gate_time[DRONE_GATE_DEBUG_MAX];
    float gate_target_in_fov[DRONE_GATE_DEBUG_MAX];
    float gate_bearing_error[DRONE_GATE_DEBUG_MAX];
    float gate_distance_to_target[DRONE_GATE_DEBUG_MAX];
    float gate_pass_count[DRONE_GATE_DEBUG_MAX];
    float gate_collision_count[DRONE_GATE_DEBUG_MAX];
    float gate_timeout_count[DRONE_GATE_DEBUG_MAX];
    float gate_oob_count[DRONE_GATE_DEBUG_MAX];
    float n;
};

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

struct DroneCudaCtx {
    int total_agents;
    int horizon;
    int task;
    int max_rings;
    int action_latency_steps;

    float alpha_dist, alpha_hover, alpha_shaping;
    float alpha_omega_xy, alpha_omega_z, alpha_omega_z_sq, alpha_omega_z_mult;
    float alpha_action_delta, alpha_reset_action_delta;
    int reset_action_interval;
    int tick;
    float hover_target_dist, oob_radius, hover_dist, hover_omega, hover_vel;
    float domain_randomization;
    float dr_mass, dr_inertia, dr_k_thrust, dr_linear_drag, dr_yaw_drag, dr_motor_lag;
    float dr_com_xy, dr_com_z;
    float dr_authority_gated;
    float dr_usable_t2w_min, dr_usable_t2w_max;
    float dr_mass_min, dr_mass_max;
    float dr_inertia_min, dr_inertia_max;
    float dr_motor_thrust_min, dr_motor_thrust_max;
    float dr_motor_tau_min, dr_motor_tau_max;
    float dr_yaw_torque_min, dr_yaw_torque_max;
    float dr_linear_drag_min, dr_linear_drag_max;
    float dr_angular_damping_min, dr_angular_damping_max;
    float dr_profile_mix;
    float adr_enabled;
    float adr_mode;
    float adr_probe_prob;
    float adr_success_threshold;
    float adr_contract_threshold;
    float adr_step;
    float adr_eval_episodes;
    float adr_init_usable_t2w_min, adr_init_usable_t2w_max;
    float adr_init_mass_min, adr_init_mass_max;
    float adr_init_inertia_min, adr_init_inertia_max;
    float adr_init_motor_thrust_min, adr_init_motor_thrust_max;
    float adr_init_motor_tau_min, adr_init_motor_tau_max;
    float adr_init_com_xy;
    float pal_probe_prob;
    int pal_probe_steps;
    float pal_probe_amp;
    float action_scale;
    int action_mode;
    float normalized_thrust_min, normalized_thrust_max;
    float reset_pos_scale, reset_yaw_range, reset_vel_max;
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

    DroneCudaState* states;
    DroneCudaParams* params;
    curandStatePhilox4_32_10_t* rng;
    DroneCudaLog* log;
    DroneCudaAdrState* adr;
    DroneCudaIsbState* isb_states;
    int* isb_counts;
    int* isb_cursors;
};

static inline float dict_float(Dict* dict, const char* key, float fallback) {
    DictItem* item = dict_get_unsafe(dict, key);
    return item == NULL ? fallback : (float)item->value;
}

static inline unsigned int dict_uint(Dict* dict, const char* key, unsigned int fallback) {
    DictItem* item = dict_get_unsafe(dict, key);
    return item == NULL ? fallback : (unsigned int)item->value;
}

static inline int latency_steps_from_seconds(float seconds) {
    int steps = (int)floorf((fmaxf(seconds, 0.0f) / DRONE_ACTION_DT) + 0.5f);
    if (steps < 0) return 0;
    if (steps > DRONE_MAX_ACTION_LATENCY_STEPS) return DRONE_MAX_ACTION_LATENCY_STEPS;
    return steps;
}

static inline void cuda_check(cudaError_t err, const char* expr, const char* file, int line) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA env error at %s:%d: %s failed: %s\n",
                file, line, expr, cudaGetErrorString(err));
    }
}

#define CUDA_ENV_CHECK(expr) cuda_check((expr), #expr, __FILE__, __LINE__)

__device__ __forceinline__ float clampf_dev(float v, float lo, float hi) {
    return fminf(fmaxf(v, lo), hi);
}

__device__ __forceinline__ float3 add3_dev(float3 a, float3 b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__device__ __forceinline__ float3 sub3_dev(float3 a, float3 b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__device__ __forceinline__ float3 scale3_dev(float3 a, float b) {
    return make_float3(a.x * b, a.y * b, a.z * b);
}

__device__ __forceinline__ float dot3_dev(float3 a, float3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ __forceinline__ float norm3_dev(float3 a) {
    return sqrtf(dot3_dev(a, a));
}

__device__ __forceinline__ int race_clamped_gate_idx_dev(const DroneCudaState* s) {
    if (s->buffer_size <= 0) return 0;
    int idx = s->buffer_idx;
    if (idx < 0) return 0;
    if (idx >= s->buffer_size) return s->buffer_size - 1;
    return idx;
}

__device__ __forceinline__ float point_segment_distance3_dev(float3 p, float3 a, float3 b) {
    float3 ab = sub3_dev(b, a);
    float denom = dot3_dev(ab, ab);
    if (denom <= 1e-6f) return norm3_dev(sub3_dev(p, b));
    float t = dot3_dev(sub3_dev(p, a), ab) / denom;
    t = clampf_dev(t, 0.0f, 1.0f);
    float3 closest = add3_dev(a, scale3_dev(ab, t));
    return norm3_dev(sub3_dev(p, closest));
}

__device__ __forceinline__ float race_track_centerline_distance_dev(const DroneCudaState* s) {
    if (s->buffer_size <= 0) return 0.0f;
    int idx = race_clamped_gate_idx_dev(s);
    int prev_idx = idx > 0 ? idx - 1 : idx;
    return point_segment_distance3_dev(s->pos, s->ring_pos[prev_idx], s->ring_pos[idx]);
}

__device__ __forceinline__ float3 race_next_gate_pos_dev(const DroneCudaState* s) {
    if (s->buffer_size <= 0) return s->target_pos;
    int idx = race_clamped_gate_idx_dev(s);
    int next_idx = idx + 1 < s->buffer_size ? idx + 1 : idx;
    return s->ring_pos[next_idx];
}

__device__ __forceinline__ float3 normalize3_dev(float3 a, float3 fallback) {
    float n = norm3_dev(a);
    if (n <= 1e-6f) return fallback;
    return scale3_dev(a, 1.0f / n);
}

__device__ __forceinline__ float rndf_dev(float a, float b, curandStatePhilox4_32_10_t* rng) {
    return a + curand_uniform(rng) * (b - a);
}

__device__ __forceinline__ float4 quat_dev(float w, float x, float y, float z) {
    return make_float4(w, x, y, z);
}

__device__ __forceinline__ float4 quat_add_dev(float4 a, float4 b) {
    return quat_dev(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}

__device__ __forceinline__ float4 quat_scale_dev(float4 a, float b) {
    return quat_dev(a.x * b, a.y * b, a.z * b, a.w * b);
}

__device__ __forceinline__ float4 quat_mul_dev(float4 a, float4 b) {
    return quat_dev(
        a.x * b.x - a.y * b.y - a.z * b.z - a.w * b.w,
        a.x * b.y + a.y * b.x + a.z * b.w - a.w * b.z,
        a.x * b.z - a.y * b.w + a.z * b.x + a.w * b.y,
        a.x * b.w + a.y * b.z - a.z * b.y + a.w * b.x);
}

__device__ __forceinline__ void quat_normalize_dev(float4* q) {
    float n = sqrtf(q->x * q->x + q->y * q->y + q->z * q->z + q->w * q->w);
    if (n > 0.0f) {
        q->x /= n;
        q->y /= n;
        q->z /= n;
        q->w /= n;
    }
}

__device__ __forceinline__ float4 quat_inverse_dev(float4 q) {
    return quat_dev(q.x, -q.y, -q.z, -q.w);
}

__device__ __forceinline__ float3 quat_rotate_dev(float4 q, float3 v) {
    float4 qv = quat_dev(0.0f, v.x, v.y, v.z);
    float4 res = quat_mul_dev(quat_mul_dev(q, qv), quat_inverse_dev(q));
    return make_float3(res.y, res.z, res.w);
}

__device__ __forceinline__ float4 quat_from_yaw_dev(float yaw) {
    float half = 0.5f * yaw;
    return quat_dev(cosf(half), 0.0f, 0.0f, sinf(half));
}

__device__ __forceinline__ float4 quat_from_axis_angle_dev(float3 axis, float angle) {
    float half = 0.5f * angle;
    float s = sinf(half);
    return quat_dev(cosf(half), axis.x * s, axis.y * s, axis.z * s);
}

__device__ __forceinline__ float dr_abs_range_dev(float v) {
    return clampf_dev(fabsf(v), 0.0f, 0.95f);
}

__device__ __forceinline__ bool dr_has_granular_dev(const DroneCudaCtx& cfg) {
    return fabsf(cfg.dr_mass) > 0.0f
        || fabsf(cfg.dr_inertia) > 0.0f
        || fabsf(cfg.dr_k_thrust) > 0.0f
        || fabsf(cfg.dr_linear_drag) > 0.0f
        || fabsf(cfg.dr_yaw_drag) > 0.0f
        || fabsf(cfg.dr_motor_lag) > 0.0f
        || fabsf(cfg.dr_com_xy) > 0.0f
        || fabsf(cfg.dr_com_z) > 0.0f;
}

__device__ __forceinline__ float dr_param_range_dev(const DroneCudaCtx& cfg, float granular) {
    if (cfg.domain_randomization <= 0.0f) return 0.0f;
    return dr_has_granular_dev(cfg) ? dr_abs_range_dev(granular)
                                    : dr_abs_range_dev(cfg.domain_randomization);
}

__device__ __forceinline__ float dr_sample_mult_dev(curandStatePhilox4_32_10_t* rng,
                                                    float range) {
    range = dr_abs_range_dev(range);
    return rndf_dev(1.0f - range, 1.0f + range, rng);
}

__device__ __forceinline__ bool dr_authority_gated_dev(const DroneCudaCtx& cfg) {
    return cfg.domain_randomization > 0.0f && cfg.dr_authority_gated > 0.0f;
}

__device__ __forceinline__ bool dr_structured_legacy_mix_dev(const DroneCudaCtx& cfg) {
    return dr_authority_gated_dev(cfg) && cfg.dr_profile_mix >= 3.0f
        && cfg.dr_profile_mix < 4.0f;
}

__device__ __forceinline__ float dr_sample_range_dev(curandStatePhilox4_32_10_t* rng,
                                                     float lo, float hi, float fallback) {
    if (hi < lo) return fallback;
    if (hi == lo) return lo;
    return rndf_dev(lo, hi, rng);
}

__device__ __forceinline__ int dr_risk_score_dev(float usable_t2w, float mass_mult,
                                                 float inertia_max,
                                                 float motor_thrust_min,
                                                 float motor_tau_max,
                                                 float com_offset_norm) {
    int risk = 0;
    if (usable_t2w < 2.15f) risk++;
    if (mass_mult > 1.18f) risk++;
    if (inertia_max > 1.50f) risk++;
    if (motor_thrust_min < 0.88f) risk++;
    if (motor_tau_max > 0.22f) risk++;
    if (com_offset_norm > 0.025f) risk++;
    return risk;
}

__device__ __forceinline__ bool adr_authority_active_dev(const DroneCudaCtx& cfg) {
    return cfg.adr_enabled > 0.0f && cfg.adr != NULL
        && cfg.adr->enabled != 0 && cfg.adr->mode == DRONE_ADR_MODE_AUTHORITY
        && dr_authority_gated_dev(cfg);
}

__device__ __forceinline__ bool adr_legacy_paper_active_dev(const DroneCudaCtx& cfg) {
    return cfg.adr_enabled > 0.0f && cfg.adr != NULL
        && cfg.adr->enabled != 0 && cfg.adr->mode == DRONE_ADR_MODE_LEGACY_PAPER
        && cfg.domain_randomization > 0.0f && !dr_authority_gated_dev(cfg);
}

__device__ __forceinline__ int adr_edge_idx_dev(int param, int side) {
    return param * 2 + side;
}

__device__ __forceinline__ float adr_param_step_dev(const DroneCudaCtx& cfg, int param) {
    float step = fmaxf(cfg.adr_step, 1e-5f);
    if (param == DRONE_ADR_MOTOR_TAU) return step * 0.25f;
    if (param == DRONE_ADR_COM_XY) return step * 0.25f;
    return step;
}

__device__ void adr_adjust_edge_dev(DroneCudaAdrState* adr, const DroneCudaCtx& cfg,
                                    int param, int side, bool success) {
    float step = adr_param_step_dev(cfg, param);
    if (param == DRONE_ADR_USABLE_T2W) {
        float gap = 0.05f;
        if (side == DRONE_ADR_SIDE_LOW) {
            adr->usable_t2w_min = success
                ? fmaxf(cfg.dr_usable_t2w_min, adr->usable_t2w_min - step)
                : fminf(adr->usable_t2w_max - gap, adr->usable_t2w_min + step);
        } else {
            adr->usable_t2w_max = success
                ? fminf(cfg.dr_usable_t2w_max, adr->usable_t2w_max + step)
                : fmaxf(adr->usable_t2w_min + gap, adr->usable_t2w_max - step);
        }
    } else if (param == DRONE_ADR_MASS) {
        float gap = 0.01f;
        if (side == DRONE_ADR_SIDE_LOW) {
            adr->mass_min = success
                ? fmaxf(cfg.dr_mass_min, adr->mass_min - step)
                : fminf(adr->mass_max - gap, adr->mass_min + step);
        } else {
            adr->mass_max = success
                ? fminf(cfg.dr_mass_max, adr->mass_max + step)
                : fmaxf(adr->mass_min + gap, adr->mass_max - step);
        }
    } else if (param == DRONE_ADR_INERTIA) {
        float gap = 0.01f;
        if (side == DRONE_ADR_SIDE_LOW) {
            adr->inertia_min = success
                ? fmaxf(cfg.dr_inertia_min, adr->inertia_min - step)
                : fminf(adr->inertia_max - gap, adr->inertia_min + step);
        } else {
            adr->inertia_max = success
                ? fminf(cfg.dr_inertia_max, adr->inertia_max + step)
                : fmaxf(adr->inertia_min + gap, adr->inertia_max - step);
        }
    } else if (param == DRONE_ADR_MOTOR_THRUST) {
        float gap = 0.01f;
        if (side == DRONE_ADR_SIDE_LOW) {
            adr->motor_thrust_min = success
                ? fmaxf(cfg.dr_motor_thrust_min, adr->motor_thrust_min - step)
                : fminf(adr->motor_thrust_max - gap, adr->motor_thrust_min + step);
        } else {
            adr->motor_thrust_max = success
                ? fminf(cfg.dr_motor_thrust_max, adr->motor_thrust_max + step)
                : fmaxf(adr->motor_thrust_min + gap, adr->motor_thrust_max - step);
        }
    } else if (param == DRONE_ADR_MOTOR_TAU) {
        float gap = 0.002f;
        if (side == DRONE_ADR_SIDE_LOW) {
            adr->motor_tau_min = success
                ? fmaxf(cfg.dr_motor_tau_min, adr->motor_tau_min - step)
                : fminf(adr->motor_tau_max - gap, adr->motor_tau_min + step);
        } else {
            adr->motor_tau_max = success
                ? fminf(cfg.dr_motor_tau_max, adr->motor_tau_max + step)
                : fmaxf(adr->motor_tau_min + gap, adr->motor_tau_max - step);
        }
    } else if (param == DRONE_ADR_COM_XY && side == DRONE_ADR_SIDE_HIGH) {
        adr->com_xy = success
            ? fminf(fabsf(cfg.dr_com_xy), adr->com_xy + step)
            : fmaxf(0.0f, adr->com_xy - step);
    }
}

__device__ void adr_apply_bounds_and_probe_dev(
    DroneCudaState* s, const DroneCudaCtx& cfg, curandStatePhilox4_32_10_t* rng,
    float* usable_t2w_min, float* usable_t2w_max,
    float* mass_min, float* mass_max,
    float* inertia_min, float* inertia_max,
    float* motor_thrust_min, float* motor_thrust_max,
    float* motor_tau_min, float* motor_tau_max,
    float* com_xy_range) {
    s->adr_probe_param = -1;
    s->adr_probe_side = -1;
    if (!adr_authority_active_dev(cfg)) return;

    DroneCudaAdrState* adr = cfg.adr;
    *usable_t2w_min = adr->usable_t2w_min;
    *usable_t2w_max = adr->usable_t2w_max;
    *mass_min = adr->mass_min;
    *mass_max = adr->mass_max;
    *inertia_min = adr->inertia_min;
    *inertia_max = adr->inertia_max;
    *motor_thrust_min = adr->motor_thrust_min;
    *motor_thrust_max = adr->motor_thrust_max;
    *motor_tau_min = adr->motor_tau_min;
    *motor_tau_max = adr->motor_tau_max;
    *com_xy_range = fminf(fabsf(cfg.dr_com_xy), adr->com_xy);

    float probe_prob = clampf_dev(cfg.adr_probe_prob, 0.0f, 1.0f);
    if (curand_uniform(rng) > probe_prob) return;

    int param = (int)floorf(curand_uniform(rng) * (float)DRONE_ADR_PARAM_COUNT);
    if (param >= DRONE_ADR_PARAM_COUNT) param = DRONE_ADR_PARAM_COUNT - 1;
    int side = curand_uniform(rng) < 0.5f ? DRONE_ADR_SIDE_LOW : DRONE_ADR_SIDE_HIGH;
    if (param == DRONE_ADR_COM_XY) side = DRONE_ADR_SIDE_HIGH;

    s->adr_probe_param = param;
    s->adr_probe_side = side;

    if (param == DRONE_ADR_USABLE_T2W) {
        float v = side == DRONE_ADR_SIDE_LOW ? adr->usable_t2w_min : adr->usable_t2w_max;
        *usable_t2w_min = v;
        *usable_t2w_max = v;
    } else if (param == DRONE_ADR_MASS) {
        float v = side == DRONE_ADR_SIDE_LOW ? adr->mass_min : adr->mass_max;
        *mass_min = v;
        *mass_max = v;
    } else if (param == DRONE_ADR_INERTIA) {
        float v = side == DRONE_ADR_SIDE_LOW ? adr->inertia_min : adr->inertia_max;
        *inertia_min = v;
        *inertia_max = v;
    } else if (param == DRONE_ADR_MOTOR_THRUST) {
        float v = side == DRONE_ADR_SIDE_LOW ? adr->motor_thrust_min : adr->motor_thrust_max;
        *motor_thrust_min = v;
        *motor_thrust_max = v;
    } else if (param == DRONE_ADR_MOTOR_TAU) {
        float v = side == DRONE_ADR_SIDE_LOW ? adr->motor_tau_min : adr->motor_tau_max;
        *motor_tau_min = v;
        *motor_tau_max = v;
    } else if (param == DRONE_ADR_COM_XY) {
        *com_xy_range = adr->com_xy;
    }
}

__device__ __forceinline__ float adr_legacy_param_step_dev(const DroneCudaCtx& cfg, int param) {
    float step = fmaxf(cfg.adr_step, 1e-5f);
    if (param == DRONE_ADR_LEGACY_COM_XY || param == DRONE_ADR_LEGACY_COM_Z) {
        return step * 0.25f;
    }
    return step;
}

__device__ void adr_legacy_adjust_edge_dev(DroneCudaAdrState* adr, const DroneCudaCtx& cfg,
                                           int param, int side, bool expand) {
    float step = adr_legacy_param_step_dev(cfg, param);

    if (param == DRONE_ADR_LEGACY_MASS) {
        float hard_lo = fmaxf(0.01f, 1.0f - fabsf(cfg.dr_mass));
        float hard_hi = 1.0f + fabsf(cfg.dr_mass);
        if (side == DRONE_ADR_SIDE_LOW) {
            adr->legacy_mass_min = expand
                ? fmaxf(hard_lo, adr->legacy_mass_min - step)
                : fminf(adr->legacy_mass_max, adr->legacy_mass_min + step);
        } else {
            adr->legacy_mass_max = expand
                ? fminf(hard_hi, adr->legacy_mass_max + step)
                : fmaxf(adr->legacy_mass_min, adr->legacy_mass_max - step);
        }
    } else if (param == DRONE_ADR_LEGACY_INERTIA) {
        float hard_lo = fmaxf(0.01f, 1.0f - fabsf(cfg.dr_inertia));
        float hard_hi = 1.0f + fabsf(cfg.dr_inertia);
        if (side == DRONE_ADR_SIDE_LOW) {
            adr->legacy_inertia_min = expand
                ? fmaxf(hard_lo, adr->legacy_inertia_min - step)
                : fminf(adr->legacy_inertia_max, adr->legacy_inertia_min + step);
        } else {
            adr->legacy_inertia_max = expand
                ? fminf(hard_hi, adr->legacy_inertia_max + step)
                : fmaxf(adr->legacy_inertia_min, adr->legacy_inertia_max - step);
        }
    } else if (param == DRONE_ADR_LEGACY_K_THRUST) {
        float hard_lo = fmaxf(0.01f, 1.0f - fabsf(cfg.dr_k_thrust));
        float hard_hi = 1.0f + fabsf(cfg.dr_k_thrust);
        if (side == DRONE_ADR_SIDE_LOW) {
            adr->legacy_k_thrust_min = expand
                ? fmaxf(hard_lo, adr->legacy_k_thrust_min - step)
                : fminf(adr->legacy_k_thrust_max, adr->legacy_k_thrust_min + step);
        } else {
            adr->legacy_k_thrust_max = expand
                ? fminf(hard_hi, adr->legacy_k_thrust_max + step)
                : fmaxf(adr->legacy_k_thrust_min, adr->legacy_k_thrust_max - step);
        }
    } else if (param == DRONE_ADR_LEGACY_LINEAR_DRAG) {
        float hard_lo = fmaxf(0.01f, 1.0f - fabsf(cfg.dr_linear_drag));
        float hard_hi = 1.0f + fabsf(cfg.dr_linear_drag);
        if (side == DRONE_ADR_SIDE_LOW) {
            adr->legacy_linear_drag_min = expand
                ? fmaxf(hard_lo, adr->legacy_linear_drag_min - step)
                : fminf(adr->legacy_linear_drag_max, adr->legacy_linear_drag_min + step);
        } else {
            adr->legacy_linear_drag_max = expand
                ? fminf(hard_hi, adr->legacy_linear_drag_max + step)
                : fmaxf(adr->legacy_linear_drag_min, adr->legacy_linear_drag_max - step);
        }
    } else if (param == DRONE_ADR_LEGACY_YAW_DRAG) {
        float hard_lo = fmaxf(0.01f, 1.0f - fabsf(cfg.dr_yaw_drag));
        float hard_hi = 1.0f + fabsf(cfg.dr_yaw_drag);
        if (side == DRONE_ADR_SIDE_LOW) {
            adr->legacy_yaw_drag_min = expand
                ? fmaxf(hard_lo, adr->legacy_yaw_drag_min - step)
                : fminf(adr->legacy_yaw_drag_max, adr->legacy_yaw_drag_min + step);
        } else {
            adr->legacy_yaw_drag_max = expand
                ? fminf(hard_hi, adr->legacy_yaw_drag_max + step)
                : fmaxf(adr->legacy_yaw_drag_min, adr->legacy_yaw_drag_max - step);
        }
    } else if (param == DRONE_ADR_LEGACY_MOTOR_LAG) {
        float hard_lo = fmaxf(0.01f, 1.0f - fabsf(cfg.dr_motor_lag));
        float hard_hi = 1.0f + fabsf(cfg.dr_motor_lag);
        if (side == DRONE_ADR_SIDE_LOW) {
            adr->legacy_motor_lag_min = expand
                ? fmaxf(hard_lo, adr->legacy_motor_lag_min - step)
                : fminf(adr->legacy_motor_lag_max, adr->legacy_motor_lag_min + step);
        } else {
            adr->legacy_motor_lag_max = expand
                ? fminf(hard_hi, adr->legacy_motor_lag_max + step)
                : fmaxf(adr->legacy_motor_lag_min, adr->legacy_motor_lag_max - step);
        }
    } else if (param == DRONE_ADR_LEGACY_COM_XY && side == DRONE_ADR_SIDE_HIGH) {
        adr->legacy_com_xy = expand
            ? fminf(fabsf(cfg.dr_com_xy), adr->legacy_com_xy + step)
            : fmaxf(0.0f, adr->legacy_com_xy - step);
    } else if (param == DRONE_ADR_LEGACY_COM_Z && side == DRONE_ADR_SIDE_HIGH) {
        adr->legacy_com_z = expand
            ? fminf(fabsf(cfg.dr_com_z), adr->legacy_com_z + step)
            : fmaxf(0.0f, adr->legacy_com_z - step);
    }
}

__device__ void adr_legacy_apply_bounds_and_probe_dev(
    DroneCudaState* s, const DroneCudaCtx& cfg, curandStatePhilox4_32_10_t* rng,
    float* mass_min, float* mass_max,
    float* inertia_min, float* inertia_max,
    float* k_thrust_min, float* k_thrust_max,
    float* linear_drag_min, float* linear_drag_max,
    float* yaw_drag_min, float* yaw_drag_max,
    float* motor_lag_min, float* motor_lag_max,
    float* com_xy_range, float* com_z_range) {
    if (!adr_legacy_paper_active_dev(cfg)) return;

    DroneCudaAdrState* adr = cfg.adr;
    *mass_min = adr->legacy_mass_min;
    *mass_max = adr->legacy_mass_max;
    *inertia_min = adr->legacy_inertia_min;
    *inertia_max = adr->legacy_inertia_max;
    *k_thrust_min = adr->legacy_k_thrust_min;
    *k_thrust_max = adr->legacy_k_thrust_max;
    *linear_drag_min = adr->legacy_linear_drag_min;
    *linear_drag_max = adr->legacy_linear_drag_max;
    *yaw_drag_min = adr->legacy_yaw_drag_min;
    *yaw_drag_max = adr->legacy_yaw_drag_max;
    *motor_lag_min = adr->legacy_motor_lag_min;
    *motor_lag_max = adr->legacy_motor_lag_max;
    *com_xy_range = fminf(fabsf(cfg.dr_com_xy), adr->legacy_com_xy);
    *com_z_range = fminf(fabsf(cfg.dr_com_z), adr->legacy_com_z);

    float probe_prob = clampf_dev(cfg.adr_probe_prob, 0.0f, 1.0f);
    if (curand_uniform(rng) > probe_prob) return;

    int param = (int)floorf(curand_uniform(rng) * (float)DRONE_ADR_LEGACY_PARAM_COUNT);
    if (param >= DRONE_ADR_LEGACY_PARAM_COUNT) param = DRONE_ADR_LEGACY_PARAM_COUNT - 1;
    int side = curand_uniform(rng) < 0.5f ? DRONE_ADR_SIDE_LOW : DRONE_ADR_SIDE_HIGH;
    if (param == DRONE_ADR_LEGACY_COM_XY || param == DRONE_ADR_LEGACY_COM_Z) {
        side = DRONE_ADR_SIDE_HIGH;
    }

    s->adr_probe_param = param;
    s->adr_probe_side = side;

    if (param == DRONE_ADR_LEGACY_MASS) {
        float v = side == DRONE_ADR_SIDE_LOW ? adr->legacy_mass_min : adr->legacy_mass_max;
        *mass_min = v; *mass_max = v;
    } else if (param == DRONE_ADR_LEGACY_INERTIA) {
        float v = side == DRONE_ADR_SIDE_LOW ? adr->legacy_inertia_min : adr->legacy_inertia_max;
        *inertia_min = v; *inertia_max = v;
    } else if (param == DRONE_ADR_LEGACY_K_THRUST) {
        float v = side == DRONE_ADR_SIDE_LOW ? adr->legacy_k_thrust_min : adr->legacy_k_thrust_max;
        *k_thrust_min = v; *k_thrust_max = v;
    } else if (param == DRONE_ADR_LEGACY_LINEAR_DRAG) {
        float v = side == DRONE_ADR_SIDE_LOW ? adr->legacy_linear_drag_min : adr->legacy_linear_drag_max;
        *linear_drag_min = v; *linear_drag_max = v;
    } else if (param == DRONE_ADR_LEGACY_YAW_DRAG) {
        float v = side == DRONE_ADR_SIDE_LOW ? adr->legacy_yaw_drag_min : adr->legacy_yaw_drag_max;
        *yaw_drag_min = v; *yaw_drag_max = v;
    } else if (param == DRONE_ADR_LEGACY_MOTOR_LAG) {
        float v = side == DRONE_ADR_SIDE_LOW ? adr->legacy_motor_lag_min : adr->legacy_motor_lag_max;
        *motor_lag_min = v; *motor_lag_max = v;
    } else if (param == DRONE_ADR_LEGACY_COM_XY) {
        *com_xy_range = adr->legacy_com_xy;
    } else if (param == DRONE_ADR_LEGACY_COM_Z) {
        *com_z_range = adr->legacy_com_z;
    }
}

__device__ void adr_record_done_dev(const DroneCudaCtx& cfg, const DroneCudaState* s,
                                    bool timeout) {
    DroneCudaAdrState* adr = cfg.adr;
    if (adr == NULL || s->adr_probe_param < 0 || s->adr_probe_side < 0) return;

    if (adr_authority_active_dev(cfg)) {
        int idx = adr_edge_idx_dev(s->adr_probe_param, s->adr_probe_side);
        if (idx < 0 || idx >= DRONE_ADR_EDGE_COUNT) return;

        if (timeout) atomicAdd(&adr->successes[idx], 1u);
        unsigned int count = atomicAdd(&adr->counts[idx], 1u) + 1u;
        unsigned int target = (unsigned int)fmaxf(cfg.adr_eval_episodes, 1.0f);
        if (count < target) return;
        if (atomicCAS(&adr->updating[idx], 0u, 1u) != 0u) return;

        unsigned int total = adr->counts[idx];
        unsigned int good = adr->successes[idx];
        float rate = total > 0u ? ((float)good / (float)total) : 0.0f;
        bool success = rate >= cfg.adr_success_threshold;
        adr_adjust_edge_dev(adr, cfg, s->adr_probe_param, s->adr_probe_side, success);
        adr->counts[idx] = 0u;
        adr->successes[idx] = 0u;
        atomicAdd(&adr->updates[idx], 1u);
        adr->updating[idx] = 0u;
    } else if (adr_legacy_paper_active_dev(cfg)) {
        int idx = adr_edge_idx_dev(s->adr_probe_param, s->adr_probe_side);
        if (idx < 0 || idx >= DRONE_ADR_LEGACY_EDGE_COUNT) return;

        float perf = timeout ? 1.0f : 0.0f;
        atomicAdd(&adr->legacy_perf_sum[idx], perf);
        unsigned int count = atomicAdd(&adr->legacy_counts[idx], 1u) + 1u;
        unsigned int target = (unsigned int)fmaxf(cfg.adr_eval_episodes, 1.0f);
        if (count < target) return;
        if (atomicCAS(&adr->legacy_updating[idx], 0u, 1u) != 0u) return;

        unsigned int total = adr->legacy_counts[idx];
        float avg = total > 0u ? (adr->legacy_perf_sum[idx] / (float)total) : 0.0f;
        if (avg >= cfg.adr_success_threshold) {
            adr_legacy_adjust_edge_dev(adr, cfg, s->adr_probe_param, s->adr_probe_side, true);
        } else if (avg <= cfg.adr_contract_threshold) {
            adr_legacy_adjust_edge_dev(adr, cfg, s->adr_probe_param, s->adr_probe_side, false);
        }
        adr->legacy_counts[idx] = 0u;
        adr->legacy_perf_sum[idx] = 0.0f;
        atomicAdd(&adr->legacy_updates[idx], 1u);
        adr->legacy_updating[idx] = 0u;
    }
}

__device__ __forceinline__ float motor_thrust_coeff_dev(const DroneCudaParams* p, int i) {
    return p->k_thrust * p->motor_thrust_scale[i];
}

__device__ __forceinline__ float max_motor_thrust_i_dev(const DroneCudaParams* p, int i) {
    return motor_thrust_coeff_dev(p, i) * p->max_rpm * p->max_rpm;
}

__device__ float total_max_motor_thrust_dev(const DroneCudaParams* p) {
    float total = 0.0f;
    #pragma unroll
    for (int i = 0; i < 4; i++) total += max_motor_thrust_i_dev(p, i);
    return total;
}

__device__ float max_motor_thrust_dev(const DroneCudaParams* p) {
    return 0.25f * total_max_motor_thrust_dev(p);
}

__device__ bool solve_allocation_dev(const DroneCudaParams* p, float total_thrust,
                                     float3 torque, float out[4]) {
    float a[4][5] = {
        {1.0f, 1.0f, 1.0f, 1.0f, total_thrust},
        {p->motor_y[0], p->motor_y[1], p->motor_y[2], p->motor_y[3], torque.x},
        {-p->motor_x[0], -p->motor_x[1], -p->motor_x[2], -p->motor_x[3], torque.y},
        {p->k_drag * p->yaw_sign[0], p->k_drag * p->yaw_sign[1],
         p->k_drag * p->yaw_sign[2], p->k_drag * p->yaw_sign[3], torque.z},
    };

    for (int col = 0; col < 4; col++) {
        int pivot = col;
        float best = fabsf(a[col][col]);
        for (int row = col + 1; row < 4; row++) {
            float candidate = fabsf(a[row][col]);
            if (candidate > best) {
                best = candidate;
                pivot = row;
            }
        }
        if (best < 1e-8f) return false;
        if (pivot != col) {
            for (int k = col; k < 5; k++) {
                float tmp = a[col][k];
                a[col][k] = a[pivot][k];
                a[pivot][k] = tmp;
            }
        }
        float inv = 1.0f / a[col][col];
        for (int k = col; k < 5; k++) a[col][k] *= inv;
        for (int row = 0; row < 4; row++) {
            if (row == col) continue;
            float f = a[row][col];
            for (int k = col; k < 5; k++) a[row][k] -= f * a[col][k];
        }
    }

    #pragma unroll
    for (int i = 0; i < 4; i++) {
        out[i] = clampf_dev(a[i][4], 0.0f, max_motor_thrust_i_dev(p, i));
    }
    return true;
}

__device__ void hover_trim_thrusts_dev(const DroneCudaParams* p, float out[4]) {
    if (!solve_allocation_dev(p, p->mass * p->gravity, make_float3(0.0f, 0.0f, 0.0f), out)) {
        float fallback = 0.25f * p->mass * p->gravity;
        #pragma unroll
        for (int i = 0; i < 4; i++) out[i] = fallback;
    }
}

__device__ __forceinline__ float thrust_to_rpm_i_dev(const DroneCudaParams* p, int i,
                                                     float thrust) {
    float k = fmaxf(motor_thrust_coeff_dev(p, i), 1e-12f);
    thrust = clampf_dev(thrust, 0.0f, max_motor_thrust_i_dev(p, i));
    return sqrtf(thrust / k);
}

__device__ __forceinline__ float thrust_to_rpm_dev(const DroneCudaParams* p, float thrust) {
    return thrust_to_rpm_i_dev(p, 0, thrust);
}

__device__ __forceinline__ float normalized_thrust_command_dev(const DroneCudaParams* p,
                                                               float raw_action) {
    float f_hat = 0.5f * (clampf_dev(raw_action, -1.0f, 1.0f) + 1.0f);
    float lo = clampf_dev(p->normalized_thrust_min, 0.0f, 1.0f);
    float hi = clampf_dev(p->normalized_thrust_max, lo, 1.0f);
    return clampf_dev(f_hat, lo, hi);
}

__device__ void init_params_dev(DroneCudaParams* p, DroneCudaState* s, const DroneCudaCtx& cfg,
                                curandStatePhilox4_32_10_t* rng) {
    s->adr_probe_param = -1;
    s->adr_probe_side = -1;
    bool authority_gated = dr_authority_gated_dev(cfg);
    float usable_t2w_min = cfg.dr_usable_t2w_min;
    float usable_t2w_max = cfg.dr_usable_t2w_max;
    float mass_min = cfg.dr_mass_min;
    float mass_max = cfg.dr_mass_max;
    float inertia_min = cfg.dr_inertia_min;
    float inertia_max = cfg.dr_inertia_max;
    float motor_thrust_min = cfg.dr_motor_thrust_min;
    float motor_thrust_max = cfg.dr_motor_thrust_max;
    float motor_tau_min = cfg.dr_motor_tau_min;
    float motor_tau_max = cfg.dr_motor_tau_max;
    float yaw_torque_min = cfg.dr_yaw_torque_min;
    float yaw_torque_max = cfg.dr_yaw_torque_max;
    float linear_drag_min = cfg.dr_linear_drag_min;
    float linear_drag_max = cfg.dr_linear_drag_max;
    float angular_damping_min = cfg.dr_angular_damping_min;
    float angular_damping_max = cfg.dr_angular_damping_max;
    float com_xy_range = cfg.domain_randomization > 0.0f ? fabsf(cfg.dr_com_xy) : 0.0f;
    float com_z_range = cfg.domain_randomization > 0.0f ? fabsf(cfg.dr_com_z) : 0.0f;
    float normalized_thrust_max = cfg.normalized_thrust_max;
    bool direct_k_thrust_profile = false;
    float direct_k_thrust_min = 1.0f;
    float direct_k_thrust_max = 1.0f;
    int risk_limit = 100;
    bool force_three_plus_one = false;
    bool force_fast_slow_mismatch = false;

    if (dr_structured_legacy_mix_dev(cfg)) {
        float profile = rndf_dev(0.0f, 1.0f, rng);
        if (profile < 0.35f) {
            float hard = rndf_dev(0.0f, 1.0f, rng);
            if (hard < 0.65f) {
                usable_t2w_min = 2.35f; usable_t2w_max = 3.80f;
                mass_min = 0.90f; mass_max = 1.15f;
                inertia_min = 0.75f; inertia_max = 1.40f;
                motor_thrust_min = 0.92f; motor_thrust_max = 1.08f;
                motor_tau_min = 0.08f; motor_tau_max = 0.18f;
                yaw_torque_min = 0.80f; yaw_torque_max = 1.20f;
                linear_drag_min = 0.50f; linear_drag_max = 1.50f;
                angular_damping_min = 0.70f; angular_damping_max = 1.50f;
                com_xy_range = 0.018f; com_z_range = 0.010f;
                risk_limit = 2;
            } else {
                usable_t2w_min = 2.10f; usable_t2w_max = 4.00f;
                mass_min = 0.82f; mass_max = 1.22f;
                inertia_min = 0.60f; inertia_max = 1.55f;
                motor_thrust_min = 0.88f; motor_thrust_max = 1.12f;
                motor_tau_min = 0.08f; motor_tau_max = 0.24f;
                yaw_torque_min = 0.75f; yaw_torque_max = 1.30f;
                linear_drag_min = 0.30f; linear_drag_max = 1.90f;
                angular_damping_min = 0.50f; angular_damping_max = 2.00f;
                com_xy_range = 0.025f; com_z_range = 0.015f;
                risk_limit = 3;
            }
        } else if (profile < 0.55f) {
            usable_t2w_min = 2.00f; usable_t2w_max = 2.30f;
            mass_min = 1.12f; mass_max = 1.25f;
            inertia_min = 1.35f; inertia_max = 1.65f;
            motor_thrust_min = 0.84f; motor_thrust_max = 1.05f;
            motor_tau_min = 0.18f; motor_tau_max = 0.28f;
            yaw_torque_min = 0.75f; yaw_torque_max = 1.30f;
            linear_drag_min = 1.20f; linear_drag_max = 2.00f;
            angular_damping_min = 1.20f; angular_damping_max = 2.00f;
            com_xy_range = 0.030f; com_z_range = 0.018f;
            risk_limit = 4;
        } else if (profile < 0.70f) {
            usable_t2w_min = 2.60f; usable_t2w_max = 4.20f;
            mass_min = 0.75f; mass_max = 0.95f;
            inertia_min = 0.45f; inertia_max = 0.85f;
            motor_thrust_min = 0.88f; motor_thrust_max = 1.12f;
            motor_tau_min = 0.05f; motor_tau_max = 0.24f;
            yaw_torque_min = 0.75f; yaw_torque_max = 1.30f;
            linear_drag_min = 0.25f; linear_drag_max = 1.20f;
            angular_damping_min = 0.50f; angular_damping_max = 1.50f;
            com_xy_range = 0.018f; com_z_range = 0.010f;
            risk_limit = 3;
            force_fast_slow_mismatch = true;
        } else if (profile < 0.90f) {
            usable_t2w_min = 2.25f; usable_t2w_max = 4.00f;
            mass_min = 0.85f; mass_max = 1.20f;
            inertia_min = 0.70f; inertia_max = 1.45f;
            motor_thrust_min = 0.92f; motor_thrust_max = 1.08f;
            motor_tau_min = 0.08f; motor_tau_max = 0.18f;
            yaw_torque_min = 0.85f; yaw_torque_max = 1.15f;
            linear_drag_min = 0.50f; linear_drag_max = 1.50f;
            angular_damping_min = 0.70f; angular_damping_max = 1.70f;
            com_xy_range = 0.022f; com_z_range = 0.012f;
            risk_limit = 3;
            force_three_plus_one = true;
        } else {
            usable_t2w_min = 2.20f; usable_t2w_max = 3.80f;
            mass_min = 0.85f; mass_max = 1.15f;
            inertia_min = 0.70f; inertia_max = 1.40f;
            motor_thrust_min = 0.90f; motor_thrust_max = 1.10f;
            motor_tau_min = 0.08f; motor_tau_max = 0.20f;
            yaw_torque_min = 0.80f; yaw_torque_max = 1.25f;
            linear_drag_min = 0.50f; linear_drag_max = 1.50f;
            angular_damping_min = 0.50f; angular_damping_max = 1.50f;
            com_xy_range = 0.015f; com_z_range = 0.010f;
            risk_limit = 2;
        }
    } else if (authority_gated && cfg.dr_profile_mix >= 2.0f && cfg.dr_profile_mix < 3.0f) {
        float profile = rndf_dev(0.0f, 1.0f, rng);
        if (profile < 0.50f) {
            usable_t2w_min = 2.00f; usable_t2w_max = 4.20f;
            mass_min = 0.80f; mass_max = 1.25f;
            inertia_min = 0.60f; inertia_max = 1.60f;
            motor_thrust_min = 0.85f; motor_thrust_max = 1.15f;
            motor_tau_min = 0.06f; motor_tau_max = 0.24f;
            yaw_torque_min = 0.75f; yaw_torque_max = 1.30f;
            linear_drag_min = 0.25f; linear_drag_max = 2.00f;
            angular_damping_min = 0.50f; angular_damping_max = 2.00f;
            com_xy_range = 0.025f; com_z_range = 0.015f;
            risk_limit = 3;
        } else if (profile < 0.75f) {
            normalized_thrust_max = rndf_dev(0.45f, 0.60f, rng);
            mass_min = 0.95f; mass_max = 1.10f;
            inertia_min = 0.90f; inertia_max = 1.30f;
            motor_thrust_min = 0.95f; motor_thrust_max = 1.05f;
            motor_tau_min = 0.08f; motor_tau_max = 0.18f;
            yaw_torque_min = 0.85f; yaw_torque_max = 1.15f;
            linear_drag_min = 0.50f; linear_drag_max = 1.50f;
            angular_damping_min = 0.50f; angular_damping_max = 1.50f;
            com_xy_range = 0.012f; com_z_range = 0.008f;
            direct_k_thrust_profile = true;
            direct_k_thrust_min = 2.50f;
            direct_k_thrust_max = 3.20f;
        } else if (profile < 0.90f) {
            usable_t2w_min = 2.20f; usable_t2w_max = 3.80f;
            mass_min = 0.90f; mass_max = 1.20f;
            inertia_min = 0.80f; inertia_max = 1.50f;
            motor_thrust_min = 0.90f; motor_thrust_max = 1.10f;
            motor_tau_min = 0.20f; motor_tau_max = 0.26f;
            yaw_torque_min = 0.80f; yaw_torque_max = 1.25f;
            linear_drag_min = 0.50f; linear_drag_max = 1.75f;
            angular_damping_min = 0.50f; angular_damping_max = 2.00f;
            com_xy_range = 0.018f; com_z_range = 0.010f;
            risk_limit = 3;
        } else {
            usable_t2w_min = 2.40f; usable_t2w_max = 4.00f;
            mass_min = 0.80f; mass_max = 1.20f;
            inertia_min = 0.60f; inertia_max = 1.50f;
            motor_thrust_min = 0.88f; motor_thrust_max = 1.15f;
            motor_tau_min = 0.06f; motor_tau_max = 0.22f;
            yaw_torque_min = 0.75f; yaw_torque_max = 1.30f;
            linear_drag_min = 0.25f; linear_drag_max = 2.00f;
            angular_damping_min = 0.50f; angular_damping_max = 2.00f;
            com_xy_range = 0.022f; com_z_range = 0.012f;
            risk_limit = 2;
        }
    } else if (authority_gated && cfg.dr_profile_mix > 0.0f) {
        float profile = rndf_dev(0.0f, 1.0f, rng);
        if (profile < 0.40f) {
            usable_t2w_min = 2.2f; usable_t2w_max = 3.8f;
            mass_min = 0.85f; mass_max = 1.15f;
            inertia_min = 0.70f; inertia_max = 1.40f;
            motor_thrust_min = 0.90f; motor_thrust_max = 1.10f;
            motor_tau_min = 0.08f; motor_tau_max = 0.20f;
            yaw_torque_min = 0.80f; yaw_torque_max = 1.25f;
            linear_drag_min = 0.50f; linear_drag_max = 1.50f;
            angular_damping_min = 0.50f; angular_damping_max = 1.50f;
            com_xy_range = 0.015f; com_z_range = 0.010f;
        } else if (profile < 0.80f) {
            usable_t2w_min = 2.10f; usable_t2w_max = 4.20f;
            mass_min = 0.80f; mass_max = 1.22f;
            inertia_min = 0.60f; inertia_max = 1.55f;
            motor_thrust_min = 0.86f; motor_thrust_max = 1.15f;
            motor_tau_min = 0.06f; motor_tau_max = 0.22f;
            yaw_torque_min = 0.75f; yaw_torque_max = 1.30f;
            linear_drag_min = 0.25f; linear_drag_max = 2.00f;
            angular_damping_min = 0.50f; angular_damping_max = 2.00f;
            com_xy_range = 0.022f; com_z_range = 0.012f;
            risk_limit = 2;
        } else {
            float edge = rndf_dev(0.0f, 1.0f, rng);
            usable_t2w_min = 2.10f; usable_t2w_max = 4.20f;
            mass_min = 0.80f; mass_max = 1.22f;
            inertia_min = 0.60f; inertia_max = 1.55f;
            motor_thrust_min = 0.86f; motor_thrust_max = 1.15f;
            motor_tau_min = 0.06f; motor_tau_max = 0.22f;
            yaw_torque_min = 0.75f; yaw_torque_max = 1.25f;
            linear_drag_min = 0.50f; linear_drag_max = 1.50f;
            angular_damping_min = 0.50f; angular_damping_max = 2.00f;
            com_xy_range = 0.022f; com_z_range = 0.012f;
            risk_limit = 3;
            if (edge < 0.25f) {
                usable_t2w_min = 2.00f; usable_t2w_max = 2.15f;
                motor_tau_max = 0.20f;
            } else if (edge < 0.50f) {
                usable_t2w_min = 2.20f;
                motor_tau_min = 0.22f; motor_tau_max = 0.24f;
            } else if (edge < 0.75f) {
                usable_t2w_min = 2.20f;
                mass_min = 1.18f; mass_max = 1.25f;
                inertia_min = 1.45f; inertia_max = 1.60f;
            } else {
                usable_t2w_min = 2.25f;
                com_xy_range = 0.025f; com_z_range = 0.015f;
            }
        }
    }

    adr_apply_bounds_and_probe_dev(
        s, cfg, rng, &usable_t2w_min, &usable_t2w_max, &mass_min, &mass_max,
        &inertia_min, &inertia_max, &motor_thrust_min, &motor_thrust_max,
        &motor_tau_min, &motor_tau_max, &com_xy_range);

    float mass_mult = 1.0f;
    float ixx_mult = 1.0f;
    float iyy_mult = 1.0f;
    float izz_mult = 1.0f;
    float k_thrust_mult = dr_sample_mult_dev(rng, dr_param_range_dev(cfg, cfg.dr_k_thrust));
    float linear_drag_mult = 1.0f;
    float yaw_drag_mult = dr_sample_mult_dev(rng, dr_param_range_dev(cfg, cfg.dr_yaw_drag));
    float motor_lag_mult = dr_sample_mult_dev(rng, dr_param_range_dev(cfg, cfg.dr_motor_lag));
    float angular_damping_mult = 1.0f;
    float motor_thrust_scale[4] = {1.0f, 1.0f, 1.0f, 1.0f};
    float motor_tau[4] = {
        BASE_K_MOT * motor_lag_mult, BASE_K_MOT * motor_lag_mult,
        BASE_K_MOT * motor_lag_mult, BASE_K_MOT * motor_lag_mult
    };
    float yaw_torque_scale[4] = {1.0f, 1.0f, 1.0f, 1.0f};
    float com_x = 0.0f;
    float com_y = 0.0f;
    float com_z = 0.0f;
    if (authority_gated) {
        float usable_t2w = dr_sample_range_dev(
            rng, usable_t2w_min, usable_t2w_max, 2.5f);
        for (int attempt = 0; attempt < 16; attempt++) {
            mass_mult = dr_sample_range_dev(rng, mass_min, mass_max, 1.0f);
            ixx_mult = dr_sample_range_dev(rng, inertia_min, inertia_max, 1.0f);
            iyy_mult = dr_sample_range_dev(rng, inertia_min, inertia_max, 1.0f);
            izz_mult = dr_sample_range_dev(rng, inertia_min, inertia_max, 1.0f);
            linear_drag_mult = dr_sample_range_dev(rng, linear_drag_min, linear_drag_max, 1.0f);
            angular_damping_mult = dr_sample_range_dev(
                rng, angular_damping_min, angular_damping_max, 1.0f);
            usable_t2w = dr_sample_range_dev(rng, usable_t2w_min, usable_t2w_max, 2.5f);
            com_x = rndf_dev(-com_xy_range, com_xy_range, rng);
            com_y = rndf_dev(-com_xy_range, com_xy_range, rng);
            com_z = rndf_dev(-com_z_range, com_z_range, rng);

            float motor_scale_min = 2.0f;
            float motor_tau_sample_max = 0.0f;
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                motor_thrust_scale[i] = dr_sample_range_dev(
                    rng, motor_thrust_min, motor_thrust_max, 1.0f);
                motor_tau[i] = dr_sample_range_dev(
                    rng, motor_tau_min, motor_tau_max, BASE_K_MOT);
                yaw_torque_scale[i] = dr_sample_range_dev(
                    rng, yaw_torque_min, yaw_torque_max, 1.0f);
                motor_scale_min = fminf(motor_scale_min, motor_thrust_scale[i]);
                motor_tau_sample_max = fmaxf(motor_tau_sample_max, motor_tau[i]);
            }

            if (force_three_plus_one) {
                int weak = (int)floorf(rndf_dev(0.0f, 4.0f, rng));
                if (weak > 3) weak = 3;
                #pragma unroll
                for (int i = 0; i < 4; i++) {
                    motor_thrust_scale[i] = dr_sample_range_dev(rng, 0.96f, 1.06f, 1.0f);
                    motor_tau[i] = dr_sample_range_dev(rng, 0.08f, 0.16f, BASE_K_MOT);
                    yaw_torque_scale[i] = dr_sample_range_dev(rng, 0.90f, 1.10f, 1.0f);
                }
                motor_thrust_scale[weak] = dr_sample_range_dev(rng, 0.78f, 0.88f, 0.84f);
                motor_tau[weak] = dr_sample_range_dev(rng, 0.22f, 0.28f, 0.24f);
                yaw_torque_scale[weak] = dr_sample_range_dev(rng, 0.70f, 0.95f, 0.85f);
            }

            if (force_fast_slow_mismatch) {
                int slow = (int)floorf(rndf_dev(0.0f, 4.0f, rng));
                if (slow > 3) slow = 3;
                int fast = (slow + 2) & 3;
                motor_tau[slow] = dr_sample_range_dev(rng, 0.20f, 0.26f, 0.22f);
                motor_tau[fast] = dr_sample_range_dev(rng, 0.05f, 0.09f, 0.07f);
                motor_thrust_scale[slow] = dr_sample_range_dev(rng, 0.90f, 1.05f, 0.98f);
                motor_thrust_scale[fast] = dr_sample_range_dev(rng, 0.95f, 1.12f, 1.02f);
            }

            motor_scale_min = 2.0f;
            motor_tau_sample_max = 0.0f;
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                motor_scale_min = fminf(motor_scale_min, motor_thrust_scale[i]);
                motor_tau_sample_max = fmaxf(motor_tau_sample_max, motor_tau[i]);
            }

            float inertia_sample_max = fmaxf(ixx_mult, fmaxf(iyy_mult, izz_mult));
            float com_norm = sqrtf(com_x * com_x + com_y * com_y + com_z * com_z);
            int risk = dr_risk_score_dev(
                usable_t2w, mass_mult, inertia_sample_max, motor_scale_min,
                motor_tau_sample_max, com_norm);
            if (risk <= risk_limit) break;
        }

        float sum_motor_scale = 0.0f;
        #pragma unroll
        for (int i = 0; i < 4; i++) sum_motor_scale += motor_thrust_scale[i];
        if (direct_k_thrust_profile) {
            k_thrust_mult = dr_sample_range_dev(
                rng, direct_k_thrust_min, direct_k_thrust_max, 2.91f);
        } else {
            float cap = clampf_dev(normalized_thrust_max, 1e-3f, 1.0f);
            float base_motor_max = BASE_K_THRUST * BASE_MAX_RPM * BASE_MAX_RPM;
            k_thrust_mult = usable_t2w * (BASE_MASS * mass_mult * BASE_GRAVITY)
                / fmaxf(cap * base_motor_max * sum_motor_scale, 1e-6f);
        }
        yaw_drag_mult = 1.0f;
        motor_lag_mult = 0.0f;
        #pragma unroll
        for (int i = 0; i < 4; i++) motor_lag_mult += motor_tau[i] / BASE_K_MOT;
        motor_lag_mult *= 0.25f;
    } else {
        float legacy_mass_min = 1.0f - dr_param_range_dev(cfg, cfg.dr_mass);
        float legacy_mass_max = 1.0f + dr_param_range_dev(cfg, cfg.dr_mass);
        float legacy_inertia_min = 1.0f - dr_param_range_dev(cfg, cfg.dr_inertia);
        float legacy_inertia_max = 1.0f + dr_param_range_dev(cfg, cfg.dr_inertia);
        float legacy_k_thrust_min = 1.0f - dr_param_range_dev(cfg, cfg.dr_k_thrust);
        float legacy_k_thrust_max = 1.0f + dr_param_range_dev(cfg, cfg.dr_k_thrust);
        float legacy_linear_drag_min = 1.0f - dr_param_range_dev(cfg, cfg.dr_linear_drag);
        float legacy_linear_drag_max = 1.0f + dr_param_range_dev(cfg, cfg.dr_linear_drag);
        float legacy_yaw_drag_min = 1.0f - dr_param_range_dev(cfg, cfg.dr_yaw_drag);
        float legacy_yaw_drag_max = 1.0f + dr_param_range_dev(cfg, cfg.dr_yaw_drag);
        float legacy_motor_lag_min = 1.0f - dr_param_range_dev(cfg, cfg.dr_motor_lag);
        float legacy_motor_lag_max = 1.0f + dr_param_range_dev(cfg, cfg.dr_motor_lag);
        adr_legacy_apply_bounds_and_probe_dev(
            s, cfg, rng, &legacy_mass_min, &legacy_mass_max,
            &legacy_inertia_min, &legacy_inertia_max,
            &legacy_k_thrust_min, &legacy_k_thrust_max,
            &legacy_linear_drag_min, &legacy_linear_drag_max,
            &legacy_yaw_drag_min, &legacy_yaw_drag_max,
            &legacy_motor_lag_min, &legacy_motor_lag_max,
            &com_xy_range, &com_z_range);

        mass_mult = dr_sample_range_dev(rng, legacy_mass_min, legacy_mass_max, 1.0f);
        ixx_mult = dr_sample_range_dev(rng, legacy_inertia_min, legacy_inertia_max, 1.0f);
        iyy_mult = dr_sample_range_dev(rng, legacy_inertia_min, legacy_inertia_max, 1.0f);
        izz_mult = dr_sample_range_dev(rng, legacy_inertia_min, legacy_inertia_max, 1.0f);
        k_thrust_mult = dr_sample_range_dev(rng, legacy_k_thrust_min, legacy_k_thrust_max, 1.0f);
        linear_drag_mult = dr_sample_range_dev(rng, legacy_linear_drag_min, legacy_linear_drag_max, 1.0f);
        yaw_drag_mult = dr_sample_range_dev(rng, legacy_yaw_drag_min, legacy_yaw_drag_max, 1.0f);
        motor_lag_mult = dr_sample_range_dev(rng, legacy_motor_lag_min, legacy_motor_lag_max, 1.0f);
        #pragma unroll
        for (int i = 0; i < 4; i++) motor_tau[i] = BASE_K_MOT * motor_lag_mult;
        com_x = rndf_dev(-com_xy_range, com_xy_range, rng);
        com_y = rndf_dev(-com_xy_range, com_xy_range, rng);
        com_z = rndf_dev(-com_z_range, com_z_range, rng);
    }

    p->mass = BASE_MASS * mass_mult;
    p->ixx = BASE_IXX * ixx_mult;
    p->iyy = BASE_IYY * iyy_mult;
    p->izz = BASE_IZZ * izz_mult;
    p->k_thrust = BASE_K_THRUST * k_thrust_mult;
    p->k_ang_damp = BASE_K_ANG_DAMP * angular_damping_mult;
    p->k_drag = BASE_K_DRAG * yaw_drag_mult;
    p->b_drag = BASE_B_DRAG * linear_drag_mult;
    p->gravity = BASE_GRAVITY;
    p->max_rpm = BASE_MAX_RPM;
    p->max_vel = BASE_MAX_VEL;
    p->max_omega = BASE_MAX_OMEGA;
    p->k_mot = BASE_K_MOT * motor_lag_mult;
    p->action_scale = cfg.action_scale;
    p->action_mode = cfg.action_mode;
    p->normalized_thrust_min = cfg.normalized_thrust_min;
    p->normalized_thrust_max = clampf_dev(normalized_thrust_max, 0.0f, 1.0f);
    p->com_x = com_x;
    p->com_y = com_y;
    p->com_z = com_z;
    p->mass_mult = mass_mult;
    p->ixx_mult = ixx_mult;
    p->iyy_mult = iyy_mult;
    p->izz_mult = izz_mult;
    p->k_thrust_mult = k_thrust_mult;
    p->linear_drag_mult = linear_drag_mult;
    p->yaw_drag_mult = yaw_drag_mult;
    p->motor_lag_mult = motor_lag_mult;
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        p->motor_thrust_scale[i] = motor_thrust_scale[i];
        p->motor_tau[i] = motor_tau[i];
        p->yaw_torque_scale[i] = yaw_torque_scale[i];
    }

    p->motor_x[0] = BASE_MOTOR_FL_X - com_x;
    p->motor_y[0] = BASE_MOTOR_FL_Y - com_y;
    p->yaw_sign[0] = BASE_YAW_SIGN_FL;
    p->motor_x[1] = BASE_MOTOR_FR_X - com_x;
    p->motor_y[1] = BASE_MOTOR_FR_Y - com_y;
    p->yaw_sign[1] = BASE_YAW_SIGN_FR;
    p->motor_x[2] = BASE_MOTOR_RL_X - com_x;
    p->motor_y[2] = BASE_MOTOR_RL_Y - com_y;
    p->yaw_sign[2] = BASE_YAW_SIGN_RL;
    p->motor_x[3] = BASE_MOTOR_RR_X - com_x;
    p->motor_y[3] = BASE_MOTOR_RR_Y - com_y;
    p->yaw_sign[3] = BASE_YAW_SIGN_RR;
}

__device__ void compute_derivatives_dev(const DroneCudaState* s, const DroneCudaParams* p,
                                        const float actions[4], DroneCudaDerivative* d) {
    float trim[4];
    hover_trim_thrusts_dev(p, trim);
    float target_rpms[4];
    bool hover_equilibrium = p->action_mode == M4D_ACTION_HOVER_TRIM;
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        float target_thrust;
        float max_thrust_i = max_motor_thrust_i_dev(p, i);
        if (p->action_mode == M4D_ACTION_NORMALIZED_THRUST) {
            target_thrust = normalized_thrust_command_dev(p, actions[i]) * max_thrust_i;
        } else {
            float action = clampf_dev(actions[i] * p->action_scale, -1.0f, 1.0f);
            if (fabsf(action) > 1e-8f) hover_equilibrium = false;
            target_thrust = action >= 0.0f
                ? trim[i] + action * (max_thrust_i - trim[i])
                : trim[i] + action * trim[i];
        }
        target_rpms[i] = thrust_to_rpm_i_dev(p, i, target_thrust);
        d->rpm_dot[i] = (1.0f / fmaxf(p->motor_tau[i], 1e-4f)) * (target_rpms[i] - s->rpms[i]);
        if (fabsf(target_rpms[i] - s->rpms[i]) > 1e-3f) hover_equilibrium = false;
    }

    float T[4];
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        float rpm = fmaxf(s->rpms[i], 0.0f);
        T[i] = motor_thrust_coeff_dev(p, i) * rpm * rpm;
    }

    float3 f_prop_body = make_float3(0.0f, 0.0f, T[0] + T[1] + T[2] + T[3]);
    float3 f_prop = quat_rotate_dev(s->quat, f_prop_body);
    float3 f_aero = make_float3(-p->b_drag * s->vel.x,
                                -p->b_drag * s->vel.y,
                                -p->b_drag * s->vel.z);

    d->vel = s->vel;
    d->v_dot = make_float3((f_prop.x + f_aero.x) / p->mass,
                           (f_prop.y + f_aero.y) / p->mass,
                           ((f_prop.z + f_aero.z) / p->mass) - p->gravity);

    float4 omega_q = quat_dev(0.0f, s->omega.x, s->omega.y, s->omega.z);
    d->q_dot = quat_scale_dev(quat_mul_dev(s->quat, omega_q), 0.5f);

    float3 tau_prop = make_float3(0.0f, 0.0f, 0.0f);
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        tau_prop.x += p->motor_y[i] * T[i];
        tau_prop.y += -p->motor_x[i] * T[i];
        tau_prop.z += p->k_drag * p->yaw_torque_scale[i] * p->yaw_sign[i] * T[i];
    }
    if (hover_equilibrium) {
        tau_prop = make_float3(0.0f, 0.0f, 0.0f);
    }

    float3 tau_aero = make_float3(-p->k_ang_damp * s->omega.x,
                                  -p->k_ang_damp * s->omega.y,
                                  -p->k_ang_damp * s->omega.z);
    float3 tau_iner = make_float3(
        (p->iyy - p->izz) * s->omega.y * s->omega.z,
        (p->izz - p->ixx) * s->omega.z * s->omega.x,
        (p->ixx - p->iyy) * s->omega.x * s->omega.y);

    d->w_dot = make_float3((tau_prop.x + tau_aero.x + tau_iner.x) / p->ixx,
                           (tau_prop.y + tau_aero.y + tau_iner.y) / p->iyy,
                           (tau_prop.z + tau_aero.z + tau_iner.z) / p->izz);
}

__device__ void state_step_dev(const DroneCudaState* initial, const DroneCudaDerivative* d,
                               float dt, DroneCudaState* out) {
    *out = *initial;
    out->pos = add3_dev(initial->pos, scale3_dev(d->vel, dt));
    out->vel = add3_dev(initial->vel, scale3_dev(d->v_dot, dt));
    out->quat = quat_add_dev(initial->quat, quat_scale_dev(d->q_dot, dt));
    out->omega = add3_dev(initial->omega, scale3_dev(d->w_dot, dt));
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        out->rpms[i] = initial->rpms[i] + d->rpm_dot[i] * dt;
    }
    quat_normalize_dev(&out->quat);
}

__device__ void rk4_step_dev(DroneCudaState* s, const DroneCudaParams* p, const float actions[4],
                             float dt) {
    DroneCudaDerivative k1, k2, k3, k4;
    DroneCudaState tmp;

    compute_derivatives_dev(s, p, actions, &k1);
    state_step_dev(s, &k1, dt * 0.5f, &tmp);
    compute_derivatives_dev(&tmp, p, actions, &k2);
    state_step_dev(s, &k2, dt * 0.5f, &tmp);
    compute_derivatives_dev(&tmp, p, actions, &k3);
    state_step_dev(s, &k3, dt, &tmp);
    compute_derivatives_dev(&tmp, p, actions, &k4);

    float dt_6 = dt / 6.0f;
    s->pos.x += (k1.vel.x + 2.0f * k2.vel.x + 2.0f * k3.vel.x + k4.vel.x) * dt_6;
    s->pos.y += (k1.vel.y + 2.0f * k2.vel.y + 2.0f * k3.vel.y + k4.vel.y) * dt_6;
    s->pos.z += (k1.vel.z + 2.0f * k2.vel.z + 2.0f * k3.vel.z + k4.vel.z) * dt_6;
    s->vel.x += (k1.v_dot.x + 2.0f * k2.v_dot.x + 2.0f * k3.v_dot.x + k4.v_dot.x) * dt_6;
    s->vel.y += (k1.v_dot.y + 2.0f * k2.v_dot.y + 2.0f * k3.v_dot.y + k4.v_dot.y) * dt_6;
    s->vel.z += (k1.v_dot.z + 2.0f * k2.v_dot.z + 2.0f * k3.v_dot.z + k4.v_dot.z) * dt_6;
    s->quat.x += (k1.q_dot.x + 2.0f * k2.q_dot.x + 2.0f * k3.q_dot.x + k4.q_dot.x) * dt_6;
    s->quat.y += (k1.q_dot.y + 2.0f * k2.q_dot.y + 2.0f * k3.q_dot.y + k4.q_dot.y) * dt_6;
    s->quat.z += (k1.q_dot.z + 2.0f * k2.q_dot.z + 2.0f * k3.q_dot.z + k4.q_dot.z) * dt_6;
    s->quat.w += (k1.q_dot.w + 2.0f * k2.q_dot.w + 2.0f * k3.q_dot.w + k4.q_dot.w) * dt_6;
    s->omega.x += (k1.w_dot.x + 2.0f * k2.w_dot.x + 2.0f * k3.w_dot.x + k4.w_dot.x) * dt_6;
    s->omega.y += (k1.w_dot.y + 2.0f * k2.w_dot.y + 2.0f * k3.w_dot.y + k4.w_dot.y) * dt_6;
    s->omega.z += (k1.w_dot.z + 2.0f * k2.w_dot.z + 2.0f * k3.w_dot.z + k4.w_dot.z) * dt_6;
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        s->rpms[i] +=
            (k1.rpm_dot[i] + 2.0f * k2.rpm_dot[i] + 2.0f * k3.rpm_dot[i] + k4.rpm_dot[i]) * dt_6;
    }
    quat_normalize_dev(&s->quat);
}

__device__ void move_drone_dev(DroneCudaState* s, const DroneCudaParams* p, float actions[4]) {
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        actions[i] = clampf_dev(actions[i], -1.0f, 1.0f);
    }

    for (int sub = 0; sub < DRONE_ACTION_SUBSTEPS; sub++) {
        rk4_step_dev(s, p, actions, DRONE_DT);
        s->vel.x = clampf_dev(s->vel.x, -p->max_vel, p->max_vel);
        s->vel.y = clampf_dev(s->vel.y, -p->max_vel, p->max_vel);
        s->vel.z = clampf_dev(s->vel.z, -p->max_vel, p->max_vel);
        s->omega.x = clampf_dev(s->omega.x, -p->max_omega, p->max_omega);
        s->omega.y = clampf_dev(s->omega.y, -p->max_omega, p->max_omega);
        s->omega.z = clampf_dev(s->omega.z, -p->max_omega, p->max_omega);
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            s->rpms[i] = clampf_dev(s->rpms[i], 0.0f, p->max_rpm);
        }
    }
}

__device__ float hover_potential_dev(const DroneCudaState* s, const DroneCudaCtx& cfg) {
    float dist = norm3_dev(sub3_dev(s->target_pos, s->pos));
    float vel = norm3_dev(s->vel);
    float omega = norm3_dev(s->omega);
    float d = 1.0f / (1.0f + dist / cfg.hover_dist);
    float v = 1.0f / (1.0f + vel / cfg.hover_vel);
    float w = 1.0f / (1.0f + omega / cfg.hover_omega);
    return d * (0.7f + 0.15f * v + 0.15f * w);
}

__device__ float check_hover_dev(const DroneCudaState* s, const DroneCudaCtx& cfg) {
    float dist = norm3_dev(sub3_dev(s->target_pos, s->pos));
    float vel = norm3_dev(s->vel);
    float omega = norm3_dev(s->omega);
    float d = dist / (cfg.hover_dist * 10.0f);
    float v = vel / (cfg.hover_vel * 10.0f);
    float w = omega / (cfg.hover_omega * 10.0f);
    float score = 1.0f - 0.7f * d - 0.15f * v - 0.15f * w;
    return fmaxf(score, 0.0f);
}

__device__ void set_target_hover_dev(DroneCudaState* s, const DroneCudaCtx& cfg,
                                     curandStatePhilox4_32_10_t* rng) {
    float u = rndf_dev(0.0f, 1.0f, rng);
    float v = rndf_dev(0.0f, 1.0f, rng);
    float z = 2.0f * v - 1.0f;
    float a = 2.0f * DRONE_PI * u;
    float r_xy = sqrtf(fmaxf(0.0f, 1.0f - z * z));
    float3 dir = make_float3(r_xy * cosf(a), r_xy * sinf(a), z);
    float rad = cfg.hover_target_dist * cbrtf(rndf_dev(0.0f, 1.0f, rng));
    float3 p = add3_dev(s->pos, scale3_dev(dir, rad));
    s->target_pos = make_float3(clampf_dev(p.x, -DRONE_MARGIN_X, DRONE_MARGIN_X),
                                clampf_dev(p.y, -DRONE_MARGIN_Y, DRONE_MARGIN_Y),
                                clampf_dev(p.z, -DRONE_MARGIN_Z, DRONE_MARGIN_Z));
    s->target_normal = make_float3(0.0f, 0.0f, 1.0f);
}

__device__ void set_minimal_vision_visible_target_dev(DroneCudaState* s, const DroneCudaCtx& cfg,
                                                      curandStatePhilox4_32_10_t* rng) {
    if (!(cfg.minimal_vision_spawn_visible_target > 0.0f)) return;
    if (!(cfg.minimal_vision_enabled > 0.0f)) return;
    if (cfg.task != DRONE_TASK_HOVER) return;

    float fov = fmaxf(fabsf(cfg.minimal_vision_fov), 0.1f);
    float vfov = fmaxf(fabsf(cfg.minimal_vision_vfov), 0.1f);
    float dist = rndf_dev(0.45f * cfg.hover_target_dist, cfg.hover_target_dist, rng);
    float yaw = rndf_dev(-0.35f * fov, 0.35f * fov, rng);
    float pitch = rndf_dev(-0.25f * vfov, 0.25f * vfov, rng);
    float cp = cosf(pitch);
    float3 body = make_float3(dist * cp * cosf(yaw), dist * cp * sinf(yaw), dist * sinf(pitch));
    float3 world = quat_rotate_dev(s->quat, body);
    float3 p = add3_dev(s->pos, world);
    s->target_pos = make_float3(
        clampf_dev(p.x, -DRONE_MARGIN_X, DRONE_MARGIN_X),
        clampf_dev(p.y, -DRONE_MARGIN_Y, DRONE_MARGIN_Y),
        clampf_dev(p.z, -DRONE_MARGIN_Z, DRONE_MARGIN_Z));
    s->target_normal = make_float3(0.0f, 0.0f, 1.0f);
}

__device__ __forceinline__ int race_ring_count_dev(const DroneCudaCtx& cfg) {
    int n = cfg.max_rings;
    if (cfg.race_track_mode >= 1.0f && n > 7) n = 7;
    if (n < 1) n = 1;
    if (n > DRONE_CUDA_MAX_RINGS) n = DRONE_CUDA_MAX_RINGS;
    return n;
}

__device__ __forceinline__ float3 clamp_world_dev(float3 p) {
    return make_float3(
        clampf_dev(p.x, -DRONE_MARGIN_X, DRONE_MARGIN_X),
        clampf_dev(p.y, -DRONE_MARGIN_Y, DRONE_MARGIN_Y),
        clampf_dev(p.z, -DRONE_MARGIN_Z, DRONE_MARGIN_Z));
}

__device__ float3 random_unit_dev(curandStatePhilox4_32_10_t* rng) {
    float u = rndf_dev(0.0f, 1.0f, rng);
    float v = rndf_dev(0.0f, 1.0f, rng);
    float z = 2.0f * v - 1.0f;
    float a = 2.0f * DRONE_PI * u;
    float r_xy = sqrtf(fmaxf(0.0f, 1.0f - z * z));
    return make_float3(r_xy * cosf(a), r_xy * sinf(a), z);
}

__device__ __forceinline__ float3 race_course_dir_from_yaw_pitch_dev(float yaw, float pitch) {
    float cp = cosf(pitch);
    return normalize3_dev(make_float3(
        cp * cosf(yaw),
        cp * sinf(yaw),
        sinf(pitch)), make_float3(1.0f, 0.0f, 0.0f));
}

__device__ float3 race_course_next_visible_dir_dev(float3 prev_dir, const DroneCudaCtx& cfg,
                                                  curandStatePhilox4_32_10_t* rng) {
    prev_dir = normalize3_dev(prev_dir, make_float3(1.0f, 0.0f, 0.0f));
    float base_yaw = atan2f(prev_dir.y, prev_dir.x);
    float horiz = sqrtf(prev_dir.x * prev_dir.x + prev_dir.y * prev_dir.y);
    float base_pitch = atan2f(prev_dir.z, fmaxf(horiz, 1e-3f));

    float max_yaw_delta = cfg.race_course_yaw_delta > 0.0f
        ? cfg.race_course_yaw_delta
        : fminf(0.38f, 0.18f * fmaxf(fabsf(cfg.minimal_vision_fov), 0.1f));
    float max_pitch_delta = cfg.race_course_pitch_delta > 0.0f
        ? cfg.race_course_pitch_delta
        : fminf(0.20f, 0.12f * fmaxf(fabsf(cfg.minimal_vision_vfov), 0.1f));
    float pitch_limit = cfg.race_course_pitch_limit > 0.0f
        ? cfg.race_course_pitch_limit
        : 0.30f;
    float yaw = base_yaw + rndf_dev(-max_yaw_delta, max_yaw_delta, rng);
    float pitch = clampf_dev(base_pitch + rndf_dev(-max_pitch_delta, max_pitch_delta, rng),
                             -pitch_limit, pitch_limit);
    return race_course_dir_from_yaw_pitch_dev(yaw, pitch);
}

__device__ __forceinline__ float3 race_course_side_dev(float3 dir) {
    float3 horiz = normalize3_dev(make_float3(dir.x, dir.y, 0.0f),
                                  make_float3(1.0f, 0.0f, 0.0f));
    return normalize3_dev(make_float3(-horiz.y, horiz.x, 0.0f),
                          make_float3(0.0f, 1.0f, 0.0f));
}

__device__ __forceinline__ float4 race_reset_quat_dev(float yaw, float pitch, float roll) {
    float4 q_yaw = quat_from_axis_angle_dev(make_float3(0.0f, 0.0f, 1.0f), yaw);
    float4 q_pitch = quat_from_axis_angle_dev(make_float3(0.0f, 1.0f, 0.0f), pitch);
    float4 q_roll = quat_from_axis_angle_dev(make_float3(1.0f, 0.0f, 0.0f), roll);
    float4 q = quat_mul_dev(q_yaw, quat_mul_dev(q_pitch, q_roll));
    quat_normalize_dev(&q);
    return q;
}

__device__ __forceinline__ float3 race_swift_like_base_pos_dev(int idx) {
    switch (idx) {
        case 0: return make_float3(-0.60f, -0.86f, 3.68f);
        case 1: return make_float3( 9.00f,  6.45f, 1.05f);
        case 2: return make_float3( 8.85f, -3.80f, 1.05f);
        case 3: return make_float3(-4.30f, -5.60f, 3.40f);
        case 4: return make_float3(-4.30f, -5.60f, 1.42f);
        case 5: return make_float3( 4.50f, -0.45f, 1.05f);
        default: return make_float3(-1.95f, 6.81f, 1.05f);
    }
}

__device__ __forceinline__ float race_swift_like_base_yaw_dev(int idx) {
    switch (idx) {
        case 0: return -0.34906585f;
        case 1: return  0.0f;
        case 2: return -2.26892803f;
        case 3: return -DRONE_PI;
        case 4: return  0.0f;
        case 5: return  1.39626340f;
        default: return -2.61799388f;
    }
}

__device__ __forceinline__ float3 rotate_yaw_dev(float3 p, float yaw) {
    float c = cosf(yaw);
    float s = sinf(yaw);
    return make_float3(c * p.x - s * p.y, s * p.x + c * p.y, p.z);
}

__device__ void set_target_race_dev(DroneCudaState* s) {
    int idx = s->buffer_idx;
    if (idx < 0) idx = 0;
    if (idx >= s->buffer_size) idx = s->buffer_size - 1;
    s->target_pos = s->ring_pos[idx];
    s->target_normal = s->ring_normal[idx];
}

__device__ void reset_race_course_dev(DroneCudaState* s, const DroneCudaCtx& cfg,
                                      curandStatePhilox4_32_10_t* rng) {
    int n = race_ring_count_dev(cfg);
    s->buffer_size = n;
    s->buffer_idx = 0;

    if (cfg.race_track_mode >= 1.0f) {
        bool randomized = cfg.race_track_mode >= 2.0f;
        float global_yaw = randomized ? rndf_dev(-DRONE_PI, DRONE_PI, rng) : 0.0f;
        float scale = randomized ? rndf_dev(0.90f, 1.10f, rng) : 1.0f;
        float mirror = randomized && rndf_dev(0.0f, 1.0f, rng) < 0.5f ? -1.0f : 1.0f;
        float3 offset = randomized
            ? make_float3(rndf_dev(-3.0f, 3.0f, rng), rndf_dev(-3.0f, 3.0f, rng),
                          rndf_dev(-0.15f, 0.15f, rng))
            : make_float3(0.0f, 0.0f, 0.0f);

        for (int i = 0; i < DRONE_CUDA_MAX_RINGS; i++) {
            if (i >= n) break;
            float3 p = race_swift_like_base_pos_dev(i);
            p.y *= mirror;
            p = scale3_dev(p, scale);
            p = rotate_yaw_dev(p, global_yaw);
            p = clamp_world_dev(add3_dev(p, offset));

            float yaw = global_yaw + mirror * race_swift_like_base_yaw_dev(i);
            float3 normal = normalize3_dev(make_float3(cosf(yaw), sinf(yaw), 0.0f),
                                           make_float3(1.0f, 0.0f, 0.0f));
            s->ring_pos[i] = p;
            s->ring_normal[i] = normal;
            s->ring_radius[i] = DRONE_RING_RADIUS;
        }

        set_target_race_dev(s);
        return;
    }

    float segment = fmaxf(cfg.hover_target_dist, 4.0f);
    float fov = fmaxf(fabsf(cfg.minimal_vision_fov), 0.1f);
    float vfov = fmaxf(fabsf(cfg.minimal_vision_vfov), 0.1f);
    float3 prev_pos = s->pos;
    float3 prev_dir = quat_rotate_dev(s->quat, make_float3(1.0f, 0.0f, 0.0f));
    prev_dir = normalize3_dev(prev_dir, make_float3(1.0f, 0.0f, 0.0f));

    for (int i = 0; i < DRONE_CUDA_MAX_RINGS; i++) {
        if (i >= n) break;

        float3 ring_pos;
        float3 normal;
        if (i == 0 && cfg.minimal_vision_spawn_visible_target > 0.0f) {
            float dist = rndf_dev(0.55f * segment, 0.90f * segment, rng);
            float yaw = rndf_dev(-0.18f * fov, 0.18f * fov, rng);
            float pitch = rndf_dev(-0.10f * vfov, 0.10f * vfov, rng);
            float cp = cosf(pitch);
            float3 body = make_float3(dist * cp * cosf(yaw),
                                       dist * cp * sinf(yaw),
                                       dist * sinf(pitch));
            float3 world = quat_rotate_dev(s->quat, body);
            ring_pos = clamp_world_dev(add3_dev(s->pos, world));
            normal = normalize3_dev(world, prev_dir);
        } else if (cfg.minimal_vision_spawn_visible_target > 0.0f) {
            normal = race_course_next_visible_dir_dev(prev_dir, cfg, rng);
            float spacing_lo = cfg.race_course_spacing_min > 0.0f
                ? cfg.race_course_spacing_min
                : fmaxf(3.0f, 0.75f * clampf_dev(segment, 3.0f, 5.0f));
            float spacing_hi = cfg.race_course_spacing_max > 0.0f
                ? cfg.race_course_spacing_max
                : clampf_dev(segment, 3.0f, 5.0f);
            spacing_hi = fmaxf(spacing_hi, spacing_lo);
            float dist = rndf_dev(spacing_lo, spacing_hi, rng);
            float dz_max = cfg.race_course_dz_max > 0.0f ? cfg.race_course_dz_max : 0.35f;
            float dz = clampf_dev(dist * normal.z, -dz_max, dz_max);
            float xy = sqrtf(fmaxf(0.0f, dist * dist - dz * dz));
            float3 prev_horiz = normalize3_dev(make_float3(prev_dir.x, prev_dir.y, 0.0f),
                                               make_float3(1.0f, 0.0f, 0.0f));
            float3 horiz = normalize3_dev(make_float3(normal.x, normal.y, 0.0f), prev_horiz);
            ring_pos = clamp_world_dev(add3_dev(prev_pos, make_float3(
                horiz.x * xy,
                horiz.y * xy,
                dz)));
            normal = normalize3_dev(sub3_dev(ring_pos, prev_pos), normal);
        } else {
            ring_pos = make_float3(
                rndf_dev(-DRONE_GRID_X + 2.0f * DRONE_RING_RADIUS,
                         DRONE_GRID_X - 2.0f * DRONE_RING_RADIUS, rng),
                rndf_dev(-DRONE_GRID_Y + 2.0f * DRONE_RING_RADIUS,
                         DRONE_GRID_Y - 2.0f * DRONE_RING_RADIUS, rng),
                rndf_dev(-DRONE_GRID_Z + 2.0f * DRONE_RING_RADIUS,
                         DRONE_GRID_Z - 2.0f * DRONE_RING_RADIUS, rng));
            normal = random_unit_dev(rng);
        }

        s->ring_pos[i] = ring_pos;
        s->ring_normal[i] = normal;
        s->ring_radius[i] = DRONE_RING_RADIUS;
        prev_pos = ring_pos;
        prev_dir = normal;
    }

    set_target_race_dev(s);
}

__device__ int race_random_target_idx_dev(const DroneCudaState* s,
                                          curandStatePhilox4_32_10_t* rng) {
    if (s->buffer_size <= 1) return 0;
    int idx = 1 + (int)floorf(rndf_dev(0.0f, (float)(s->buffer_size - 1), rng));
    if (idx < 1) idx = 1;
    if (idx >= s->buffer_size) idx = s->buffer_size - 1;
    return idx;
}

__device__ int race_segment_target_idx_dev(const DroneCudaState* s, const DroneCudaCtx& cfg,
                                           curandStatePhilox4_32_10_t* rng) {
    if (s->buffer_size <= 1) return 0;
    int hard_idx = (int)floorf(cfg.race_hard_gate_idx + 0.5f);
    float hard_prob = clampf_dev(cfg.race_hard_gate_prob, 0.0f, 1.0f);
    if (hard_idx >= 0 && hard_idx < s->buffer_size && rndf_dev(0.0f, 1.0f, rng) < hard_prob) {
        return hard_idx;
    }
    int idx = (int)floorf(rndf_dev(0.0f, (float)s->buffer_size, rng));
    if (idx < 0) idx = 0;
    if (idx >= s->buffer_size) idx = s->buffer_size - 1;
    return idx;
}

__device__ __forceinline__ int race_isb_gate_offset_dev(int agent_idx, int gate) {
    return agent_idx * DRONE_GATE_DEBUG_MAX + gate;
}

__device__ __forceinline__ int race_isb_state_offset_dev(int agent_idx, int gate, int slot) {
    return (race_isb_gate_offset_dev(agent_idx, gate) * DRONE_ISB_CAPACITY) + slot;
}

__device__ float race_pass_margin_dev(const DroneCudaState* s, int gate_idx) {
    if (gate_idx < 0 || gate_idx >= s->buffer_size) return -FLT_MAX;
    float3 ring_pos = s->ring_pos[gate_idx];
    float3 ring_normal = s->ring_normal[gate_idx];
    float prev_dot = dot3_dev(sub3_dev(s->prev_pos, ring_pos), ring_normal);
    float3 dir = sub3_dev(s->pos, s->prev_pos);
    float denom = dot3_dev(ring_normal, dir);
    if (fabsf(denom) < 1e-9f) return -FLT_MAX;
    float t = -prev_dot / denom;
    float3 intersection = add3_dev(s->prev_pos, scale3_dev(dir, t));
    float dist = norm3_dev(sub3_dev(intersection, ring_pos));
    return s->ring_radius[gate_idx] - dist;
}

// Keep these constants mirrored with ocean/drone/drone.h. The physics batch
// edits both files before each rebuild so CUDA training uses the same setup as
// the CPU renderer/eval path.
#define M4D_TURN_SPEED_LIMIT_ENABLED 0
#define M4D_TURN_SPEED_LIMIT 8.5f
#define M4D_TURN_SPEED_START 12.0f
#define M4D_TURN_MIN_ANGLE 0.70f

__device__ __forceinline__ float race_smoothstep01_dev(float x) {
    x = clampf_dev(x, 0.0f, 1.0f);
    return x * x * (3.0f - 2.0f * x);
}

__device__ __forceinline__ float race_turn_angle_at_gate_math_dev(
        const DroneCudaState* s, int idx) {
    if (s->buffer_size <= 0) return 0.0f;
    if (idx <= 0 || idx + 1 >= s->buffer_size) return 0.0f;

    float3 prev = s->ring_pos[idx - 1];
    float3 cur = s->ring_pos[idx];
    float3 next = s->ring_pos[idx + 1];

    float3 in_dir = normalize3_dev(sub3_dev(cur, prev), s->ring_normal[idx]);
    float3 out_dir = normalize3_dev(sub3_dev(next, cur), s->ring_normal[idx]);
    return acosf(clampf_dev(dot3_dev(in_dir, out_dir), -1.0f, 1.0f));
}

__device__ __forceinline__ void race_apply_turn_speed_limit_math_dev(
        DroneCudaState* s, const DroneCudaParams* p) {
    if (!(M4D_TURN_SPEED_LIMIT_ENABLED > 0)) return;
    if (s->buffer_size <= 0) return;

    int idx = race_clamped_gate_idx_dev(s);
    if (idx <= 0 || idx + 1 >= s->buffer_size) return;

    float theta = race_turn_angle_at_gate_math_dev(s, idx);
    if (theta < M4D_TURN_MIN_ANGLE) return;

    float3 cur = s->ring_pos[idx];
    float d = norm3_dev(sub3_dev(cur, s->pos));
    float w = race_smoothstep01_dev((M4D_TURN_SPEED_START - d) / M4D_TURN_SPEED_START);
    if (w <= 0.0f) return;

    float high_cap = p->max_vel;
    float local_cap = high_cap * (1.0f - w) + M4D_TURN_SPEED_LIMIT * w;
    float speed = norm3_dev(s->vel);

    if (speed > local_cap && speed > 1e-6f) {
        s->vel = scale3_dev(s->vel, local_cap / speed);
    }
}

__device__ void race_isb_push_dev(const DroneCudaCtx& cfg, int agent_idx,
                                  const DroneCudaState* s, int target_idx,
                                  float pass_margin) {
    if (!(cfg.race_isb_enabled > 0.0f)) return;
    if (cfg.race_track_mode >= 2.0f) return;
    if (cfg.isb_states == NULL || cfg.isb_counts == NULL || cfg.isb_cursors == NULL) return;
    if (target_idx < 0 || target_idx >= DRONE_GATE_DEBUG_MAX) return;
    float margin_min = cfg.race_isb_margin > 0.0f ? cfg.race_isb_margin : 0.8f;
    if (pass_margin < margin_min) return;

    int gate_offset = race_isb_gate_offset_dev(agent_idx, target_idx);
    int slot = cfg.isb_cursors[gate_offset];
    if (slot < 0 || slot >= DRONE_ISB_CAPACITY) slot = 0;

    DroneCudaIsbState item;
    item.pos = s->pos;
    item.vel = s->vel;
    item.quat = s->quat;
    item.omega = s->omega;
    #pragma unroll
    for (int k = 0; k < 4; k++) item.rpms[k] = s->rpms[k];

    cfg.isb_states[race_isb_state_offset_dev(agent_idx, target_idx, slot)] = item;
    cfg.isb_cursors[gate_offset] = (slot + 1) % DRONE_ISB_CAPACITY;
    if (cfg.isb_counts[gate_offset] < DRONE_ISB_CAPACITY) {
        cfg.isb_counts[gate_offset] += 1;
    }
}

__device__ void race_isb_perturb_state_dev(DroneCudaState* s, const DroneCudaCtx& cfg,
                                           curandStatePhilox4_32_10_t* rng) {
    float pos_xy = cfg.race_isb_pos_xy > 0.0f ? cfg.race_isb_pos_xy : 0.45f;
    float pos_z = cfg.race_isb_z > 0.0f ? cfg.race_isb_z : 0.25f;
    float angle = cfg.race_isb_angle > 0.0f ? cfg.race_isb_angle : 0.18f;
    float vel = cfg.race_isb_vel > 0.0f ? cfg.race_isb_vel : 0.60f;
    float omega = cfg.race_isb_omega > 0.0f ? cfg.race_isb_omega : 0.60f;

    s->pos.x += rndf_dev(-pos_xy, pos_xy, rng);
    s->pos.y += rndf_dev(-pos_xy, pos_xy, rng);
    s->pos.z += rndf_dev(-pos_z, pos_z, rng);
    s->pos = clamp_world_dev(s->pos);

    float4 dq = race_reset_quat_dev(rndf_dev(-angle, angle, rng),
                                    rndf_dev(-angle, angle, rng),
                                    rndf_dev(-angle, angle, rng));
    s->quat = quat_mul_dev(dq, s->quat);
    quat_normalize_dev(&s->quat);

    s->vel.x += rndf_dev(-vel, vel, rng);
    s->vel.y += rndf_dev(-vel, vel, rng);
    s->vel.z += rndf_dev(-vel, vel, rng);
    s->omega.x += rndf_dev(-omega, omega, rng);
    s->omega.y += rndf_dev(-omega, omega, rng);
    s->omega.z += rndf_dev(-omega, omega, rng);
}

__device__ bool race_try_isb_reset_dev(DroneCudaState* s, const DroneCudaCtx& cfg,
                                       curandStatePhilox4_32_10_t* rng,
                                       int agent_idx, int target_idx) {
    if (!(cfg.race_isb_enabled > 0.0f)) return false;
    if (cfg.race_track_mode >= 2.0f) return false;
    if (cfg.isb_states == NULL || cfg.isb_counts == NULL) return false;
    if (target_idx < 0 || target_idx >= DRONE_GATE_DEBUG_MAX) return false;
    if (rndf_dev(0.0f, 1.0f, rng) >= clampf_dev(cfg.race_isb_prob, 0.0f, 1.0f)) return false;

    int gate_offset = race_isb_gate_offset_dev(agent_idx, target_idx);
    int count = cfg.isb_counts[gate_offset];
    if (count <= 0) return false;
    if (count > DRONE_ISB_CAPACITY) count = DRONE_ISB_CAPACITY;
    int slot = (int)floorf(rndf_dev(0.0f, (float)count, rng));
    if (slot >= count) slot = count - 1;
    DroneCudaIsbState item = cfg.isb_states[race_isb_state_offset_dev(agent_idx, target_idx, slot)];

    s->pos = item.pos;
    s->vel = item.vel;
    s->quat = item.quat;
    s->omega = item.omega;
    #pragma unroll
    for (int k = 0; k < 4; k++) s->rpms[k] = item.rpms[k];
    s->buffer_idx = target_idx;
    race_isb_perturb_state_dev(s, cfg, rng);
    set_target_race_dev(s);
    return true;
}

__device__ void race_set_state_between_dev(DroneCudaState* s, int target_idx,
                                           float t_min, float t_max, float lateral,
                                           float yaw_error, float forward_speed,
                                           curandStatePhilox4_32_10_t* rng) {
    float3 prev_pos = s->ring_pos[target_idx - 1];
    float3 target_pos = s->ring_pos[target_idx];
    float3 segment = sub3_dev(target_pos, prev_pos);
    float len = fmaxf(norm3_dev(segment), 1e-3f);
    float3 dir = normalize3_dev(segment, s->ring_normal[target_idx]);
    float3 side = race_course_side_dev(dir);
    float t = rndf_dev(t_min, t_max, rng);
    float side_offset = rndf_dev(-lateral, lateral, rng);
    float z_offset = rndf_dev(-0.15f, 0.15f, rng);
    s->pos = clamp_world_dev(add3_dev(prev_pos, add3_dev(
        scale3_dev(dir, len * t),
        add3_dev(scale3_dev(side, side_offset), make_float3(0.0f, 0.0f, z_offset)))));

    float3 to_target = sub3_dev(target_pos, s->pos);
    float yaw_to_target = atan2f(to_target.y, to_target.x);
    float roll = rndf_dev(-0.04f, 0.04f, rng);
    float pitch = rndf_dev(-0.04f, 0.04f, rng);
    s->quat = race_reset_quat_dev(yaw_to_target - yaw_error, pitch, roll);
    s->vel = add3_dev(scale3_dev(dir, forward_speed),
                      scale3_dev(side, rndf_dev(-0.4f, 0.4f, rng)));
    s->vel.z += rndf_dev(-0.12f, 0.12f, rng);
    s->omega = make_float3(rndf_dev(-0.15f, 0.15f, rng),
                           rndf_dev(-0.15f, 0.15f, rng),
                           rndf_dev(-0.20f, 0.20f, rng));
    s->buffer_idx = target_idx;
}

__device__ void race_set_state_segment_start_dev(DroneCudaState* s, int start_idx,
                                                 float t_min, float t_max, float lateral,
                                                 float yaw_error, float forward_speed,
                                                 curandStatePhilox4_32_10_t* rng) {
    int n = s->buffer_size;
    if (n <= 1) return;
    start_idx = start_idx % n;
    if (start_idx < 0) start_idx += n;
    int target_idx = (start_idx + 1) % n;
    float3 start_pos = s->ring_pos[start_idx];
    float3 target_pos = s->ring_pos[target_idx];
    float3 segment = sub3_dev(target_pos, start_pos);
    float len = fmaxf(norm3_dev(segment), 1e-3f);
    float3 dir = normalize3_dev(segment, s->ring_normal[target_idx]);
    float3 side = race_course_side_dev(dir);
    float t = rndf_dev(t_min, t_max, rng);
    float side_offset = rndf_dev(-lateral, lateral, rng);
    float z_offset = rndf_dev(-0.12f, 0.12f, rng);
    s->pos = clamp_world_dev(add3_dev(start_pos, add3_dev(
        scale3_dev(dir, len * t),
        add3_dev(scale3_dev(side, side_offset), make_float3(0.0f, 0.0f, z_offset)))));

    float3 to_target = sub3_dev(target_pos, s->pos);
    float yaw_to_target = atan2f(to_target.y, to_target.x);
    float roll = rndf_dev(-0.04f, 0.04f, rng);
    float pitch = rndf_dev(-0.04f, 0.04f, rng);
    s->quat = race_reset_quat_dev(yaw_to_target - yaw_error, pitch, roll);
    s->vel = add3_dev(scale3_dev(dir, forward_speed),
                      scale3_dev(side, rndf_dev(-0.35f, 0.35f, rng)));
    s->vel.z += rndf_dev(-0.10f, 0.10f, rng);
    s->omega = make_float3(rndf_dev(-0.15f, 0.15f, rng),
                           rndf_dev(-0.15f, 0.15f, rng),
                           rndf_dev(-0.20f, 0.20f, rng));
    s->buffer_idx = target_idx;
}

__device__ void race_set_state_before_first_dev(DroneCudaState* s,
                                                float lateral, float yaw_error,
                                                float forward_speed,
                                                curandStatePhilox4_32_10_t* rng) {
    float3 target_pos = s->ring_pos[0];
    float3 dir = normalize3_dev(s->ring_normal[0], make_float3(1.0f, 0.0f, 0.0f));
    float3 side = race_course_side_dev(dir);
    float dist = rndf_dev(2.5f, 4.5f, rng);
    float side_offset = rndf_dev(-lateral, lateral, rng);
    float z_offset = rndf_dev(-0.15f, 0.15f, rng);
    s->pos = clamp_world_dev(add3_dev(target_pos, add3_dev(
        scale3_dev(dir, -dist),
        add3_dev(scale3_dev(side, side_offset), make_float3(0.0f, 0.0f, z_offset)))));

    float3 to_target = sub3_dev(target_pos, s->pos);
    float yaw_to_target = atan2f(to_target.y, to_target.x);
    float roll = rndf_dev(-0.04f, 0.04f, rng);
    float pitch = rndf_dev(-0.04f, 0.04f, rng);
    s->quat = race_reset_quat_dev(yaw_to_target - yaw_error, pitch, roll);
    s->vel = add3_dev(scale3_dev(dir, forward_speed),
                      scale3_dev(side, rndf_dev(-0.25f, 0.25f, rng)));
    s->vel.z += rndf_dev(-0.08f, 0.08f, rng);
    s->omega = make_float3(rndf_dev(-0.12f, 0.12f, rng),
                           rndf_dev(-0.12f, 0.12f, rng),
                           rndf_dev(-0.16f, 0.16f, rng));
    s->buffer_idx = 0;
}

__device__ void apply_race_reset_curriculum_dev(DroneCudaState* s, const DroneCudaCtx& cfg,
                                                curandStatePhilox4_32_10_t* rng,
                                                int agent_idx) {
    if (cfg.task != DRONE_TASK_RACE) return;
    if (!(cfg.minimal_vision_spawn_visible_target > 0.0f)) return;
    if (!(cfg.minimal_vision_enabled > 0.0f)) return;
    if (s->buffer_size <= 1) return;

    if (cfg.race_segment_mode >= 1.0f) {
        float yaw_error_frac = cfg.race_reset_yaw_error_frac > 0.0f
            ? cfg.race_reset_yaw_error_frac
            : 0.10f;
        float yaw_error = rndf_dev(-yaw_error_frac * cfg.minimal_vision_fov,
                                   yaw_error_frac * cfg.minimal_vision_fov, rng);
        float t_min = cfg.race_reset_t_min > 0.0f ? cfg.race_reset_t_min : 0.02f;
        float t_max = cfg.race_reset_t_max > 0.0f ? cfg.race_reset_t_max : 0.30f;
        float lateral = cfg.race_reset_lateral > 0.0f ? cfg.race_reset_lateral : 0.35f;
        float speed_min = cfg.race_reset_speed_min > 0.0f ? cfg.race_reset_speed_min : 0.3f;
        float speed_max = cfg.race_reset_speed_max > 0.0f ? cfg.race_reset_speed_max : 1.4f;
        int target_idx = race_segment_target_idx_dev(s, cfg, rng);
        if (race_try_isb_reset_dev(s, cfg, rng, agent_idx, target_idx)) return;
        int start_idx = (target_idx - 1 + s->buffer_size) % s->buffer_size;
        race_set_state_segment_start_dev(s, start_idx, t_min, t_max, lateral,
                                         yaw_error, rndf_dev(speed_min, speed_max, rng), rng);
        set_target_race_dev(s);
        return;
    }

    float start_prob = cfg.race_reset_start_prob > 0.0f ? cfg.race_reset_start_prob : 0.90f;
    float mix = rndf_dev(0.0f, 1.0f, rng);
    if (mix < clampf_dev(start_prob, 0.0f, 1.0f)) {
        if (cfg.race_track_mode >= 1.0f) {
            float yaw_error_frac = cfg.race_reset_yaw_error_frac > 0.0f
                ? cfg.race_reset_yaw_error_frac
                : 0.10f;
            float yaw_error = rndf_dev(-yaw_error_frac * cfg.minimal_vision_fov,
                                       yaw_error_frac * cfg.minimal_vision_fov, rng);
            float lateral = cfg.race_reset_lateral > 0.0f ? cfg.race_reset_lateral : 0.25f;
            float speed_min = cfg.race_reset_speed_min > 0.0f ? cfg.race_reset_speed_min : 0.3f;
            float speed_max = cfg.race_reset_speed_max > 0.0f ? cfg.race_reset_speed_max : 1.2f;
            race_set_state_before_first_dev(s, lateral, yaw_error,
                                            rndf_dev(speed_min, speed_max, rng), rng);
        }
        s->buffer_idx = 0;
        set_target_race_dev(s);
        return;
    }

    int target_idx = race_random_target_idx_dev(s, rng);
    float yaw_error_frac = cfg.race_reset_yaw_error_frac > 0.0f
        ? cfg.race_reset_yaw_error_frac
        : 0.10f;
    float yaw_error = rndf_dev(-yaw_error_frac * cfg.minimal_vision_fov,
                               yaw_error_frac * cfg.minimal_vision_fov, rng);
    float t_min = cfg.race_reset_t_min > 0.0f ? cfg.race_reset_t_min : 0.15f;
    float t_max = cfg.race_reset_t_max > 0.0f ? cfg.race_reset_t_max : 0.45f;
    float lateral = cfg.race_reset_lateral > 0.0f ? cfg.race_reset_lateral : 0.25f;
    float speed_min = cfg.race_reset_speed_min > 0.0f ? cfg.race_reset_speed_min : 0.3f;
    float speed_max = cfg.race_reset_speed_max > 0.0f ? cfg.race_reset_speed_max : 1.2f;
    race_set_state_between_dev(s, target_idx, t_min, t_max, lateral,
                               yaw_error, rndf_dev(speed_min, speed_max, rng), rng);

    set_target_race_dev(s);
}

__device__ int check_ring_dev(const DroneCudaState* s) {
    int idx = s->buffer_idx;
    if (idx < 0 || idx >= s->buffer_size) return 0;

    float3 ring_pos = s->ring_pos[idx];
    float3 ring_normal = s->ring_normal[idx];
    float ring_radius = s->ring_radius[idx];
    float prev_dot = dot3_dev(sub3_dev(s->prev_pos, ring_pos), ring_normal);
    float new_dot = dot3_dev(sub3_dev(s->pos, ring_pos), ring_normal);
    bool valid_dir = prev_dot < 0.0f && new_dot > 0.0f;
    bool invalid_dir = prev_dot > 0.0f && new_dot < 0.0f;
    if (!(valid_dir || invalid_dir)) return 0;

    float3 dir = sub3_dev(s->pos, s->prev_pos);
    float denom = dot3_dev(ring_normal, dir);
    if (fabsf(denom) < 1e-9f) return 0;
    float t = -prev_dot / denom;
    float3 intersection = add3_dev(s->prev_pos, scale3_dev(dir, t));
    float dist = norm3_dev(sub3_dev(intersection, ring_pos));
    if (dist < (ring_radius - 0.5f) && valid_dir) return 1;
    if (dist < (ring_radius + 0.5f)) return -1;
    return 0;
}

__device__ void reset_one_dev(DroneCudaState* s, DroneCudaParams* p, const DroneCudaCtx& cfg,
                              curandStatePhilox4_32_10_t* rng, int agent_idx) {
    DroneCudaState zero = {};
    *s = zero;
    init_params_dev(p, s, cfg, rng);

    float trim[4];
    hover_trim_thrusts_dev(p, trim);
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        s->rpms[i] = thrust_to_rpm_i_dev(p, i, trim[i]);
    }

    float pos_scale = clampf_dev(cfg.reset_pos_scale, 0.0f, 1.0f);
    s->pos = make_float3(rndf_dev(-DRONE_MARGIN_X * pos_scale, DRONE_MARGIN_X * pos_scale, rng),
                         rndf_dev(-DRONE_MARGIN_Y * pos_scale, DRONE_MARGIN_Y * pos_scale, rng),
                         rndf_dev(-DRONE_MARGIN_Z * pos_scale, DRONE_MARGIN_Z * pos_scale, rng));
    s->vel = make_float3(0.0f, 0.0f, 0.0f);
    s->omega = make_float3(0.0f, 0.0f, 0.0f);
    s->quat = quat_dev(1.0f, 0.0f, 0.0f, 0.0f);
    if (cfg.reset_yaw_range > 0.0f) {
        float yaw = rndf_dev(-cfg.reset_yaw_range, cfg.reset_yaw_range, rng);
        s->quat = quat_from_yaw_dev(yaw);
    }
    if (cfg.reset_vel_max > 0.0f) {
        float u = rndf_dev(0.0f, 1.0f, rng);
        float v = rndf_dev(0.0f, 1.0f, rng);
        float z = 2.0f * v - 1.0f;
        float a = 2.0f * DRONE_PI * u;
        float r_xy = sqrtf(fmaxf(0.0f, 1.0f - z * z));
        float3 dir = make_float3(r_xy * cosf(a), r_xy * sinf(a), z);
        float speed = cfg.reset_vel_max * cbrtf(rndf_dev(0.0f, 1.0f, rng));
        s->vel = scale3_dev(dir, speed);
    }
    if (cfg.task == DRONE_TASK_RACE) {
        reset_race_course_dev(s, cfg, rng);
        apply_race_reset_curriculum_dev(s, cfg, rng, agent_idx);
    } else {
        set_target_hover_dev(s, cfg, rng);
        set_minimal_vision_visible_target_dev(s, cfg, rng);
    }
    s->pal_probe_active = curand_uniform(rng) <= clampf_dev(cfg.pal_probe_prob, 0.0f, 1.0f);
    s->prev_pos = s->pos;
    s->prev_potential = hover_potential_dev(s, cfg);
}

__device__ __forceinline__ void apply_pal_probe_dev(
    const DroneCudaCtx& cfg, const DroneCudaState* s, float actions[4]) {
    float amp = fabsf(cfg.pal_probe_amp);
    if (!s->pal_probe_active || cfg.pal_probe_steps <= 0 || amp <= 0.0f) return;
    if (s->episode_length < 0 || s->episode_length >= cfg.pal_probe_steps) return;

    float pattern[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    switch (s->episode_length & 7) {
        case 0: pattern[0] =  1.0f; pattern[1] =  1.0f; pattern[2] =  1.0f; pattern[3] =  1.0f; break;
        case 1: pattern[0] = -1.0f; pattern[1] = -1.0f; pattern[2] = -1.0f; pattern[3] = -1.0f; break;
        case 2: pattern[0] =  1.0f; pattern[1] = -1.0f; pattern[2] =  1.0f; pattern[3] = -1.0f; break;
        case 3: pattern[0] = -1.0f; pattern[1] =  1.0f; pattern[2] = -1.0f; pattern[3] =  1.0f; break;
        case 4: pattern[0] =  1.0f; pattern[1] =  1.0f; pattern[2] = -1.0f; pattern[3] = -1.0f; break;
        case 5: pattern[0] = -1.0f; pattern[1] = -1.0f; pattern[2] =  1.0f; pattern[3] =  1.0f; break;
        case 6: pattern[0] =  1.0f; pattern[1] = -1.0f; pattern[2] = -1.0f; pattern[3] =  1.0f; break;
        default: pattern[0] = -1.0f; pattern[1] =  1.0f; pattern[2] =  1.0f; pattern[3] = -1.0f; break;
    }

    #pragma unroll
    for (int k = 0; k < 4; k++) {
        actions[k] = clampf_dev(actions[k] + amp * pattern[k], -1.0f, 1.0f);
    }
}

__device__ float minimal_vision_blob_intensity_dev(float4 q_inv, float3 pos, float3 target_pos,
                                                   float center_x, float center_y,
                                                   float sigma_x, float sigma_y,
                                                   float depth_gain) {
    float3 to_target_world = sub3_dev(target_pos, pos);
    float3 to_target = quat_rotate_dev(q_inv, to_target_world);
    float dist = norm3_dev(to_target_world);
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

__device__ float minimal_vision_gate_mask_intensity_dev(float4 q_inv, float3 pos,
                                                        float3 gate_pos, float gate_radius,
                                                        float center_x, float center_y,
                                                        float fov, float vfov,
                                                        float depth_gain) {
    float3 rel = quat_rotate_dev(q_inv, sub3_dev(gate_pos, pos));
    if (rel.x <= 0.0f) return 0.0f;

    float yaw = atan2f(rel.y, fmaxf(rel.x, 1e-5f));
    float xy = sqrtf(rel.x * rel.x + rel.y * rel.y);
    float pitch = atan2f(rel.z, fmaxf(xy, 1e-5f));
    float dist = fmaxf(norm3_dev(rel), 1e-3f);
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

__device__ void minimal_vision_pixel_rgb_dev(const DroneCudaState* s, const DroneCudaCtx& cfg,
                                             int px, int py,
                                             float* red, float* green, float* blue) {
    float4 q_inv = quat_inverse_dev(s->quat);
    float fov = fmaxf(fabsf(cfg.minimal_vision_fov), 0.1f);
    float vfov = fmaxf(fabsf(cfg.minimal_vision_vfov), 0.1f);
    float sigma = fmaxf(fabsf(cfg.minimal_vision_sigma), 0.01f);
    float depth_gain = fmaxf(fabsf(cfg.minimal_vision_depth_gain), 0.0f);
    float sigma_y = DRONE_MINIMAL_VISION_HEIGHT <= 1
        ? fmaxf(0.35f * vfov, 1e-3f)
        : fmaxf(sigma * vfov / fov, 0.01f);
    float center_y = DRONE_MINIMAL_VISION_HEIGHT <= 1
        ? 0.0f
        : 0.5f * vfov - ((float)py + 0.5f) * (vfov / (float)DRONE_MINIMAL_VISION_HEIGHT);
    float center_x = -0.5f * fov + ((float)px + 0.5f) * (fov / (float)DRONE_MINIMAL_VISION_WIDTH);

    if (cfg.minimal_vision_gate_mask > 0.0f) {
        if (cfg.task == DRONE_TASK_RACE && s->buffer_size > 0) {
            int idx0 = s->buffer_idx;
            if (idx0 < 0) idx0 = 0;
            if (idx0 >= s->buffer_size) idx0 = s->buffer_size - 1;
            int idx1 = s->buffer_size > 1 ? (idx0 + 1) % s->buffer_size : idx0;
            int idx2 = s->buffer_size > 2 ? (idx0 + 2) % s->buffer_size : idx1;
            *red = minimal_vision_gate_mask_intensity_dev(
                q_inv, s->pos, s->ring_pos[idx0], s->ring_radius[idx0],
                center_x, center_y, fov, vfov, depth_gain);
            *green = s->buffer_size > 1
                ? 0.85f * minimal_vision_gate_mask_intensity_dev(
                    q_inv, s->pos, s->ring_pos[idx1], s->ring_radius[idx1],
                    center_x, center_y, fov, vfov, depth_gain)
                : 0.0f;
            *blue = s->buffer_size > 2
                ? 0.70f * minimal_vision_gate_mask_intensity_dev(
                    q_inv, s->pos, s->ring_pos[idx2], s->ring_radius[idx2],
                    center_x, center_y, fov, vfov, depth_gain)
                : 0.0f;
            return;
        }

        *red = minimal_vision_gate_mask_intensity_dev(
            q_inv, s->pos, s->target_pos, DRONE_RING_RADIUS,
            center_x, center_y, fov, vfov, depth_gain);
        *green = 0.0f;
        *blue = 0.0f;
        return;
    }

    if (cfg.task == DRONE_TASK_RACE && s->buffer_size > 0) {
        int idx0 = s->buffer_idx;
        if (idx0 < 0) idx0 = 0;
        if (idx0 >= s->buffer_size) idx0 = s->buffer_size - 1;
        int idx1 = s->buffer_size > 1 ? (idx0 + 1) % s->buffer_size : idx0;
        int idx2 = s->buffer_size > 2 ? (idx0 + 2) % s->buffer_size : idx1;
        *red = minimal_vision_blob_intensity_dev(q_inv, s->pos, s->ring_pos[idx0],
                                                 center_x, center_y, sigma, sigma_y, depth_gain);
        *green = s->buffer_size > 1
            ? 0.85f * minimal_vision_blob_intensity_dev(q_inv, s->pos, s->ring_pos[idx1],
                                                        center_x, center_y, sigma, sigma_y, depth_gain)
            : 0.0f;
        *blue = s->buffer_size > 2
            ? 0.70f * minimal_vision_blob_intensity_dev(q_inv, s->pos, s->ring_pos[idx2],
                                                        center_x, center_y, sigma, sigma_y, depth_gain)
            : 0.0f;
        return;
    }

    float3 to_target_world = sub3_dev(s->target_pos, s->pos);
    float3 to_target = quat_rotate_dev(q_inv, to_target_world);
    float xy = sqrtf(to_target.x * to_target.x + to_target.y * to_target.y);
    float yaw = atan2f(to_target.y, fmaxf(to_target.x, 1e-3f));
    float pitch = atan2f(to_target.z, fmaxf(xy, 1e-3f));
    float yaw_norm = clampf_dev(yaw / (0.5f * fov), -1.0f, 1.0f);
    float pitch_norm = clampf_dev(pitch / (0.5f * vfov), -1.0f, 1.0f);
    float intensity = minimal_vision_blob_intensity_dev(q_inv, s->pos, s->target_pos,
                                                        center_x, center_y, sigma, sigma_y, depth_gain);
    *red = intensity * (0.65f + 0.35f * clampf_dev(-yaw_norm, 0.0f, 1.0f));
    *green = intensity * (0.65f + 0.35f * clampf_dev(yaw_norm, 0.0f, 1.0f));
    *blue = intensity * (0.55f + 0.45f * (1.0f - fabsf(pitch_norm)));
}

__device__ void compute_minimal_vision_obs_dev(const DroneCudaState* s, const DroneCudaCtx& cfg,
                                               curandStatePhilox4_32_10_t* rng, float* obs) {
    float noise = clampf_dev(fabsf(cfg.minimal_vision_noise), 0.0f, 1.0f);
    float distractors = clampf_dev(fabsf(cfg.minimal_vision_distractors), 0.0f, 1.0f);

    for (int py = 0; py < DRONE_MINIMAL_VISION_HEIGHT; py++) {
        for (int px = 0; px < DRONE_MINIMAL_VISION_WIDTH; px++) {
        float red = 0.0f;
        float green = 0.0f;
        float blue = 0.0f;
        minimal_vision_pixel_rgb_dev(s, cfg, px, py, &red, &green, &blue);

        if (distractors > 0.0f && rndf_dev(0.0f, 1.0f, rng) < 0.03f * distractors) {
            red += distractors * rndf_dev(0.0f, 0.25f, rng);
            green += distractors * rndf_dev(0.0f, 0.25f, rng);
            blue += distractors * rndf_dev(0.0f, 0.25f, rng);
        }
        if (noise > 0.0f) {
            red += rndf_dev(-noise, noise, rng);
            green += rndf_dev(-noise, noise, rng);
            blue += rndf_dev(-noise, noise, rng);
        }

        int out = 3 * (py * DRONE_MINIMAL_VISION_WIDTH + px);
        obs[out + 0] = clampf_dev(red, 0.0f, 1.0f);
        obs[out + 1] = clampf_dev(green, 0.0f, 1.0f);
        obs[out + 2] = clampf_dev(blue, 0.0f, 1.0f);
        }
    }
}

__device__ void record_retina_diagnostics_dev(DroneCudaState* s, const DroneCudaCtx& cfg) {
    float4 q_inv = quat_inverse_dev(s->quat);
    float3 to_target_world = sub3_dev(s->target_pos, s->pos);
    float3 to_target = quat_rotate_dev(q_inv, to_target_world);

    float fov = fmaxf(fabsf(cfg.minimal_vision_fov), 0.1f);
    float vfov = fmaxf(fabsf(cfg.minimal_vision_vfov), 0.1f);
    float sigma = fmaxf(fabsf(cfg.minimal_vision_sigma), 0.01f);
    float depth_gain = fmaxf(fabsf(cfg.minimal_vision_depth_gain), 0.0f);
    float dist = norm3_dev(to_target_world);
    float xy = sqrtf(to_target.x * to_target.x + to_target.y * to_target.y);
    float yaw = atan2f(to_target.y, fmaxf(to_target.x, 1e-3f));
    float pitch = atan2f(to_target.z, fmaxf(xy, 1e-3f));
    float front = to_target.x > 0.0f ? 1.0f : 0.0f;
    float depth = 1.0f / (1.0f + depth_gain * dist);
    float yaw_norm = clampf_dev(yaw / (0.5f * fov), -1.0f, 1.0f);
    float pitch_norm = clampf_dev(pitch / (0.5f * vfov), -1.0f, 1.0f);
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
            minimal_vision_pixel_rgb_dev(s, cfg, px, py, &red, &green, &blue);
            red = clampf_dev(red, 0.0f, 1.0f);
            green = clampf_dev(green, 0.0f, 1.0f);
            blue = clampf_dev(blue, 0.0f, 1.0f);
            float px_energy = red * red + green * green + blue * blue;
            rgb_sum += red + green + blue;
            rgb_sq_sum += px_energy;
            int bucket = (3 * px) / DRONE_MINIMAL_VISION_WIDTH;
            bucket = bucket < 0 ? 0 : (bucket > 2 ? 2 : bucket);
            bucket_energy[bucket] += px_energy;
        }
    }

    float inv_channels = 1.0f / (float)DRONE_MINIMAL_VISION_OBS_SIZE;
    float mean = rgb_sum * inv_channels;
    float energy = rgb_sq_sum * inv_channels;
    float var = fmaxf(0.0f, energy - mean * mean);
    float best = bucket_energy[0];
    int best_idx = 0;
    if (bucket_energy[1] > best) { best = bucket_energy[1]; best_idx = 1; }
    if (bucket_energy[2] > best) { best = bucket_energy[2]; best_idx = 2; }
    float argmax = best > 1e-8f ? (float)best_idx : -1.0f;

    s->target_in_fov_sum += (front > 0.0f && fabsf(yaw) <= 0.5f * fov
        && fabsf(pitch) <= 0.5f * vfov) ? 1.0f : 0.0f;
    s->retina_rgb_mean_sum += mean;
    s->retina_rgb_std_sum += sqrtf(var);
    s->retina_energy_sum += energy;
    s->retina_argmax_sum += argmax;
    if (argmax == 0.0f) s->retina_argmax_left_count += 1.0f;
    else if (argmax == 1.0f) s->retina_argmax_center_count += 1.0f;
    else if (argmax == 2.0f) s->retina_argmax_right_count += 1.0f;
    s->bearing_error_sum += sqrtf(yaw * yaw + pitch * pitch);
    s->distance_to_target_sum += dist;
    s->retina_signal_vs_distance_sum += energy * fmaxf(dist, 1e-3f);
    if (cfg.task == DRONE_TASK_RACE) {
        int gate_idx = race_clamped_gate_idx_dev(s);
        if (gate_idx >= 0 && gate_idx < DRONE_GATE_DEBUG_MAX) {
            float in_fov = (front > 0.0f && fabsf(yaw) <= 0.5f * fov
                && fabsf(pitch) <= 0.5f * vfov) ? 1.0f : 0.0f;
            s->gate_time[gate_idx] += 1.0f;
            s->gate_target_in_fov[gate_idx] += in_fov;
            s->gate_bearing_error[gate_idx] += sqrtf(yaw * yaw + pitch * pitch);
            s->gate_distance_to_target[gate_idx] += dist;
        }
    }
}

__device__ void compute_obs_dev(const DroneCudaState* s, const DroneCudaParams* p,
                                const DroneCudaCtx& cfg, curandStatePhilox4_32_10_t* rng,
                                float* obs) {
    #pragma unroll
    for (int i = 0; i < DRONE_OBS_SIZE; i++) obs[i] = 0.0f;

    int idx = 0;
    float4 q = s->quat;
    float4 q_inv = quat_inverse_dev(q);
    float3 linear_vel_body = quat_rotate_dev(q_inv, s->vel);
    float3 to_target_world = sub3_dev(s->target_pos, s->pos);
    float3 to_target = quat_rotate_dev(q_inv, to_target_world);

    if (!(cfg.minimal_vision_only > 0.0f)) {
        float denom = p->max_vel * 1.7320508f;
        obs[idx++] = linear_vel_body.x / denom;
        obs[idx++] = linear_vel_body.y / denom;
        obs[idx++] = linear_vel_body.z / denom;
        obs[idx++] = s->omega.x / p->max_omega;
        obs[idx++] = s->omega.y / p->max_omega;
        obs[idx++] = s->omega.z / p->max_omega;
        obs[idx++] = q.x;
        obs[idx++] = q.y;
        obs[idx++] = q.z;
        obs[idx++] = q.w;
        obs[idx++] = tanhf(to_target.x * 0.1f);
        obs[idx++] = tanhf(to_target.y * 0.1f);
        obs[idx++] = tanhf(to_target.z * 0.1f);
        obs[idx++] = tanhf(to_target.x * 10.0f);
        obs[idx++] = tanhf(to_target.y * 10.0f);
        obs[idx++] = tanhf(to_target.z * 10.0f);
        float3 normal_body = quat_rotate_dev(q_inv, s->target_normal);
        obs[idx++] = normal_body.x;
        obs[idx++] = normal_body.y;
        obs[idx++] = normal_body.z;
        obs[idx++] = s->rpms[0] / p->max_rpm;
        obs[idx++] = s->rpms[1] / p->max_rpm;
        obs[idx++] = s->rpms[2] / p->max_rpm;
        obs[idx++] = s->rpms[3] / p->max_rpm;
    }

    if (cfg.sensor_noise > 0.0f && !(cfg.minimal_vision_only > 0.0f)) {
        float noise = fminf(fabsf(cfg.sensor_noise), 1.0f);
        #pragma unroll
        for (int i = 0; i < DRONE_STATE_OBS_SIZE; i++) {
            obs[i] = clampf_dev(obs[i] + rndf_dev(-noise, noise, rng), -2.0f, 2.0f);
        }
    }
    if (!(cfg.minimal_vision_only > 0.0f) && cfg.minimal_vision_mask_target > 0.0f) {
        #pragma unroll
        for (int i = 10; i < 19; i++) obs[i] = 0.0f;
    }

    if (cfg.minimal_vision_enabled > 0.0f) {
        int offset = cfg.minimal_vision_only > 0.0f ? 0 : DRONE_STATE_OBS_SIZE;
        compute_minimal_vision_obs_dev(s, cfg, rng, obs + offset);
    }
}

__device__ void record_step_metrics_dev(DroneCudaState* s, const DroneCudaParams* p,
                                        const float raw_actions[4], float r_dist,
                                        float r_hover, float r_shaping, float r_omega,
                                        float r_omega_xy, float r_omega_z, float r_terminal,
                                        float action_delta_mean,
                                        float reset_action_jump) {
    float action_abs_sum = 0.0f;
    float action_clipped_abs_sum = 0.0f;
    float action_max_abs = 0.0f;
    float action_saturation_count = 0.0f;
    float motor_clip_low_count = 0.0f;
    float motor_clip_high_count = 0.0f;
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        float abs_action = fabsf(raw_actions[i]);
        action_abs_sum += abs_action;
        action_max_abs = fmaxf(action_max_abs, abs_action);
        if (abs_action >= 0.99f) action_saturation_count += 1.0f;
        float env_clipped = clampf_dev(raw_actions[i], -1.0f, 1.0f);
        action_clipped_abs_sum += fabsf(env_clipped);
        if (p->action_mode == M4D_ACTION_NORMALIZED_THRUST) {
            float motor_cmd = normalized_thrust_command_dev(p, env_clipped);
            float lo = clampf_dev(p->normalized_thrust_min, 0.0f, 1.0f);
            float hi = clampf_dev(p->normalized_thrust_max, lo, 1.0f);
            if (motor_cmd <= lo + 1e-5f) motor_clip_low_count += 1.0f;
            if (motor_cmd >= hi - 1e-5f) motor_clip_high_count += 1.0f;
        } else {
            float motor_action = clampf_dev(env_clipped * p->action_scale, -1.0f, 1.0f);
            if (motor_action <= -0.99f) motor_clip_low_count += 1.0f;
            if (motor_action >= 0.99f) motor_clip_high_count += 1.0f;
        }
        s->rpm_sum[i] += s->rpms[i];
    }
    s->action_abs_sum += action_abs_sum / 4.0f;
    s->action_clipped_abs_sum += action_clipped_abs_sum / 4.0f;
    s->action_max_abs = fmaxf(s->action_max_abs, action_max_abs);
    s->action_saturation_count += action_saturation_count / 4.0f;
    s->action_delta_sum += action_delta_mean;
    s->reset_action_jump_sum += reset_action_jump;
    if (reset_action_jump > 0.0f) s->reset_action_jump_count += 1.0f;
    s->motor_clip_low_count += motor_clip_low_count / 4.0f;
    s->motor_clip_high_count += motor_clip_high_count / 4.0f;
    s->instrumentation_steps += 1.0f;
    s->r_dist_sum += r_dist;
    s->r_hover_sum += r_hover;
    s->r_shaping_sum += r_shaping;
    s->r_omega_sum += r_omega;
    s->r_omega_xy_sum += r_omega_xy;
    s->r_omega_z_sum += r_omega_z;
    s->r_terminal_sum += r_terminal;
    #pragma unroll
    for (int i = 0; i < 4; i++) s->prev_action[i] = raw_actions[i];
    s->has_prev_action = 1;
}

__device__ void log_done_dev(DroneCudaLog* log, const DroneCudaState* s,
                             const DroneCudaParams* p, const DroneCudaCtx& cfg,
                             bool oob, bool timeout, bool lap_complete) {
    float steps = fmaxf(s->instrumentation_steps, 1.0f);
    atomicAdd(&log->episode_return, s->episode_return);
    atomicAdd(&log->episode_length, (float)s->episode_length);
    atomicAdd(&log->rings_passed, s->rings_passed);
    atomicAdd(&log->collisions, s->collisions);
    atomicAdd(&log->ring_collision, s->ring_collision);
    atomicAdd(&log->oob, oob ? 1.0f : 0.0f);
    atomicAdd(&log->timeout, timeout ? 1.0f : 0.0f);
    atomicAdd(&log->lap_complete, lap_complete ? 1.0f : 0.0f);
    if (cfg.task == DRONE_TASK_RACE) {
        int gate_idx = race_clamped_gate_idx_dev(s);
        if (gate_idx >= 0 && gate_idx < DRONE_GATE_DEBUG_MAX) {
            if (oob) atomicAdd(&log->gate_oob_count[gate_idx], 1.0f);
            if (timeout) atomicAdd(&log->gate_timeout_count[gate_idx], 1.0f);
        }
    }
    if (oob && cfg.task == DRONE_TASK_RACE) {
        float3 next_gate_pos = race_next_gate_pos_dev(s);
        atomicAdd(&log->oob_diag_count, 1.0f);
        atomicAdd(&log->gate_index_at_oob, (float)race_clamped_gate_idx_dev(s));
        atomicAdd(&log->position_norm_at_oob, norm3_dev(s->pos));
        atomicAdd(&log->target_gate_position_norm, norm3_dev(s->target_pos));
        atomicAdd(&log->next_gate_position_norm, norm3_dev(next_gate_pos));
        atomicAdd(&log->distance_from_track_centerline, race_track_centerline_distance_dev(s));
    }
    atomicAdd(&log->score, s->hover_score);
    atomicAdd(&log->perf, s->hover_ema);
    atomicAdd(&log->ema_dist, s->ema_dist);
    atomicAdd(&log->ema_vel, s->ema_vel);
    atomicAdd(&log->ema_omega, s->ema_omega);
    atomicAdd(&log->ema_omega_x, s->ema_omega_x);
    atomicAdd(&log->ema_omega_y, s->ema_omega_y);
    atomicAdd(&log->ema_omega_z, s->ema_omega_z);
    atomicAdd(&log->mean_abs_action, s->action_abs_sum / steps);
    atomicAdd(&log->mean_abs_action_clipped, s->action_clipped_abs_sum / steps);
    atomicAdd(&log->max_abs_action, s->action_max_abs);
    atomicAdd(&log->action_saturation_frac, s->action_saturation_count / steps);
    atomicAdd(&log->mean_abs_delta_action, s->action_delta_sum / steps);
    atomicAdd(&log->reset_action_jump_mean, s->reset_action_jump_count > 0.0f
        ? s->reset_action_jump_sum / s->reset_action_jump_count
        : 0.0f);
    atomicAdd(&log->motor_clip_low_frac, s->motor_clip_low_count / steps);
    atomicAdd(&log->motor_clip_high_frac, s->motor_clip_high_count / steps);

    float trim[4];
    hover_trim_thrusts_dev(p, trim);
    float trim_rpm_sum = 0.0f;
    float trim_rpm_max = 0.0f;
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        float rpm = thrust_to_rpm_i_dev(p, i, trim[i]);
        trim_rpm_sum += rpm;
        trim_rpm_max = fmaxf(trim_rpm_max, rpm);
    }
    atomicAdd(&log->hover_trim_rpm_mean, trim_rpm_sum / 4.0f);
    atomicAdd(&log->hover_trim_rpm_max, trim_rpm_max);
    atomicAdd(&log->hover_trim_rpm_frac_of_max, trim_rpm_max / fmaxf(p->max_rpm, 1.0f));
    atomicAdd(&log->mean_rpm_FL, s->rpm_sum[0] / steps);
    atomicAdd(&log->mean_rpm_FR, s->rpm_sum[1] / steps);
    atomicAdd(&log->mean_rpm_RL, s->rpm_sum[2] / steps);
    atomicAdd(&log->mean_rpm_RR, s->rpm_sum[3] / steps);
    atomicAdd(&log->target_in_fov_frac, s->target_in_fov_sum / steps);
    atomicAdd(&log->retina_rgb_mean, s->retina_rgb_mean_sum / steps);
    atomicAdd(&log->retina_rgb_std, s->retina_rgb_std_sum / steps);
    atomicAdd(&log->retina_energy, s->retina_energy_sum / steps);
    atomicAdd(&log->retina_left_center_right_argmax, s->retina_argmax_sum / steps);
    atomicAdd(&log->retina_argmax_left_frac, s->retina_argmax_left_count / steps);
    atomicAdd(&log->retina_argmax_center_frac, s->retina_argmax_center_count / steps);
    atomicAdd(&log->retina_argmax_right_frac, s->retina_argmax_right_count / steps);
    atomicAdd(&log->bearing_error_to_target, s->bearing_error_sum / steps);
    atomicAdd(&log->distance_to_target, s->distance_to_target_sum / steps);
    atomicAdd(&log->retina_signal_vs_distance, s->retina_signal_vs_distance_sum / steps);
    atomicAdd(&log->r_dist, s->r_dist_sum);
    atomicAdd(&log->r_hover, s->r_hover_sum);
    atomicAdd(&log->r_shaping, s->r_shaping_sum);
    atomicAdd(&log->r_omega, s->r_omega_sum);
    atomicAdd(&log->r_omega_xy, s->r_omega_xy_sum);
    atomicAdd(&log->r_omega_z, s->r_omega_z_sum);
    atomicAdd(&log->r_terminal, s->r_terminal_sum);
    atomicAdd(&log->mass_mult_mean, p->mass_mult);
    atomicAdd(&log->ixx_mult_mean, p->ixx_mult);
    atomicAdd(&log->iyy_mult_mean, p->iyy_mult);
    atomicAdd(&log->izz_mult_mean, p->izz_mult);
    float motor_scale_min = p->motor_thrust_scale[0];
    float motor_scale_max = p->motor_thrust_scale[0];
    float motor_scale_sum = 0.0f;
    #pragma unroll
    for (int m = 0; m < 4; m++) {
        motor_scale_min = fminf(motor_scale_min, p->motor_thrust_scale[m]);
        motor_scale_max = fmaxf(motor_scale_max, p->motor_thrust_scale[m]);
        motor_scale_sum += p->motor_thrust_scale[m];
    }
    atomicAdd(&log->k_thrust_mult_mean, p->k_thrust_mult * motor_scale_sum * 0.25f);
    if (dr_authority_gated_dev(cfg)) {
        atomicAdd(&log->k_thrust_mult_min, p->k_thrust_mult * motor_scale_min);
        atomicAdd(&log->k_thrust_mult_max, p->k_thrust_mult * motor_scale_max);
    } else {
        float k_range = dr_param_range_dev(cfg, cfg.dr_k_thrust);
        atomicAdd(&log->k_thrust_mult_min, 1.0f - k_range);
        atomicAdd(&log->k_thrust_mult_max, 1.0f + k_range);
    }
    atomicAdd(&log->linear_drag_mult_mean, p->linear_drag_mult);
    atomicAdd(&log->yaw_drag_mult_mean, p->yaw_drag_mult);
    atomicAdd(&log->motor_lag_mult_mean, p->motor_lag_mult);
    atomicAdd(&log->com_x_mean, p->com_x);
    atomicAdd(&log->com_y_mean, p->com_y);
    atomicAdd(&log->com_z_mean, p->com_z);
    if (cfg.task == DRONE_TASK_RACE) {
        for (int gate = 0; gate < DRONE_GATE_DEBUG_MAX; gate++) {
            atomicAdd(&log->gate_time[gate], s->gate_time[gate]);
            atomicAdd(&log->gate_target_in_fov[gate], s->gate_target_in_fov[gate]);
            atomicAdd(&log->gate_bearing_error[gate], s->gate_bearing_error[gate]);
            atomicAdd(&log->gate_distance_to_target[gate], s->gate_distance_to_target[gate]);
            atomicAdd(&log->gate_pass_count[gate], s->gate_pass_count[gate]);
            atomicAdd(&log->gate_collision_count[gate], s->gate_collision_count[gate]);
        }
    }
    atomicAdd(&log->n, 1.0f);
}

__global__ void drone_rng_init_kernel(curandStatePhilox4_32_10_t* rng, int n,
                                      unsigned long long seed) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    curand_init(seed, (unsigned long long)i, 0, &rng[i]);
}

__global__ void drone_reset_kernel(DroneCudaCtx cfg, float* observations, float* rewards,
                                   float* terminals) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= cfg.total_agents) return;
    curandStatePhilox4_32_10_t r = cfg.rng[i];
    DroneCudaState s;
    DroneCudaParams p;
    reset_one_dev(&s, &p, cfg, &r, i);
    cfg.states[i] = s;
    cfg.params[i] = p;
    cfg.rng[i] = r;
    rewards[i] = 0.0f;
    terminals[i] = 0.0f;
    compute_obs_dev(&s, &p, cfg, &r, observations + (size_t)i * DRONE_OBS_SIZE);
    cfg.rng[i] = r;
}

__global__ void drone_step_kernel(DroneCudaCtx cfg, const float* actions, float* observations,
                                  float* rewards, float* terminals, int start, int count) {
    int local = blockIdx.x * blockDim.x + threadIdx.x;
    if (local >= count) return;
    int i = start + local;

    curandStatePhilox4_32_10_t r = cfg.rng[i];
    DroneCudaState s = cfg.states[i];
    DroneCudaParams p = cfg.params[i];

    float raw_actions[4];
    #pragma unroll
    for (int k = 0; k < 4; k++) {
        raw_actions[k] = actions[(size_t)i * DRONE_NUM_ATNS + k];
    }
    apply_pal_probe_dev(cfg, &s, raw_actions);

    float action_delta_mean = 0.0f;
    if (s.has_prev_action) {
        #pragma unroll
        for (int k = 0; k < 4; k++) {
            action_delta_mean += fabsf(raw_actions[k] - s.prev_action[k]);
        }
        action_delta_mean *= 0.25f;
    }
    bool reset_action_boundary = cfg.reset_action_interval > 0
        && ((cfg.tick - 1 + cfg.reset_action_interval) % cfg.reset_action_interval) == 0;
    float reset_action_jump = reset_action_boundary ? action_delta_mean : 0.0f;

    s.prev_pos = s.pos;
    s.action_history_idx = (s.action_history_idx + 1) % (DRONE_MAX_ACTION_LATENCY_STEPS + 1);
    #pragma unroll
    for (int k = 0; k < 4; k++) {
        s.action_history[s.action_history_idx][k] = raw_actions[k];
    }
    int read_idx = s.action_history_idx - cfg.action_latency_steps;
    if (read_idx < 0) read_idx += DRONE_MAX_ACTION_LATENCY_STEPS + 1;
    float delayed_actions[4];
    #pragma unroll
    for (int k = 0; k < 4; k++) {
        delayed_actions[k] = s.action_history[read_idx][k];
    }

    move_drone_dev(&s, &p, delayed_actions);
    if (cfg.task == DRONE_TASK_RACE) {
        race_apply_turn_speed_limit_math_dev(&s, &p);
    }
    s.episode_length++;

    float curr_dist = norm3_dev(sub3_dev(s.target_pos, s.pos));
    float prev_dist = norm3_dev(sub3_dev(s.target_pos, s.prev_pos));
    bool oob = curr_dist > cfg.oob_radius;
    bool timeout = s.episode_length >= cfg.horizon;
    int ring_result = 0;
    bool lap_complete = false;
    float r_terminal = 0.0f;
    if (cfg.task == DRONE_TASK_RACE) {
        int current_gate_idx = race_clamped_gate_idx_dev(&s);
        ring_result = check_ring_dev(&s);
        if (ring_result == 1) {
            s.rings_passed += 1.0f;
            s.race_gate_bank += 1.0f;
            if (current_gate_idx >= 0 && current_gate_idx < DRONE_GATE_DEBUG_MAX) {
                s.gate_pass_count[current_gate_idx] += 1.0f;
            }
            if (s.buffer_size > 0) {
                int next_target = (current_gate_idx + 1) % s.buffer_size;
                race_isb_push_dev(cfg, i, &s, next_target,
                                  race_pass_margin_dev(&s, current_gate_idx));
            }
            lap_complete = cfg.race_segment_mode >= 1.0f
                || (s.buffer_size > 0 && s.buffer_idx == s.buffer_size - 1);
            r_terminal += 0.2f;
        } else if (ring_result == -1) {
            s.ring_collision += 1.0f;
            s.collisions += 1.0f;
            if (current_gate_idx >= 0 && current_gate_idx < DRONE_GATE_DEBUG_MAX) {
                s.gate_collision_count[current_gate_idx] += 1.0f;
            }
            r_terminal -= 2.0f;
        }
        if (oob) {
            float speed = norm3_dev(s.vel);
            r_terminal += -10.0f
                        - s.race_gate_bank
                        - 0.05f * speed * speed;
            s.race_gate_bank = 0.0f;
        } else if (lap_complete || timeout) {
            r_terminal += s.race_gate_bank;
            s.race_gate_bank = 0.0f;
        }
    }
    float omega = norm3_dev(s.omega);
    float omega_xy = sqrtf(s.omega.x * s.omega.x + s.omega.y * s.omega.y);
    float omega_z = s.omega.z;
    float r_omega_xy = -cfg.alpha_omega_xy * omega_xy;
    float r_omega_z = -cfg.alpha_omega_z * fabsf(omega_z)
                    - cfg.alpha_omega_z_sq * cfg.alpha_omega_z_mult * omega_z * omega_z;
    float curr = hover_potential_dev(&s, cfg);
    float r_dist = cfg.alpha_dist * (prev_dist - curr_dist);
    float r_hover = cfg.alpha_hover * curr;
    float r_shaping = cfg.alpha_shaping * (curr - s.prev_potential);
    float r_omega = r_omega_xy + r_omega_z;
    float r_action_delta = -cfg.alpha_action_delta * action_delta_mean
                         -cfg.alpha_reset_action_delta * reset_action_jump;
    float reward = r_dist + r_hover + r_shaping + r_omega + r_terminal + r_action_delta;
    s.prev_potential = curr;

    float h = check_hover_dev(&s, cfg);
    s.hover_score += h;
    s.hover_ema = (1.0f - 0.02f) * s.hover_ema + 0.02f * h;
    s.ema_dist = 0.99f * s.ema_dist + 0.01f * curr_dist;
    s.ema_vel = 0.99f * s.ema_vel + 0.01f * norm3_dev(s.vel);
    s.ema_omega = 0.99f * s.ema_omega + 0.01f * omega;
    s.ema_omega_x = 0.99f * s.ema_omega_x + 0.01f * fabsf(s.omega.x);
    s.ema_omega_y = 0.99f * s.ema_omega_y + 0.01f * fabsf(s.omega.y);
    s.ema_omega_z = 0.99f * s.ema_omega_z + 0.01f * fabsf(s.omega.z);
    record_retina_diagnostics_dev(&s, cfg);
    record_step_metrics_dev(&s, &p, raw_actions, r_dist, r_hover, r_shaping, r_omega,
                            r_omega_xy, r_omega_z, r_terminal,
                            action_delta_mean, reset_action_jump);
    s.episode_return += reward;

    if (cfg.task == DRONE_TASK_RACE && ring_result == 1 && !lap_complete) {
        s.buffer_idx = s.buffer_idx + 1;
        set_target_race_dev(&s);
        s.prev_pos = s.pos;
        s.prev_potential = hover_potential_dev(&s, cfg);
    }

    bool done = oob || timeout || lap_complete;
    rewards[i] = reward;
    terminals[i] = done ? 1.0f : 0.0f;

    if (done) {
        log_done_dev(cfg.log, &s, &p, cfg, oob, timeout, lap_complete);
        adr_record_done_dev(cfg, &s, timeout);
        reset_one_dev(&s, &p, cfg, &r, i);
    }

    compute_obs_dev(&s, &p, cfg, &r, observations + (size_t)i * DRONE_OBS_SIZE);
    cfg.states[i] = s;
    cfg.params[i] = p;
    cfg.rng[i] = r;
}

static DroneCudaCtx make_host_ctx(StaticVec* vec, Dict* vec_kwargs, Dict* env_kwargs) {
    DroneCudaCtx ctx;
    memset(&ctx, 0, sizeof(ctx));
    ctx.total_agents = vec->total_agents;
    ctx.horizon = DRONE_HORIZON;
    ctx.task = (int)dict_float(env_kwargs, "task", 1.0f);
    ctx.max_rings = (int)dict_float(env_kwargs, "max_rings", 10.0f);
    if (ctx.max_rings < 1) ctx.max_rings = 1;
    if (ctx.max_rings > DRONE_CUDA_MAX_RINGS) ctx.max_rings = DRONE_CUDA_MAX_RINGS;
    ctx.alpha_dist = dict_float(env_kwargs, "alpha_dist", 0.782192f);
    ctx.alpha_hover = dict_float(env_kwargs, "alpha_hover", 0.071445f);
    ctx.alpha_shaping = dict_float(env_kwargs, "alpha_shaping", 3.9754f);
    ctx.alpha_omega_xy = dict_float(env_kwargs, "alpha_omega_xy",
                                    dict_float(env_kwargs, "alpha_omega", 0.00135588f));
    ctx.alpha_omega_z = dict_float(env_kwargs, "alpha_omega_z",
                                   dict_float(env_kwargs, "alpha_omega", 0.00135588f));
    ctx.alpha_omega_z_sq = dict_float(env_kwargs, "alpha_omega_z_sq", 0.0025f);
    ctx.alpha_omega_z_mult = dict_float(env_kwargs, "alpha_omega_z_mult", 5.0f);
    ctx.alpha_action_delta = dict_float(env_kwargs, "alpha_action_delta", 0.0f);
    ctx.alpha_reset_action_delta = dict_float(env_kwargs, "alpha_reset_action_delta", 0.0f);
    ctx.reset_action_interval = (int)dict_float(env_kwargs, "reset_action_interval", 0.0f);
    ctx.hover_target_dist = dict_float(env_kwargs, "hover_target_dist", 5.0f);
    ctx.oob_radius = dict_float(env_kwargs, "oob_radius", 12.0f);
    ctx.hover_dist = dict_float(env_kwargs, "hover_dist", 0.1f);
    ctx.hover_omega = dict_float(env_kwargs, "hover_omega", 0.1f);
    ctx.hover_vel = dict_float(env_kwargs, "hover_vel", 0.1f);
    ctx.domain_randomization = dict_float(env_kwargs, "domain_randomization", 0.0f);
    ctx.dr_mass = dict_float(env_kwargs, "dr_mass", 0.0f);
    ctx.dr_inertia = dict_float(env_kwargs, "dr_inertia", 0.0f);
    ctx.dr_k_thrust = dict_float(env_kwargs, "dr_k_thrust", 0.0f);
    ctx.dr_linear_drag = dict_float(env_kwargs, "dr_linear_drag", 0.0f);
    ctx.dr_yaw_drag = dict_float(env_kwargs, "dr_yaw_drag", 0.0f);
    ctx.dr_motor_lag = dict_float(env_kwargs, "dr_motor_lag", 0.0f);
    ctx.dr_com_xy = dict_float(env_kwargs, "dr_com_xy", 0.0f);
    ctx.dr_com_z = dict_float(env_kwargs, "dr_com_z", 0.0f);
    ctx.dr_authority_gated = dict_float(env_kwargs, "dr_authority_gated", 0.0f);
    ctx.dr_usable_t2w_min = dict_float(env_kwargs, "dr_usable_t2w_min", 0.0f);
    ctx.dr_usable_t2w_max = dict_float(env_kwargs, "dr_usable_t2w_max", 0.0f);
    ctx.dr_mass_min = dict_float(env_kwargs, "dr_mass_min", 1.0f);
    ctx.dr_mass_max = dict_float(env_kwargs, "dr_mass_max", 1.0f);
    ctx.dr_inertia_min = dict_float(env_kwargs, "dr_inertia_min", 1.0f);
    ctx.dr_inertia_max = dict_float(env_kwargs, "dr_inertia_max", 1.0f);
    ctx.dr_motor_thrust_min = dict_float(env_kwargs, "dr_motor_thrust_min", 1.0f);
    ctx.dr_motor_thrust_max = dict_float(env_kwargs, "dr_motor_thrust_max", 1.0f);
    ctx.dr_motor_tau_min = dict_float(env_kwargs, "dr_motor_tau_min", BASE_K_MOT);
    ctx.dr_motor_tau_max = dict_float(env_kwargs, "dr_motor_tau_max", BASE_K_MOT);
    ctx.dr_yaw_torque_min = dict_float(env_kwargs, "dr_yaw_torque_min", 1.0f);
    ctx.dr_yaw_torque_max = dict_float(env_kwargs, "dr_yaw_torque_max", 1.0f);
    ctx.dr_linear_drag_min = dict_float(env_kwargs, "dr_linear_drag_min", 1.0f);
    ctx.dr_linear_drag_max = dict_float(env_kwargs, "dr_linear_drag_max", 1.0f);
    ctx.dr_angular_damping_min = dict_float(env_kwargs, "dr_angular_damping_min", 1.0f);
    ctx.dr_angular_damping_max = dict_float(env_kwargs, "dr_angular_damping_max", 1.0f);
    ctx.dr_profile_mix = dict_float(env_kwargs, "dr_profile_mix", 0.0f);
    ctx.adr_enabled = dict_float(env_kwargs, "adr_enabled", 0.0f);
    ctx.adr_mode = dict_float(env_kwargs, "adr_mode", (float)DRONE_ADR_MODE_AUTHORITY);
    ctx.adr_probe_prob = dict_float(env_kwargs, "adr_probe_prob", 0.02f);
    ctx.adr_success_threshold = dict_float(env_kwargs, "adr_success_threshold", 0.90f);
    ctx.adr_contract_threshold = dict_float(env_kwargs, "adr_contract_threshold", 0.50f);
    ctx.adr_step = dict_float(env_kwargs, "adr_step", 0.02f);
    ctx.adr_eval_episodes = dict_float(env_kwargs, "adr_eval_episodes", 64.0f);
    ctx.adr_init_usable_t2w_min = dict_float(env_kwargs, "adr_init_usable_t2w_min", 2.40f);
    ctx.adr_init_usable_t2w_max = dict_float(env_kwargs, "adr_init_usable_t2w_max", 3.40f);
    ctx.adr_init_mass_min = dict_float(env_kwargs, "adr_init_mass_min", 0.90f);
    ctx.adr_init_mass_max = dict_float(env_kwargs, "adr_init_mass_max", 1.10f);
    ctx.adr_init_inertia_min = dict_float(env_kwargs, "adr_init_inertia_min", 0.80f);
    ctx.adr_init_inertia_max = dict_float(env_kwargs, "adr_init_inertia_max", 1.30f);
    ctx.adr_init_motor_thrust_min = dict_float(env_kwargs, "adr_init_motor_thrust_min", 0.92f);
    ctx.adr_init_motor_thrust_max = dict_float(env_kwargs, "adr_init_motor_thrust_max", 1.08f);
    ctx.adr_init_motor_tau_min = dict_float(env_kwargs, "adr_init_motor_tau_min", 0.08f);
    ctx.adr_init_motor_tau_max = dict_float(env_kwargs, "adr_init_motor_tau_max", 0.18f);
    ctx.adr_init_com_xy = dict_float(env_kwargs, "adr_init_com_xy", 0.012f);
    ctx.pal_probe_prob = dict_float(env_kwargs, "pal_probe_prob", 0.0f);
    ctx.pal_probe_steps = (int)dict_float(env_kwargs, "pal_probe_steps", 0.0f);
    ctx.pal_probe_amp = dict_float(env_kwargs, "pal_probe_amp", 0.0f);
    ctx.action_scale = dict_float(env_kwargs, "action_scale", 1.0f);
    ctx.action_mode = (int)dict_float(env_kwargs, "action_mode", (float)M4D_ACTION_HOVER_TRIM);
    ctx.normalized_thrust_min = dict_float(env_kwargs, "normalized_thrust_min", 0.0f);
    ctx.normalized_thrust_max = dict_float(env_kwargs, "normalized_thrust_max", 1.0f);
    ctx.reset_pos_scale = dict_float(env_kwargs, "reset_pos_scale", 1.0f);
    ctx.reset_yaw_range = dict_float(env_kwargs, "reset_yaw_range", DRONE_PI);
    ctx.reset_vel_max = dict_float(env_kwargs, "reset_vel_max", 0.0f);
    ctx.sensor_noise = dict_float(env_kwargs, "sensor_noise", 0.0f);
    ctx.minimal_vision_enabled = dict_float(env_kwargs, "minimal_vision_enabled", 0.0f);
    ctx.minimal_vision_only = dict_float(env_kwargs, "minimal_vision_only", 0.0f);
    ctx.minimal_vision_mask_target = dict_float(env_kwargs, "minimal_vision_mask_target", 0.0f);
    ctx.minimal_vision_fov = dict_float(env_kwargs, "minimal_vision_fov", 2.0943951f);
    ctx.minimal_vision_vfov = dict_float(env_kwargs, "minimal_vision_vfov", 1.3962634f);
    ctx.minimal_vision_sigma = dict_float(env_kwargs, "minimal_vision_sigma", 0.45f);
    ctx.minimal_vision_depth_gain = dict_float(env_kwargs, "minimal_vision_depth_gain", 0.08f);
    ctx.minimal_vision_noise = dict_float(env_kwargs, "minimal_vision_noise", 0.0f);
    ctx.minimal_vision_distractors = dict_float(env_kwargs, "minimal_vision_distractors", 0.0f);
    ctx.minimal_vision_spawn_visible_target = dict_float(env_kwargs, "minimal_vision_spawn_visible_target", 0.0f);
    ctx.minimal_vision_gate_mask = dict_float(env_kwargs, "minimal_vision_gate_mask", 0.0f);
    ctx.race_track_mode = dict_float(env_kwargs, "race_track_mode", 0.0f);
    ctx.race_segment_mode = dict_float(env_kwargs, "race_segment_mode", 0.0f);
    ctx.race_isb_enabled = dict_float(env_kwargs, "race_isb_enabled", 0.0f);
    ctx.race_isb_prob = dict_float(env_kwargs, "race_isb_prob", 0.0f);
    ctx.race_isb_margin = dict_float(env_kwargs, "race_isb_margin", 0.8f);
    ctx.race_isb_pos_xy = dict_float(env_kwargs, "race_isb_pos_xy", 0.45f);
    ctx.race_isb_z = dict_float(env_kwargs, "race_isb_z", 0.25f);
    ctx.race_isb_angle = dict_float(env_kwargs, "race_isb_angle", 0.18f);
    ctx.race_isb_vel = dict_float(env_kwargs, "race_isb_vel", 0.60f);
    ctx.race_isb_omega = dict_float(env_kwargs, "race_isb_omega", 0.60f);
    ctx.race_hard_gate_idx = dict_float(env_kwargs, "race_hard_gate_idx", -1.0f);
    ctx.race_hard_gate_prob = dict_float(env_kwargs, "race_hard_gate_prob", 0.0f);
    ctx.race_course_yaw_delta = dict_float(env_kwargs, "race_course_yaw_delta", 0.0f);
    ctx.race_course_pitch_delta = dict_float(env_kwargs, "race_course_pitch_delta", 0.0f);
    ctx.race_course_pitch_limit = dict_float(env_kwargs, "race_course_pitch_limit", 0.0f);
    ctx.race_course_spacing_min = dict_float(env_kwargs, "race_course_spacing_min", 0.0f);
    ctx.race_course_spacing_max = dict_float(env_kwargs, "race_course_spacing_max", 0.0f);
    ctx.race_course_dz_max = dict_float(env_kwargs, "race_course_dz_max", 0.0f);
    ctx.race_reset_start_prob = dict_float(env_kwargs, "race_reset_start_prob", 0.0f);
    ctx.race_reset_t_min = dict_float(env_kwargs, "race_reset_t_min", 0.0f);
    ctx.race_reset_t_max = dict_float(env_kwargs, "race_reset_t_max", 0.0f);
    ctx.race_reset_lateral = dict_float(env_kwargs, "race_reset_lateral", 0.0f);
    ctx.race_reset_yaw_error_frac = dict_float(env_kwargs, "race_reset_yaw_error_frac", 0.0f);
    ctx.race_reset_speed_min = dict_float(env_kwargs, "race_reset_speed_min", 0.0f);
    ctx.race_reset_speed_max = dict_float(env_kwargs, "race_reset_speed_max", 0.0f);
    ctx.action_latency_steps = latency_steps_from_seconds(dict_float(env_kwargs, "action_latency", 0.0f));
    (void)vec_kwargs;
    return ctx;
}

static float clamp_host(float v, float lo, float hi) {
    if (hi < lo) {
        float tmp = lo;
        lo = hi;
        hi = tmp;
    }
    return fminf(fmaxf(v, lo), hi);
}

static void adr_range_host(float hard_lo, float hard_hi, float init_lo, float init_hi,
                           float fallback_lo, float fallback_hi, float gap,
                           float* out_lo, float* out_hi) {
    if (hard_hi <= hard_lo) {
        hard_lo = fallback_lo;
        hard_hi = fallback_hi;
    }
    init_lo = clamp_host(init_lo, hard_lo, hard_hi);
    init_hi = clamp_host(init_hi, hard_lo, hard_hi);
    if (init_hi < init_lo + gap) {
        init_lo = clamp_host(fallback_lo, hard_lo, hard_hi);
        init_hi = clamp_host(fallback_hi, hard_lo, hard_hi);
    }
    if (init_hi < init_lo + gap) {
        init_lo = hard_lo;
        init_hi = hard_hi;
    }
    *out_lo = init_lo;
    *out_hi = init_hi;
}

static DroneCudaAdrState make_host_adr_state(const DroneCudaCtx* ctx) {
    DroneCudaAdrState adr;
    memset(&adr, 0, sizeof(adr));
    adr.enabled = ctx->adr_enabled > 0.0f ? 1 : 0;
    adr.mode = (int)ctx->adr_mode;
    if (adr.mode != DRONE_ADR_MODE_LEGACY_PAPER) {
        adr.mode = DRONE_ADR_MODE_AUTHORITY;
    }
    adr_range_host(ctx->dr_usable_t2w_min, ctx->dr_usable_t2w_max,
                   ctx->adr_init_usable_t2w_min, ctx->adr_init_usable_t2w_max,
                   2.40f, 3.40f, 0.05f, &adr.usable_t2w_min, &adr.usable_t2w_max);
    adr_range_host(ctx->dr_mass_min, ctx->dr_mass_max,
                   ctx->adr_init_mass_min, ctx->adr_init_mass_max,
                   0.90f, 1.10f, 0.01f, &adr.mass_min, &adr.mass_max);
    adr_range_host(ctx->dr_inertia_min, ctx->dr_inertia_max,
                   ctx->adr_init_inertia_min, ctx->adr_init_inertia_max,
                   0.80f, 1.30f, 0.01f, &adr.inertia_min, &adr.inertia_max);
    adr_range_host(ctx->dr_motor_thrust_min, ctx->dr_motor_thrust_max,
                   ctx->adr_init_motor_thrust_min, ctx->adr_init_motor_thrust_max,
                   0.92f, 1.08f, 0.01f, &adr.motor_thrust_min, &adr.motor_thrust_max);
    adr_range_host(ctx->dr_motor_tau_min, ctx->dr_motor_tau_max,
                   ctx->adr_init_motor_tau_min, ctx->adr_init_motor_tau_max,
                   0.08f, 0.18f, 0.002f, &adr.motor_tau_min, &adr.motor_tau_max);
    adr.com_xy = clamp_host(ctx->adr_init_com_xy, 0.0f, fabsf(ctx->dr_com_xy));
    adr.legacy_mass_min = 1.0f;
    adr.legacy_mass_max = 1.0f;
    adr.legacy_inertia_min = 1.0f;
    adr.legacy_inertia_max = 1.0f;
    adr.legacy_k_thrust_min = 1.0f;
    adr.legacy_k_thrust_max = 1.0f;
    adr.legacy_linear_drag_min = 1.0f;
    adr.legacy_linear_drag_max = 1.0f;
    adr.legacy_yaw_drag_min = 1.0f;
    adr.legacy_yaw_drag_max = 1.0f;
    adr.legacy_motor_lag_min = 1.0f;
    adr.legacy_motor_lag_max = 1.0f;
    adr.legacy_com_xy = 0.0f;
    adr.legacy_com_z = 0.0f;
    return adr;
}

extern "C" void cuda_env_init(StaticVec* vec, Dict* vec_kwargs, Dict* env_kwargs) {
    DroneCudaCtx* ctx = (DroneCudaCtx*)calloc(1, sizeof(DroneCudaCtx));
    *ctx = make_host_ctx(vec, vec_kwargs, env_kwargs);
    if (ctx->task != DRONE_TASK_HOVER && ctx->task != DRONE_TASK_RACE) {
        fprintf(stderr, "drone CUDA env implements HOVER/RACE only; got task=%d\n", ctx->task);
    }
    CUDA_ENV_CHECK(cudaMalloc((void**)&ctx->states,
                              (size_t)ctx->total_agents * sizeof(DroneCudaState)));
    CUDA_ENV_CHECK(cudaMalloc((void**)&ctx->params,
                              (size_t)ctx->total_agents * sizeof(DroneCudaParams)));
    CUDA_ENV_CHECK(cudaMalloc((void**)&ctx->rng,
                              (size_t)ctx->total_agents * sizeof(curandStatePhilox4_32_10_t)));
    CUDA_ENV_CHECK(cudaMalloc((void**)&ctx->log, sizeof(DroneCudaLog)));
    CUDA_ENV_CHECK(cudaMemset(ctx->log, 0, sizeof(DroneCudaLog)));
    size_t isb_gates = (size_t)ctx->total_agents * DRONE_GATE_DEBUG_MAX;
    size_t isb_states = isb_gates * DRONE_ISB_CAPACITY;
    CUDA_ENV_CHECK(cudaMalloc((void**)&ctx->isb_states,
                              isb_states * sizeof(DroneCudaIsbState)));
    CUDA_ENV_CHECK(cudaMalloc((void**)&ctx->isb_counts, isb_gates * sizeof(int)));
    CUDA_ENV_CHECK(cudaMalloc((void**)&ctx->isb_cursors, isb_gates * sizeof(int)));
    CUDA_ENV_CHECK(cudaMemset(ctx->isb_states, 0, isb_states * sizeof(DroneCudaIsbState)));
    CUDA_ENV_CHECK(cudaMemset(ctx->isb_counts, 0, isb_gates * sizeof(int)));
    CUDA_ENV_CHECK(cudaMemset(ctx->isb_cursors, 0, isb_gates * sizeof(int)));
    if (ctx->adr_enabled > 0.0f) {
        DroneCudaAdrState host_adr = make_host_adr_state(ctx);
        CUDA_ENV_CHECK(cudaMalloc((void**)&ctx->adr, sizeof(DroneCudaAdrState)));
        CUDA_ENV_CHECK(cudaMemcpy(ctx->adr, &host_adr, sizeof(host_adr), cudaMemcpyHostToDevice));
    }

    int block = 128;
    int grid = (ctx->total_agents + block - 1) / block;
    unsigned int seed = dict_uint(vec_kwargs, "seed", 1u);
    drone_rng_init_kernel<<<grid, block>>>(ctx->rng, ctx->total_agents, seed);
    CUDA_ENV_CHECK(cudaGetLastError());
    vec->cuda_env = ctx;
    cuda_env_reset(vec);
}

extern "C" void cuda_env_reset(StaticVec* vec) {
    DroneCudaCtx* ctx = (DroneCudaCtx*)vec->cuda_env;
    if (ctx == NULL) return;
    ctx->tick = 0;
    CUDA_ENV_CHECK(cudaMemset(ctx->log, 0, sizeof(DroneCudaLog)));
    size_t isb_gates = (size_t)ctx->total_agents * DRONE_GATE_DEBUG_MAX;
    if (ctx->isb_counts != NULL) {
        CUDA_ENV_CHECK(cudaMemset(ctx->isb_counts, 0, isb_gates * sizeof(int)));
    }
    if (ctx->isb_cursors != NULL) {
        CUDA_ENV_CHECK(cudaMemset(ctx->isb_cursors, 0, isb_gates * sizeof(int)));
    }
    int block = 128;
    int grid = (ctx->total_agents + block - 1) / block;
    drone_reset_kernel<<<grid, block>>>(*ctx, (float*)vec->gpu_observations,
                                        vec->gpu_rewards, vec->gpu_terminals);
    CUDA_ENV_CHECK(cudaGetLastError());
}

extern "C" void cuda_env_step_buffer(StaticVec* vec, int agent_start, int count,
                                     cudaStream_t stream) {
    DroneCudaCtx* ctx = (DroneCudaCtx*)vec->cuda_env;
    if (ctx == NULL || count <= 0) return;
    if (agent_start == 0) ctx->tick = (ctx->tick + 1) % DRONE_HORIZON;
    int block = 128;
    int grid = (count + block - 1) / block;
    drone_step_kernel<<<grid, block, 0, stream>>>(*ctx, vec->gpu_actions,
                                                  (float*)vec->gpu_observations,
                                                  vec->gpu_rewards, vec->gpu_terminals,
                                                  agent_start, count);
    CUDA_ENV_CHECK(cudaGetLastError());
}

extern "C" void cuda_env_step_all(StaticVec* vec, cudaStream_t stream) {
    cuda_env_step_buffer(vec, 0, vec->total_agents, stream);
}

static inline void set_gate_debug_log_cuda(Dict* out, const DroneCudaLog* log, int gate,
                                           const char* time_key, const char* fov_key,
                                           const char* bearing_key, const char* dist_key,
                                           const char* pass_key, const char* collision_key,
                                           const char* timeout_key, const char* oob_key,
                                           float total_gate_time, float episode_inv) {
    float gate_time = log->gate_time[gate];
    float gate_inv = gate_time > 1e-6f ? 1.0f / gate_time : 0.0f;
    float total_inv = total_gate_time > 1e-6f ? 1.0f / total_gate_time : 0.0f;
    dict_set(out, time_key, gate_time * total_inv);
    dict_set(out, fov_key, log->gate_target_in_fov[gate] * gate_inv);
    dict_set(out, bearing_key, log->gate_bearing_error[gate] * gate_inv);
    dict_set(out, dist_key, log->gate_distance_to_target[gate] * gate_inv);
    dict_set(out, pass_key, log->gate_pass_count[gate] * episode_inv);
    dict_set(out, collision_key, log->gate_collision_count[gate] * episode_inv);
    dict_set(out, timeout_key, log->gate_timeout_count[gate] * episode_inv);
    dict_set(out, oob_key, log->gate_oob_count[gate] * episode_inv);
}

#define SET_GATE_DEBUG_LOG_CUDA(G) \
    set_gate_debug_log_cuda(out, &h, G, \
        "g" #G "_time_frac", "g" #G "_fov", "g" #G "_bearing", "g" #G "_dist", \
        "g" #G "_pass", "g" #G "_collision", "g" #G "_timeout", "g" #G "_oob", \
        total_gate_time, inv)

extern "C" void cuda_env_log(StaticVec* vec, Dict* out) {
    DroneCudaCtx* ctx = (DroneCudaCtx*)vec->cuda_env;
    if (ctx == NULL) return;
    DroneCudaLog h;
    CUDA_ENV_CHECK(cudaMemcpy(&h, ctx->log, sizeof(h), cudaMemcpyDeviceToHost));
    if (h.n <= 0.0f) return;
    float inv = 1.0f / h.n;
    dict_set(out, "perf", h.perf * inv);
    dict_set(out, "score", h.score * inv);
    dict_set(out, "rings_passed", h.rings_passed * inv);
    dict_set(out, "ring_collisions", h.ring_collision * inv);
    dict_set(out, "collisions", h.collisions * inv);
    dict_set(out, "oob", h.oob * inv);
    dict_set(out, "timeout", h.timeout * inv);
    dict_set(out, "lap_complete", h.lap_complete * inv);
    dict_set(out, "episode_return", h.episode_return * inv);
    dict_set(out, "episode_length", h.episode_length * inv);
    dict_set(out, "ema_dist", h.ema_dist * inv);
    dict_set(out, "ema_vel", h.ema_vel * inv);
    dict_set(out, "ema_omega", h.ema_omega * inv);
    dict_set(out, "action_saturation_frac", h.action_saturation_frac * inv);
    dict_set(out, "mean_abs_delta_action", h.mean_abs_delta_action * inv);
    dict_set(out, "target_in_fov_frac", h.target_in_fov_frac * inv);
    dict_set(out, "retina_energy", h.retina_energy * inv);
    dict_set(out, "bearing_error_to_target", h.bearing_error_to_target * inv);
    dict_set(out, "distance_to_target", h.distance_to_target * inv);
    {
        float oob_diag_inv = h.oob_diag_count > 1e-6f ? 1.0f / h.oob_diag_count : 0.0f;
        dict_set(out, "oob_diag_count", h.oob_diag_count * inv);
        dict_set(out, "gate_index_at_oob", h.gate_index_at_oob * oob_diag_inv);
        dict_set(out, "distance_from_track_centerline", h.distance_from_track_centerline * oob_diag_inv);
        float total_gate_time = 0.0f;
        for (int gate = 0; gate < DRONE_GATE_DEBUG_MAX; gate++) {
            total_gate_time += h.gate_time[gate];
        }
        SET_GATE_DEBUG_LOG_CUDA(0);
        SET_GATE_DEBUG_LOG_CUDA(1);
        SET_GATE_DEBUG_LOG_CUDA(2);
        SET_GATE_DEBUG_LOG_CUDA(3);
        SET_GATE_DEBUG_LOG_CUDA(4);
        SET_GATE_DEBUG_LOG_CUDA(5);
        SET_GATE_DEBUG_LOG_CUDA(6);
        SET_GATE_DEBUG_LOG_CUDA(7);
    }
    dict_set(out, "n", h.n);
    CUDA_ENV_CHECK(cudaMemset(ctx->log, 0, sizeof(DroneCudaLog)));
    return;

    dict_set(out, "perf", h.perf * inv);
    dict_set(out, "score", h.score * inv);
    dict_set(out, "rings_passed", h.rings_passed * inv);
    dict_set(out, "ring_collisions", h.ring_collision * inv);
    dict_set(out, "collisions", h.collisions * inv);
    dict_set(out, "oob", h.oob * inv);
    dict_set(out, "timeout", h.timeout * inv);
    dict_set(out, "lap_complete", h.lap_complete * inv);
    dict_set(out, "episode_return", h.episode_return * inv);
    dict_set(out, "episode_length", h.episode_length * inv);
    dict_set(out, "ema_dist", h.ema_dist * inv);
    dict_set(out, "ema_vel", h.ema_vel * inv);
    dict_set(out, "ema_omega", h.ema_omega * inv);
    dict_set(out, "ema_omega_x", h.ema_omega_x * inv);
    dict_set(out, "ema_omega_y", h.ema_omega_y * inv);
    dict_set(out, "ema_omega_z", h.ema_omega_z * inv);
    dict_set(out, "mean_abs_action", h.mean_abs_action * inv);
    dict_set(out, "mean_abs_action_clipped", h.mean_abs_action_clipped * inv);
    dict_set(out, "max_abs_action", h.max_abs_action * inv);
    dict_set(out, "action_saturation_frac", h.action_saturation_frac * inv);
    dict_set(out, "mean_abs_delta_action", h.mean_abs_delta_action * inv);
    dict_set(out, "reset_action_jump_mean", h.reset_action_jump_mean * inv);
    dict_set(out, "motor_clip_low_frac", h.motor_clip_low_frac * inv);
    dict_set(out, "motor_clip_high_frac", h.motor_clip_high_frac * inv);
    dict_set(out, "hover_trim_rpm_mean", h.hover_trim_rpm_mean * inv);
    dict_set(out, "hover_trim_rpm_max", h.hover_trim_rpm_max * inv);
    dict_set(out, "hover_trim_rpm_frac_of_max", h.hover_trim_rpm_frac_of_max * inv);
    dict_set(out, "mean_rpm_FL", h.mean_rpm_FL * inv);
    dict_set(out, "mean_rpm_FR", h.mean_rpm_FR * inv);
    dict_set(out, "mean_rpm_RL", h.mean_rpm_RL * inv);
    dict_set(out, "mean_rpm_RR", h.mean_rpm_RR * inv);
    dict_set(out, "target_in_fov_frac", h.target_in_fov_frac * inv);
    dict_set(out, "retina_rgb_mean", h.retina_rgb_mean * inv);
    dict_set(out, "retina_rgb_std", h.retina_rgb_std * inv);
    dict_set(out, "retina_energy", h.retina_energy * inv);
    dict_set(out, "retina_left_center_right_argmax", h.retina_left_center_right_argmax * inv);
    dict_set(out, "retina_argmax_left_frac", h.retina_argmax_left_frac * inv);
    dict_set(out, "retina_argmax_center_frac", h.retina_argmax_center_frac * inv);
    dict_set(out, "retina_argmax_right_frac", h.retina_argmax_right_frac * inv);
    dict_set(out, "bearing_error_to_target", h.bearing_error_to_target * inv);
    dict_set(out, "distance_to_target", h.distance_to_target * inv);
    dict_set(out, "retina_signal_vs_distance", h.retina_signal_vs_distance * inv);
    dict_set(out, "r_dist", h.r_dist * inv);
    dict_set(out, "r_hover", h.r_hover * inv);
    dict_set(out, "r_shaping", h.r_shaping * inv);
    dict_set(out, "r_omega", h.r_omega * inv);
    dict_set(out, "r_omega_xy", h.r_omega_xy * inv);
    dict_set(out, "r_omega_z", h.r_omega_z * inv);
    dict_set(out, "r_terminal", h.r_terminal * inv);
    dict_set(out, "mass_mult_mean", h.mass_mult_mean * inv);
    dict_set(out, "ixx_mult_mean", h.ixx_mult_mean * inv);
    dict_set(out, "iyy_mult_mean", h.iyy_mult_mean * inv);
    dict_set(out, "izz_mult_mean", h.izz_mult_mean * inv);
    dict_set(out, "k_thrust_mult_mean", h.k_thrust_mult_mean * inv);
    dict_set(out, "k_thrust_mult_min", h.k_thrust_mult_min * inv);
    dict_set(out, "k_thrust_mult_max", h.k_thrust_mult_max * inv);
    dict_set(out, "linear_drag_mult_mean", h.linear_drag_mult_mean * inv);
    dict_set(out, "yaw_drag_mult_mean", h.yaw_drag_mult_mean * inv);
    dict_set(out, "motor_lag_mult_mean", h.motor_lag_mult_mean * inv);
    dict_set(out, "com_x_mean", h.com_x_mean * inv);
    dict_set(out, "com_y_mean", h.com_y_mean * inv);
    dict_set(out, "com_z_mean", h.com_z_mean * inv);
    float oob_diag_inv = h.oob_diag_count > 1e-6f ? 1.0f / h.oob_diag_count : 0.0f;
    dict_set(out, "oob_diag_count", h.oob_diag_count * inv);
    dict_set(out, "gate_index_at_oob", h.gate_index_at_oob * oob_diag_inv);
    dict_set(out, "position_norm_at_oob", h.position_norm_at_oob * oob_diag_inv);
    dict_set(out, "target_gate_position_norm", h.target_gate_position_norm * oob_diag_inv);
    dict_set(out, "next_gate_position_norm", h.next_gate_position_norm * oob_diag_inv);
    dict_set(out, "distance_from_track_centerline", h.distance_from_track_centerline * oob_diag_inv);
    float total_gate_time = 0.0f;
    for (int gate = 0; gate < DRONE_GATE_DEBUG_MAX; gate++) {
        total_gate_time += h.gate_time[gate];
    }
    SET_GATE_DEBUG_LOG_CUDA(0);
    SET_GATE_DEBUG_LOG_CUDA(1);
    SET_GATE_DEBUG_LOG_CUDA(2);
    SET_GATE_DEBUG_LOG_CUDA(3);
    SET_GATE_DEBUG_LOG_CUDA(4);
    SET_GATE_DEBUG_LOG_CUDA(5);
    SET_GATE_DEBUG_LOG_CUDA(6);
    SET_GATE_DEBUG_LOG_CUDA(7);
    if (ctx->adr != NULL) {
        DroneCudaAdrState adr;
        CUDA_ENV_CHECK(cudaMemcpy(&adr, ctx->adr, sizeof(adr), cudaMemcpyDeviceToHost));
        unsigned int updates = 0u;
        unsigned int pending = 0u;
        for (int i = 0; i < DRONE_ADR_EDGE_COUNT; i++) {
            updates += adr.updates[i];
            pending += adr.counts[i];
        }
        for (int i = 0; i < DRONE_ADR_LEGACY_EDGE_COUNT; i++) {
            updates += adr.legacy_updates[i];
            pending += adr.legacy_counts[i];
        }
        dict_set(out, "adr_enabled", (float)adr.enabled);
        dict_set(out, "adr_mode", (float)adr.mode);
        dict_set(out, "adr_updates", (float)updates);
        dict_set(out, "adr_pending_probes", (float)pending);
        dict_set(out, "adr_usable_t2w_min", adr.usable_t2w_min);
        dict_set(out, "adr_usable_t2w_max", adr.usable_t2w_max);
        dict_set(out, "adr_mass_min", adr.mass_min);
        dict_set(out, "adr_mass_max", adr.mass_max);
        dict_set(out, "adr_inertia_min", adr.inertia_min);
        dict_set(out, "adr_inertia_max", adr.inertia_max);
        dict_set(out, "adr_motor_thrust_min", adr.motor_thrust_min);
        dict_set(out, "adr_motor_thrust_max", adr.motor_thrust_max);
        dict_set(out, "adr_motor_tau_min", adr.motor_tau_min);
        dict_set(out, "adr_motor_tau_max", adr.motor_tau_max);
        dict_set(out, "adr_com_xy", adr.com_xy);
        dict_set(out, "adr_legacy_mass_min", adr.legacy_mass_min);
        dict_set(out, "adr_legacy_mass_max", adr.legacy_mass_max);
        dict_set(out, "adr_legacy_inertia_min", adr.legacy_inertia_min);
        dict_set(out, "adr_legacy_inertia_max", adr.legacy_inertia_max);
        dict_set(out, "adr_legacy_k_thrust_min", adr.legacy_k_thrust_min);
        dict_set(out, "adr_legacy_k_thrust_max", adr.legacy_k_thrust_max);
        dict_set(out, "adr_legacy_linear_drag_min", adr.legacy_linear_drag_min);
        dict_set(out, "adr_legacy_linear_drag_max", adr.legacy_linear_drag_max);
        dict_set(out, "adr_legacy_yaw_drag_min", adr.legacy_yaw_drag_min);
        dict_set(out, "adr_legacy_yaw_drag_max", adr.legacy_yaw_drag_max);
        dict_set(out, "adr_legacy_motor_lag_min", adr.legacy_motor_lag_min);
        dict_set(out, "adr_legacy_motor_lag_max", adr.legacy_motor_lag_max);
        dict_set(out, "adr_legacy_com_xy", adr.legacy_com_xy);
        dict_set(out, "adr_legacy_com_z", adr.legacy_com_z);
    }
    dict_set(out, "n", h.n);
    CUDA_ENV_CHECK(cudaMemset(ctx->log, 0, sizeof(DroneCudaLog)));
}

static void hover_trim_thrusts_host(const DroneCudaParams* p, float out[4]) {
    float a[4][5] = {
        {1.0f, 1.0f, 1.0f, 1.0f, p->mass * p->gravity},
        {p->motor_y[0], p->motor_y[1], p->motor_y[2], p->motor_y[3], 0.0f},
        {-p->motor_x[0], -p->motor_x[1], -p->motor_x[2], -p->motor_x[3], 0.0f},
        {p->k_drag * p->yaw_torque_scale[0] * p->yaw_sign[0],
         p->k_drag * p->yaw_torque_scale[1] * p->yaw_sign[1],
         p->k_drag * p->yaw_torque_scale[2] * p->yaw_sign[2],
         p->k_drag * p->yaw_torque_scale[3] * p->yaw_sign[3], 0.0f},
    };

    bool ok = true;
    for (int col = 0; col < 4; col++) {
        int pivot = col;
        float best = fabsf(a[col][col]);
        for (int row = col + 1; row < 4; row++) {
            float candidate = fabsf(a[row][col]);
            if (candidate > best) {
                best = candidate;
                pivot = row;
            }
        }
        if (best < 1e-8f) {
            ok = false;
            break;
        }
        if (pivot != col) {
            for (int k = col; k < 5; k++) {
                float tmp = a[col][k];
                a[col][k] = a[pivot][k];
                a[pivot][k] = tmp;
            }
        }
        float inv = 1.0f / a[col][col];
        for (int k = col; k < 5; k++) a[col][k] *= inv;
        for (int row = 0; row < 4; row++) {
            if (row == col) continue;
            float f = a[row][col];
            for (int k = col; k < 5; k++) a[row][k] -= f * a[col][k];
        }
    }

    if (!ok) {
        float fallback = 0.25f * p->mass * p->gravity;
        for (int i = 0; i < 4; i++) out[i] = fallback;
        return;
    }

    for (int i = 0; i < 4; i++) {
        float max_t = p->k_thrust * p->motor_thrust_scale[i] * p->max_rpm * p->max_rpm;
        out[i] = fminf(fmaxf(a[i][4], 0.0f), max_t);
    }
}

extern "C" int drone_debug_cuda_state(StaticVec* vec, int agent_idx, DroneDebugState* out) {
    DroneCudaCtx* ctx = (DroneCudaCtx*)vec->cuda_env;
    if (ctx == NULL || out == NULL || agent_idx < 0 || agent_idx >= ctx->total_agents) {
        return 0;
    }

    DroneCudaState s;
    DroneCudaParams p;
    cudaError_t err = cudaMemcpy(&s, ctx->states + agent_idx, sizeof(s), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) return 0;
    err = cudaMemcpy(&p, ctx->params + agent_idx, sizeof(p), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) return 0;

    memset(out, 0, sizeof(*out));
    out->pos[0] = s.pos.x;
    out->pos[1] = s.pos.y;
    out->pos[2] = s.pos.z;
    out->vel[0] = s.vel.x;
    out->vel[1] = s.vel.y;
    out->vel[2] = s.vel.z;
    out->quat[0] = s.quat.x;
    out->quat[1] = s.quat.y;
    out->quat[2] = s.quat.z;
    out->quat[3] = s.quat.w;
    out->omega[0] = s.omega.x;
    out->omega[1] = s.omega.y;
    out->omega[2] = s.omega.z;
    for (int i = 0; i < 4; i++) out->rpms[i] = s.rpms[i];
    out->target_pos[0] = s.target_pos.x;
    out->target_pos[1] = s.target_pos.y;
    out->target_pos[2] = s.target_pos.z;
    out->target_normal[0] = s.target_normal.x;
    out->target_normal[1] = s.target_normal.y;
    out->target_normal[2] = s.target_normal.z;
    out->prev_pos[0] = s.prev_pos.x;
    out->prev_pos[1] = s.prev_pos.y;
    out->prev_pos[2] = s.prev_pos.z;
    out->prev_potential = s.prev_potential;
    out->episode_return = s.episode_return;
    out->episode_length = s.episode_length;
    out->mass = p.mass;
    out->ixx = p.ixx;
    out->iyy = p.iyy;
    out->izz = p.izz;
    out->k_thrust = p.k_thrust;
    out->k_drag = p.k_drag;
    out->b_drag = p.b_drag;
    out->k_mot = p.k_mot;
    out->action_scale = p.action_scale;
    for (int i = 0; i < 4; i++) {
        out->motor_x[i] = p.motor_x[i];
        out->motor_y[i] = p.motor_y[i];
        out->yaw_sign[i] = p.yaw_sign[i];
    }
    hover_trim_thrusts_host(&p, out->hover_trim);
    return 1;
}

extern "C" void cuda_env_close(StaticVec* vec) {
    DroneCudaCtx* ctx = (DroneCudaCtx*)vec->cuda_env;
    if (ctx == NULL) return;
    CUDA_ENV_CHECK(cudaFree(ctx->states));
    CUDA_ENV_CHECK(cudaFree(ctx->params));
    CUDA_ENV_CHECK(cudaFree(ctx->rng));
    CUDA_ENV_CHECK(cudaFree(ctx->log));
    if (ctx->adr != NULL) CUDA_ENV_CHECK(cudaFree(ctx->adr));
    if (ctx->isb_states != NULL) CUDA_ENV_CHECK(cudaFree(ctx->isb_states));
    if (ctx->isb_counts != NULL) CUDA_ENV_CHECK(cudaFree(ctx->isb_counts));
    if (ctx->isb_cursors != NULL) CUDA_ENV_CHECK(cudaFree(ctx->isb_cursors));
    free(ctx);
    vec->cuda_env = NULL;
}
