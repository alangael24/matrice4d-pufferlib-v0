#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cuda_bf16.h>

#ifdef PRECISION_FLOAT
typedef float precision_t;
#else
typedef __nv_bfloat16 precision_t;
#endif

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "vecenv.h"

#define DRONE_OBS_SIZE 23
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

struct DroneCudaParams {
    float mass, ixx, iyy, izz;
    float motor_x[4], motor_y[4], yaw_sign[4];
    float k_thrust, k_ang_damp, k_drag, b_drag, gravity;
    float max_rpm, max_vel, max_omega, k_mot, action_scale;
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
    float motor_clip_low_count;
    float motor_clip_high_count;
    float rpm_sum[4];
    float instrumentation_steps;

    float r_dist_sum;
    float r_hover_sum;
    float r_shaping_sum;
    float r_omega_sum;
    float r_omega_xy_sum;
    float r_omega_z_sum;
    float r_terminal_sum;

    float action_history[DRONE_MAX_ACTION_LATENCY_STEPS + 1][4];
    int action_history_idx;
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
    float motor_clip_low_frac;
    float motor_clip_high_frac;
    float hover_trim_rpm_mean;
    float hover_trim_rpm_max;
    float hover_trim_rpm_frac_of_max;
    float mean_rpm_FL;
    float mean_rpm_FR;
    float mean_rpm_RL;
    float mean_rpm_RR;
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
    float n;
};

struct DroneCudaCtx {
    int total_agents;
    int horizon;
    int task;
    int action_latency_steps;

    float alpha_dist, alpha_hover, alpha_shaping;
    float alpha_omega_xy, alpha_omega_z, alpha_omega_z_sq, alpha_omega_z_mult;
    float hover_target_dist, oob_radius, hover_dist, hover_omega, hover_vel;
    float domain_randomization;
    float dr_mass, dr_inertia, dr_k_thrust, dr_linear_drag, dr_yaw_drag, dr_motor_lag;
    float dr_com_xy, dr_com_z;
    float action_scale;
    float reset_pos_scale, reset_yaw_range, reset_vel_max;
    float sensor_noise;

    DroneCudaState* states;
    DroneCudaParams* params;
    curandStatePhilox4_32_10_t* rng;
    DroneCudaLog* log;
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

__device__ float max_motor_thrust_dev(const DroneCudaParams* p) {
    return p->k_thrust * p->max_rpm * p->max_rpm;
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

    float max_t = max_motor_thrust_dev(p);
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        out[i] = clampf_dev(a[i][4], 0.0f, max_t);
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

__device__ __forceinline__ float thrust_to_rpm_dev(const DroneCudaParams* p, float thrust) {
    thrust = clampf_dev(thrust, 0.0f, max_motor_thrust_dev(p));
    return sqrtf(thrust / p->k_thrust);
}

__device__ void init_params_dev(DroneCudaParams* p, const DroneCudaCtx& cfg,
                                curandStatePhilox4_32_10_t* rng) {
    float mass_mult = dr_sample_mult_dev(rng, dr_param_range_dev(cfg, cfg.dr_mass));
    float ixx_mult = dr_sample_mult_dev(rng, dr_param_range_dev(cfg, cfg.dr_inertia));
    float iyy_mult = dr_sample_mult_dev(rng, dr_param_range_dev(cfg, cfg.dr_inertia));
    float izz_mult = dr_sample_mult_dev(rng, dr_param_range_dev(cfg, cfg.dr_inertia));
    float k_thrust_mult = dr_sample_mult_dev(rng, dr_param_range_dev(cfg, cfg.dr_k_thrust));
    float linear_drag_mult = dr_sample_mult_dev(rng, dr_param_range_dev(cfg, cfg.dr_linear_drag));
    float yaw_drag_mult = dr_sample_mult_dev(rng, dr_param_range_dev(cfg, cfg.dr_yaw_drag));
    float motor_lag_mult = dr_sample_mult_dev(rng, dr_param_range_dev(cfg, cfg.dr_motor_lag));

    float com_xy = cfg.domain_randomization > 0.0f ? fabsf(cfg.dr_com_xy) : 0.0f;
    float com_z_range = cfg.domain_randomization > 0.0f ? fabsf(cfg.dr_com_z) : 0.0f;
    float com_x = rndf_dev(-com_xy, com_xy, rng);
    float com_y = rndf_dev(-com_xy, com_xy, rng);
    float com_z = rndf_dev(-com_z_range, com_z_range, rng);

    p->mass = BASE_MASS * mass_mult;
    p->ixx = BASE_IXX * ixx_mult;
    p->iyy = BASE_IYY * iyy_mult;
    p->izz = BASE_IZZ * izz_mult;
    p->k_thrust = BASE_K_THRUST * k_thrust_mult;
    p->k_ang_damp = BASE_K_ANG_DAMP;
    p->k_drag = BASE_K_DRAG * yaw_drag_mult;
    p->b_drag = BASE_B_DRAG * linear_drag_mult;
    p->gravity = BASE_GRAVITY;
    p->max_rpm = BASE_MAX_RPM;
    p->max_vel = BASE_MAX_VEL;
    p->max_omega = BASE_MAX_OMEGA;
    p->k_mot = BASE_K_MOT * motor_lag_mult;
    p->action_scale = cfg.action_scale;
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
    float max_thrust = max_motor_thrust_dev(p);
    float target_rpms[4];
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        float action = clampf_dev(actions[i] * p->action_scale, -1.0f, 1.0f);
        float target_thrust = action >= 0.0f
            ? trim[i] + action * (max_thrust - trim[i])
            : trim[i] + action * trim[i];
        target_rpms[i] = thrust_to_rpm_dev(p, target_thrust);
        d->rpm_dot[i] = (1.0f / p->k_mot) * (target_rpms[i] - s->rpms[i]);
    }

    float T[4];
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        float rpm = fmaxf(s->rpms[i], 0.0f);
        T[i] = p->k_thrust * rpm * rpm;
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
        tau_prop.z += p->k_drag * p->yaw_sign[i] * T[i];
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

__device__ void reset_one_dev(DroneCudaState* s, DroneCudaParams* p, const DroneCudaCtx& cfg,
                              curandStatePhilox4_32_10_t* rng) {
    DroneCudaState zero = {};
    *s = zero;
    init_params_dev(p, cfg, rng);

    float trim[4];
    hover_trim_thrusts_dev(p, trim);
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        s->rpms[i] = thrust_to_rpm_dev(p, trim[i]);
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
    set_target_hover_dev(s, cfg, rng);
    s->prev_pos = s->pos;
    s->prev_potential = hover_potential_dev(s, cfg);
}

__device__ void compute_obs_dev(const DroneCudaState* s, const DroneCudaParams* p,
                                const DroneCudaCtx& cfg, curandStatePhilox4_32_10_t* rng,
                                float* obs) {
    int idx = 0;
    float4 q = s->quat;
    float4 q_inv = quat_inverse_dev(q);
    float3 linear_vel_body = quat_rotate_dev(q_inv, s->vel);
    float3 to_target_world = sub3_dev(s->target_pos, s->pos);
    float3 to_target = quat_rotate_dev(q_inv, to_target_world);
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

    if (cfg.sensor_noise > 0.0f) {
        float noise = fminf(fabsf(cfg.sensor_noise), 1.0f);
        #pragma unroll
        for (int i = 0; i < DRONE_OBS_SIZE; i++) {
            obs[i] = clampf_dev(obs[i] + rndf_dev(-noise, noise, rng), -2.0f, 2.0f);
        }
    }
}

__device__ void record_step_metrics_dev(DroneCudaState* s, const DroneCudaParams* p,
                                        const float raw_actions[4], float r_dist,
                                        float r_hover, float r_shaping, float r_omega,
                                        float r_omega_xy, float r_omega_z) {
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
        float motor_action = clampf_dev(env_clipped * p->action_scale, -1.0f, 1.0f);
        if (motor_action <= -0.99f) motor_clip_low_count += 1.0f;
        if (motor_action >= 0.99f) motor_clip_high_count += 1.0f;
        s->rpm_sum[i] += s->rpms[i];
    }
    s->action_abs_sum += action_abs_sum / 4.0f;
    s->action_clipped_abs_sum += action_clipped_abs_sum / 4.0f;
    s->action_max_abs = fmaxf(s->action_max_abs, action_max_abs);
    s->action_saturation_count += action_saturation_count / 4.0f;
    s->motor_clip_low_count += motor_clip_low_count / 4.0f;
    s->motor_clip_high_count += motor_clip_high_count / 4.0f;
    s->instrumentation_steps += 1.0f;
    s->r_dist_sum += r_dist;
    s->r_hover_sum += r_hover;
    s->r_shaping_sum += r_shaping;
    s->r_omega_sum += r_omega;
    s->r_omega_xy_sum += r_omega_xy;
    s->r_omega_z_sum += r_omega_z;
}

__device__ void log_done_dev(DroneCudaLog* log, const DroneCudaState* s,
                             const DroneCudaParams* p, const DroneCudaCtx& cfg,
                             bool oob, bool timeout) {
    float steps = fmaxf(s->instrumentation_steps, 1.0f);
    atomicAdd(&log->episode_return, s->episode_return);
    atomicAdd(&log->episode_length, (float)s->episode_length);
    atomicAdd(&log->oob, oob ? 1.0f : 0.0f);
    atomicAdd(&log->timeout, timeout ? 1.0f : 0.0f);
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
    atomicAdd(&log->motor_clip_low_frac, s->motor_clip_low_count / steps);
    atomicAdd(&log->motor_clip_high_frac, s->motor_clip_high_count / steps);

    float trim[4];
    hover_trim_thrusts_dev(p, trim);
    float trim_rpm_sum = 0.0f;
    float trim_rpm_max = 0.0f;
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        float rpm = thrust_to_rpm_dev(p, trim[i]);
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
    atomicAdd(&log->r_dist, s->r_dist_sum);
    atomicAdd(&log->r_hover, s->r_hover_sum);
    atomicAdd(&log->r_shaping, s->r_shaping_sum);
    atomicAdd(&log->r_omega, s->r_omega_sum);
    atomicAdd(&log->r_omega_xy, s->r_omega_xy_sum);
    atomicAdd(&log->r_omega_z, s->r_omega_z_sum);
    atomicAdd(&log->mass_mult_mean, p->mass_mult);
    atomicAdd(&log->ixx_mult_mean, p->ixx_mult);
    atomicAdd(&log->iyy_mult_mean, p->iyy_mult);
    atomicAdd(&log->izz_mult_mean, p->izz_mult);
    atomicAdd(&log->k_thrust_mult_mean, p->k_thrust_mult);
    float k_range = dr_param_range_dev(cfg, cfg.dr_k_thrust);
    atomicAdd(&log->k_thrust_mult_min, 1.0f - k_range);
    atomicAdd(&log->k_thrust_mult_max, 1.0f + k_range);
    atomicAdd(&log->linear_drag_mult_mean, p->linear_drag_mult);
    atomicAdd(&log->yaw_drag_mult_mean, p->yaw_drag_mult);
    atomicAdd(&log->motor_lag_mult_mean, p->motor_lag_mult);
    atomicAdd(&log->com_x_mean, p->com_x);
    atomicAdd(&log->com_y_mean, p->com_y);
    atomicAdd(&log->com_z_mean, p->com_z);
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
    reset_one_dev(&s, &p, cfg, &r);
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
    s.episode_length++;

    float curr_dist = norm3_dev(sub3_dev(s.target_pos, s.pos));
    float prev_dist = norm3_dev(sub3_dev(s.target_pos, s.prev_pos));
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
    float reward = r_dist + r_hover + r_shaping + r_omega;
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
    record_step_metrics_dev(&s, &p, raw_actions, r_dist, r_hover, r_shaping, r_omega,
                            r_omega_xy, r_omega_z);
    s.episode_return += reward;

    bool oob = curr_dist > cfg.oob_radius;
    bool timeout = s.episode_length >= cfg.horizon;
    bool done = oob || timeout;
    rewards[i] = reward;
    terminals[i] = done ? 1.0f : 0.0f;

    if (done) {
        log_done_dev(cfg.log, &s, &p, cfg, oob, timeout);
        reset_one_dev(&s, &p, cfg, &r);
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
    ctx.alpha_dist = dict_float(env_kwargs, "alpha_dist", 0.782192f);
    ctx.alpha_hover = dict_float(env_kwargs, "alpha_hover", 0.071445f);
    ctx.alpha_shaping = dict_float(env_kwargs, "alpha_shaping", 3.9754f);
    ctx.alpha_omega_xy = dict_float(env_kwargs, "alpha_omega_xy",
                                    dict_float(env_kwargs, "alpha_omega", 0.00135588f));
    ctx.alpha_omega_z = dict_float(env_kwargs, "alpha_omega_z",
                                   dict_float(env_kwargs, "alpha_omega", 0.00135588f));
    ctx.alpha_omega_z_sq = dict_float(env_kwargs, "alpha_omega_z_sq", 0.0025f);
    ctx.alpha_omega_z_mult = dict_float(env_kwargs, "alpha_omega_z_mult", 5.0f);
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
    ctx.action_scale = dict_float(env_kwargs, "action_scale", 1.0f);
    ctx.reset_pos_scale = dict_float(env_kwargs, "reset_pos_scale", 1.0f);
    ctx.reset_yaw_range = dict_float(env_kwargs, "reset_yaw_range", DRONE_PI);
    ctx.reset_vel_max = dict_float(env_kwargs, "reset_vel_max", 0.0f);
    ctx.sensor_noise = dict_float(env_kwargs, "sensor_noise", 0.0f);
    ctx.action_latency_steps = latency_steps_from_seconds(dict_float(env_kwargs, "action_latency", 0.0f));
    (void)vec_kwargs;
    return ctx;
}

extern "C" void cuda_env_init(StaticVec* vec, Dict* vec_kwargs, Dict* env_kwargs) {
    DroneCudaCtx* ctx = (DroneCudaCtx*)calloc(1, sizeof(DroneCudaCtx));
    *ctx = make_host_ctx(vec, vec_kwargs, env_kwargs);
    if (ctx->task != 1) {
        fprintf(stderr, "drone CUDA env currently implements HOVER only; got task=%d\n", ctx->task);
    }
    CUDA_ENV_CHECK(cudaMalloc((void**)&ctx->states,
                              (size_t)ctx->total_agents * sizeof(DroneCudaState)));
    CUDA_ENV_CHECK(cudaMalloc((void**)&ctx->params,
                              (size_t)ctx->total_agents * sizeof(DroneCudaParams)));
    CUDA_ENV_CHECK(cudaMalloc((void**)&ctx->rng,
                              (size_t)ctx->total_agents * sizeof(curandStatePhilox4_32_10_t)));
    CUDA_ENV_CHECK(cudaMalloc((void**)&ctx->log, sizeof(DroneCudaLog)));
    CUDA_ENV_CHECK(cudaMemset(ctx->log, 0, sizeof(DroneCudaLog)));

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
    CUDA_ENV_CHECK(cudaMemset(ctx->log, 0, sizeof(DroneCudaLog)));
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
    dict_set(out, "motor_clip_low_frac", h.motor_clip_low_frac * inv);
    dict_set(out, "motor_clip_high_frac", h.motor_clip_high_frac * inv);
    dict_set(out, "hover_trim_rpm_mean", h.hover_trim_rpm_mean * inv);
    dict_set(out, "hover_trim_rpm_max", h.hover_trim_rpm_max * inv);
    dict_set(out, "hover_trim_rpm_frac_of_max", h.hover_trim_rpm_frac_of_max * inv);
    dict_set(out, "mean_rpm_FL", h.mean_rpm_FL * inv);
    dict_set(out, "mean_rpm_FR", h.mean_rpm_FR * inv);
    dict_set(out, "mean_rpm_RL", h.mean_rpm_RL * inv);
    dict_set(out, "mean_rpm_RR", h.mean_rpm_RR * inv);
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
    dict_set(out, "n", h.n);
    CUDA_ENV_CHECK(cudaMemset(ctx->log, 0, sizeof(DroneCudaLog)));
}

extern "C" void cuda_env_close(StaticVec* vec) {
    DroneCudaCtx* ctx = (DroneCudaCtx*)vec->cuda_env;
    if (ctx == NULL) return;
    CUDA_ENV_CHECK(cudaFree(ctx->states));
    CUDA_ENV_CHECK(cudaFree(ctx->params));
    CUDA_ENV_CHECK(cudaFree(ctx->rng));
    CUDA_ENV_CHECK(cudaFree(ctx->log));
    free(ctx);
    vec->cuda_env = NULL;
}
