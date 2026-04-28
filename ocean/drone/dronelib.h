// Originally made by Sam Turner and Finlay Sanders, 2025.
// Included in pufferlib under the original project's MIT license.
// https://github.com/tensaur/drone

#pragma once

#include <float.h>
#include <math.h>
#include <stdbool.h>
#include <stdlib.h>

// Visualisation properties
#define WIDTH 1080
#define HEIGHT 720
#define TRAIL_LENGTH 50

// Matrice 4D V0 CAD-aligned flight-dynamics profile.
// Scope: hover/go-to-point RL, not a full physical digital twin.
#define BASE_MASS 1.850f
#define BASE_IXX 4.0250e-2f        // kgm²
#define BASE_IYY 3.9390e-2f        // kgm²
#define BASE_IZZ 7.0230e-2f      // kgm²
#define BASE_ARM_LEN 0.24925f       // m, legacy equivalent only
#define BASE_K_THRUST 1.4863332e-7f // N/RPM^2, hover at 5525 RPM
#define BASE_K_DRAG 0.020f          // estimated yaw moment / thrust ratio
#define BASE_GRAVITY 9.81f
#define BASE_MAX_RPM 8500.0f
#define BASE_K_MOT 0.20f

#define BASE_K_ANG_DAMP 0.010f
#define BASE_B_DRAG 0.150f
#define BASE_MAX_VEL 21.0f
#define BASE_MAX_OMEGA 3.4906585f // rad/s = 200 deg/s

// DJI Matrice 4D CAD motor datums, meters, motor/action order [FL, FR, RL, RR].
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

// Simulation properties
#define GRID_X 120.0f
#define GRID_Y 120.0f
#define GRID_Z 60.0f
#define MARGIN_X (GRID_X - 1)
#define MARGIN_Y (GRID_Y - 1)
#define MARGIN_Z (GRID_Z - 1)
#define RING_RADIUS 2.0f
#define V_TARGET 0.05f

// Core Parameters
#define DT 0.002f // 500 Hz
#define ACTION_SUBSTEPS 5
#define ACTION_DT (DT * (float)ACTION_SUBSTEPS) // 100 Hz
#define MAX_ACTION_LATENCY_STEPS 8

#define DT_RNG 0.0f

// Corner to corner distance
#define MAX_DIST                                                                                   \
    sqrtf((2 * GRID_X) * (2 * GRID_X) + (2 * GRID_Y) * (2 * GRID_Y) + (2 * GRID_Z) * (2 * GRID_Z))

typedef struct Log Log;
struct Log {
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

typedef struct {
    float w, x, y, z;
} Quat;

typedef struct {
    float x, y, z;
} Vec3;

typedef struct {
    Vec3 pos;
    Vec3 vel;
    Quat orientation;
    Vec3 normal;
    float radius;
} Target;

typedef struct {
    Vec3 pos[TRAIL_LENGTH];
    int index;
    int count;
} Trail;

typedef struct {
    Vec3 pos;      // global position (x, y, z)
    Vec3 vel;      // linear velocity (u, v, w)
    Quat quat;     // roll/pitch/yaw (phi/theta/psi) as a quaternion
    Vec3 omega;    // angular velocity (p, q, r)
    float rpms[4]; // motor RPMs
} State;

typedef struct {
    Vec3 vel;         // Derivative of position
    Vec3 v_dot;       // Derivative of velocity
    Quat q_dot;       // Derivative of quaternion
    Vec3 w_dot;       // Derivative of angular velocity
    float rpm_dot[4]; // Derivative of motor RPMs
} StateDerivative;

typedef struct {
    float mass;       // kg
    float ixx;        // kgm^2
    float iyy;        // kgm^2
    float izz;        // kgm^2
    float arm_len;     // m, retained for legacy/reference only
    float motor_x[4];  // m, action/motor order: [FL, FR, RL, RR]
    float motor_y[4];  // m, action/motor order: [FL, FR, RL, RR]
    float yaw_sign[4]; // rotor reaction torque signs, same motor order
    float k_thrust;   // thrust coefficient (T = k * rpm^2)
    float k_ang_damp; // angular damping coefficient
    float k_drag;     // yaw moment constant (torque-to-thrust ratio style)
    float b_drag;     // linear drag coefficient
    float gravity;    // m/s^2 (positive, world gravity points -z)
    float max_rpm;    // RPM
    float max_vel;    // m/s (observation clamp)
    float max_omega;  // rad/s (observation clamp)
    float k_mot;      // s (motor RPM time constant)
    float action_scale; // policy action multiplier around hover trim
    float com_x;      // m, center-of-mass offset relative CAD datum
    float com_y;      // m
    float com_z;      // m, logged for DR even though V0 thrust model ignores it
    float mass_mult;
    float ixx_mult;
    float iyy_mult;
    float izz_mult;
    float k_thrust_mult;
    float linear_drag_mult;
    float yaw_drag_mult;
    float motor_lag_mult;
} Params;

typedef struct {
    float enabled;
    float mass;
    float inertia;
    float k_thrust;
    float linear_drag;
    float yaw_drag;
    float motor_lag;
    float com_xy;
    float com_z;
} DomainRandomization;

typedef struct {
    // core state and parameters
    State state;
    Params params;
    Vec3 prev_pos;

    // current target
    Target* target;

    // target buffer
    Target* buffer;
    int buffer_idx;
    int buffer_size;

    // logging utils
    float last_dist_reward;
    float episode_return;
    int episode_length;
    float score;
    float collisions;
    int rings_passed;
    float hover_score;
    float prev_potential;
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
    float action_history[MAX_ACTION_LATENCY_STEPS + 1][4];
    int action_history_idx;
} Drone;

static inline float clampf(float v, float min, float max) {
    if (v < min) return min;
    if (v > max) return max;
    return v;
}

static inline float rndf(float a, float b, unsigned int* rng) {
    return a + ((float)rand_r(rng) / (float)RAND_MAX) * (b - a);
}

static inline float dr_abs_range(float v) {
    return clampf(fabsf(v), 0.0f, 0.95f);
}

static inline bool dr_has_granular(const DomainRandomization* dr) {
    return dr != NULL && (
        fabsf(dr->mass) > 0.0f ||
        fabsf(dr->inertia) > 0.0f ||
        fabsf(dr->k_thrust) > 0.0f ||
        fabsf(dr->linear_drag) > 0.0f ||
        fabsf(dr->yaw_drag) > 0.0f ||
        fabsf(dr->motor_lag) > 0.0f ||
        fabsf(dr->com_xy) > 0.0f ||
        fabsf(dr->com_z) > 0.0f
    );
}

static inline float dr_param_range(const DomainRandomization* dr, float granular) {
    if (dr == NULL || dr->enabled <= 0.0f) return 0.0f;
    if (dr_has_granular(dr)) return dr_abs_range(granular);
    return dr_abs_range(dr->enabled);
}

static inline float dr_sample_mult(unsigned int* rng, float range) {
    range = dr_abs_range(range);
    return rndf(1.0f - range, 1.0f + range, rng);
}

static inline Vec3 add3(Vec3 a, Vec3 b) { return (Vec3){a.x + b.x, a.y + b.y, a.z + b.z}; }
static inline Vec3 sub3(Vec3 a, Vec3 b) { return (Vec3){a.x - b.x, a.y - b.y, a.z - b.z}; }
static inline Vec3 scalmul3(Vec3 a, float b) { return (Vec3){a.x * b, a.y * b, a.z * b}; }

static inline Quat add_quat(Quat a, Quat b) {
    return (Quat){a.w + b.w, a.x + b.x, a.y + b.y, a.z + b.z};
}
static inline Quat scalmul_quat(Quat a, float b) {
    return (Quat){a.w * b, a.x * b, a.y * b, a.z * b};
}

static inline float dot3(Vec3 a, Vec3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
static inline float norm3(Vec3 a) { return sqrtf(dot3(a, a)); }

static inline void clamp3(Vec3* vec, float min, float max) {
    vec->x = clampf(vec->x, min, max);
    vec->y = clampf(vec->y, min, max);
    vec->z = clampf(vec->z, min, max);
}

static inline void clamp4(float a[4], float min, float max) {
    a[0] = clampf(a[0], min, max);
    a[1] = clampf(a[1], min, max);
    a[2] = clampf(a[2], min, max);
    a[3] = clampf(a[3], min, max);
}

static inline Quat quat_mul(Quat q1, Quat q2) {
    Quat out;
    out.w = q1.w * q2.w - q1.x * q2.x - q1.y * q2.y - q1.z * q2.z;
    out.x = q1.w * q2.x + q1.x * q2.w + q1.y * q2.z - q1.z * q2.y;
    out.y = q1.w * q2.y - q1.x * q2.z + q1.y * q2.w + q1.z * q2.x;
    out.z = q1.w * q2.z + q1.x * q2.y - q1.y * q2.x + q1.z * q2.w;
    return out;
}

static inline void quat_normalize(Quat* q) {
    float n = sqrtf(q->w * q->w + q->x * q->x + q->y * q->y + q->z * q->z);
    if (n > 0.0f) {
        q->w /= n;
        q->x /= n;
        q->y /= n;
        q->z /= n;
    }
}

static inline Vec3 quat_rotate(Quat q, Vec3 v) {
    Quat qv = (Quat){0.0f, v.x, v.y, v.z};
    Quat tmp = quat_mul(q, qv);
    Quat q_conj = (Quat){q.w, -q.x, -q.y, -q.z};
    Quat res = quat_mul(tmp, q_conj);
    return (Vec3){res.x, res.y, res.z};
}

static inline Quat quat_inverse(Quat q) { return (Quat){q.w, -q.x, -q.y, -q.z}; }

static inline Quat rndquat(unsigned int* rng) {
    float u1 = rndf(0.0f, 1.0f, rng);
    float u2 = rndf(0.0f, 1.0f, rng);
    float u3 = rndf(0.0f, 1.0f, rng);

    float sqrt_1_minus_u1 = sqrtf(1.0f - u1);
    float sqrt_u1 = sqrtf(u1);

    float pi_2_u2 = 2.0f * (float)M_PI * u2;
    float pi_2_u3 = 2.0f * (float)M_PI * u3;

    Quat q;
    q.w = sqrt_1_minus_u1 * sinf(pi_2_u2);
    q.x = sqrt_1_minus_u1 * cosf(pi_2_u2);
    q.y = sqrt_u1 * sinf(pi_2_u3);
    q.z = sqrt_u1 * cosf(pi_2_u3);
    return q;
}

static inline Quat quat_from_axis_angle(Vec3 axis, float angle) {
    float half = angle * 0.5f;
    float s = sinf(half);
    return (Quat){cosf(half), axis.x * s, axis.y * s, axis.z * s};
}

static inline Target rndring(unsigned int* rng, float radius) {
    Target ring = (Target){0};

    ring.pos.x = rndf(-GRID_X + 2 * radius, GRID_X - 2 * radius, rng);
    ring.pos.y = rndf(-GRID_Y + 2 * radius, GRID_Y - 2 * radius, rng);
    ring.pos.z = rndf(-GRID_Z + 2 * radius, GRID_Z - 2 * radius, rng);

    ring.orientation = rndquat(rng);

    Vec3 base_normal = (Vec3){0.0f, 0.0f, 1.0f};
    ring.normal = quat_rotate(ring.orientation, base_normal);

    ring.radius = radius;
    return ring;
}

static inline float max_motor_thrust(const Params* p) {
    return p->k_thrust * p->max_rpm * p->max_rpm;
}

static inline bool solve_allocation(const Params* p, float total_thrust, Vec3 torque, float out[4]) {
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

    float max_t = max_motor_thrust(p);
    for (int i = 0; i < 4; i++) {
        out[i] = clampf(a[i][4], 0.0f, max_t);
    }
    return true;
}

static inline void hover_trim_thrusts(const Params* p, float out[4]) {
    if (!solve_allocation(p, p->mass * p->gravity, (Vec3){0.0f, 0.0f, 0.0f}, out)) {
        float fallback = 0.25f * p->mass * p->gravity;
        for (int i = 0; i < 4; i++) out[i] = fallback;
    }
}

static inline float thrust_to_rpm(const Params* p, float thrust) {
    thrust = clampf(thrust, 0.0f, max_motor_thrust(p));
    return sqrtf(thrust / p->k_thrust);
}

static inline void init_drone(Drone* drone, unsigned int* rng, const DomainRandomization* dr) {
    float mass_mult = dr_sample_mult(rng, dr_param_range(dr, dr == NULL ? 0.0f : dr->mass));
    float ixx_mult = dr_sample_mult(rng, dr_param_range(dr, dr == NULL ? 0.0f : dr->inertia));
    float iyy_mult = dr_sample_mult(rng, dr_param_range(dr, dr == NULL ? 0.0f : dr->inertia));
    float izz_mult = dr_sample_mult(rng, dr_param_range(dr, dr == NULL ? 0.0f : dr->inertia));
    float k_thrust_mult = dr_sample_mult(rng, dr_param_range(dr, dr == NULL ? 0.0f : dr->k_thrust));
    float linear_drag_mult = dr_sample_mult(rng, dr_param_range(dr, dr == NULL ? 0.0f : dr->linear_drag));
    float yaw_drag_mult = dr_sample_mult(rng, dr_param_range(dr, dr == NULL ? 0.0f : dr->yaw_drag));
    float motor_lag_mult = dr_sample_mult(rng, dr_param_range(dr, dr == NULL ? 0.0f : dr->motor_lag));

    float com_xy = (dr != NULL && dr->enabled > 0.0f) ? fabsf(dr->com_xy) : 0.0f;
    float com_z_range = (dr != NULL && dr->enabled > 0.0f) ? fabsf(dr->com_z) : 0.0f;
    float com_x = rndf(-com_xy, com_xy, rng);
    float com_y = rndf(-com_xy, com_xy, rng);
    float com_z = rndf(-com_z_range, com_z_range, rng);

    drone->params.arm_len = BASE_ARM_LEN;
    drone->params.mass = BASE_MASS * mass_mult;
    drone->params.ixx = BASE_IXX * ixx_mult;
    drone->params.iyy = BASE_IYY * iyy_mult;
    drone->params.izz = BASE_IZZ * izz_mult;
    drone->params.k_thrust = BASE_K_THRUST * k_thrust_mult;
    drone->params.k_ang_damp = BASE_K_ANG_DAMP;
    drone->params.k_drag = BASE_K_DRAG * yaw_drag_mult;
    drone->params.b_drag = BASE_B_DRAG * linear_drag_mult;
    drone->params.gravity = BASE_GRAVITY;

    drone->params.max_rpm = BASE_MAX_RPM;
    drone->params.max_vel = BASE_MAX_VEL;
    drone->params.max_omega = BASE_MAX_OMEGA;

    drone->params.k_mot = BASE_K_MOT * motor_lag_mult;
    drone->params.action_scale = 1.0f;
    drone->params.com_x = com_x;
    drone->params.com_y = com_y;
    drone->params.com_z = com_z;
    drone->params.mass_mult = mass_mult;
    drone->params.ixx_mult = ixx_mult;
    drone->params.iyy_mult = iyy_mult;
    drone->params.izz_mult = izz_mult;
    drone->params.k_thrust_mult = k_thrust_mult;
    drone->params.linear_drag_mult = linear_drag_mult;
    drone->params.yaw_drag_mult = yaw_drag_mult;
    drone->params.motor_lag_mult = motor_lag_mult;

    // Effective lever arms are expressed relative to the sampled COM.
    drone->params.motor_x[0] = BASE_MOTOR_FL_X - com_x;
    drone->params.motor_y[0] = BASE_MOTOR_FL_Y - com_y;
    drone->params.yaw_sign[0] = BASE_YAW_SIGN_FL;
    drone->params.motor_x[1] = BASE_MOTOR_FR_X - com_x;
    drone->params.motor_y[1] = BASE_MOTOR_FR_Y - com_y;
    drone->params.yaw_sign[1] = BASE_YAW_SIGN_FR;
    drone->params.motor_x[2] = BASE_MOTOR_RL_X - com_x;
    drone->params.motor_y[2] = BASE_MOTOR_RL_Y - com_y;
    drone->params.yaw_sign[2] = BASE_YAW_SIGN_RL;
    drone->params.motor_x[3] = BASE_MOTOR_RR_X - com_x;
    drone->params.motor_y[3] = BASE_MOTOR_RR_Y - com_y;
    drone->params.yaw_sign[3] = BASE_YAW_SIGN_RR;

    float trim[4];
    hover_trim_thrusts(&drone->params, trim);
    for (int i = 0; i < 4; i++)
        drone->state.rpms[i] = thrust_to_rpm(&drone->params, trim[i]);

    drone->state.pos = (Vec3){0.0f, 0.0f, 0.0f};
    drone->prev_pos = drone->state.pos;
    drone->state.vel = (Vec3){0.0f, 0.0f, 0.0f};
    drone->state.omega = (Vec3){0.0f, 0.0f, 0.0f};
    drone->state.quat = (Quat){1.0f, 0.0f, 0.0f, 0.0f};
    drone->action_history_idx = 0;
    for (int h = 0; h < MAX_ACTION_LATENCY_STEPS + 1; h++) {
        for (int m = 0; m < 4; m++) {
            drone->action_history[h][m] = 0.0f;
        }
    }
}

static inline void compute_derivatives(State* state, Params* params, float* actions,
                                       StateDerivative* derivatives) {
    float trim[4];
    hover_trim_thrusts(params, trim);
    float max_thrust = max_motor_thrust(params);
    float target_rpms[4];
    for (int i = 0; i < 4; i++) {
        float action = clampf(actions[i] * params->action_scale, -1.0f, 1.0f);
        float target_thrust = action >= 0.0f
            ? trim[i] + action * (max_thrust - trim[i])
            : trim[i] + action * trim[i];
        target_rpms[i] = thrust_to_rpm(params, target_thrust);
    }

    float rpm_dot[4];
    for (int i = 0; i < 4; i++) {
        rpm_dot[i] = (1.0f / params->k_mot) * (target_rpms[i] - state->rpms[i]);
    }

    // motor thrusts
    float T[4];
    for (int i = 0; i < 4; i++) {
        float rpm = state->rpms[i];
        if (rpm < 0.0f) rpm = 0.0f;
        T[i] = params->k_thrust * rpm * rpm;
    }

    // body frame net force
    Vec3 F_prop_body = (Vec3){0.0f, 0.0f, T[0] + T[1] + T[2] + T[3]};

    // body frame force -> world frame force
    Vec3 F_prop = quat_rotate(state->quat, F_prop_body);

    // world frame linear drag
    Vec3 F_aero;
    F_aero.x = -params->b_drag * state->vel.x;
    F_aero.y = -params->b_drag * state->vel.y;
    F_aero.z = -params->b_drag * state->vel.z;

    // linear acceleration
    Vec3 v_dot;
    v_dot.x = (F_prop.x + F_aero.x) / params->mass;
    v_dot.y = (F_prop.y + F_aero.y) / params->mass;
    v_dot.z = ((F_prop.z + F_aero.z) / params->mass) - params->gravity;

    // quaternion rates
    Quat omega_q = (Quat){0.0f, state->omega.x, state->omega.y, state->omega.z};
    Quat q_dot = quat_mul(state->quat, omega_q);
    q_dot.w *= 0.5f;
    q_dot.x *= 0.5f;
    q_dot.y *= 0.5f;
    q_dot.z *= 0.5f;

    // body frame torques (plus copter)
    // Vec3 Tau_prop;
    // Tau_prop.x = params->arm_len*(T[1] - T[3]);
    // Tau_prop.y = params->arm_len*(T[2] - T[0]);
    // Tau_prop.z = params->k_drag*(T[0] - T[1] + T[2] - T[3]);

    Vec3 Tau_prop;
    // CAD-aligned body torques from exact motor lever arms.
    // Motor/action order: [FL, FR, RL, RR].
    // tau_x = sum(y_i*T_i), tau_y = sum(-x_i*T_i).
    Tau_prop.x = 0.0f;
    Tau_prop.y = 0.0f;
    Tau_prop.z = 0.0f;
    for (int i = 0; i < 4; i++) {
        Tau_prop.x += params->motor_y[i] * T[i];
        Tau_prop.y += -params->motor_x[i] * T[i];
        Tau_prop.z += params->k_drag * params->yaw_sign[i] * T[i];
    }

    // torque from angular damping
    Vec3 Tau_aero;
    Tau_aero.x = -params->k_ang_damp * state->omega.x;
    Tau_aero.y = -params->k_ang_damp * state->omega.y;
    Tau_aero.z = -params->k_ang_damp * state->omega.z;

    // gyroscopic torque
    Vec3 Tau_iner;
    Tau_iner.x = (params->iyy - params->izz) * state->omega.y * state->omega.z;
    Tau_iner.y = (params->izz - params->ixx) * state->omega.z * state->omega.x;
    Tau_iner.z = (params->ixx - params->iyy) * state->omega.x * state->omega.y;

    // angular velocity rates
    Vec3 w_dot;
    w_dot.x = (Tau_prop.x + Tau_aero.x + Tau_iner.x) / params->ixx;
    w_dot.y = (Tau_prop.y + Tau_aero.y + Tau_iner.y) / params->iyy;
    w_dot.z = (Tau_prop.z + Tau_aero.z + Tau_iner.z) / params->izz;

    derivatives->vel = state->vel;
    derivatives->v_dot = v_dot;
    derivatives->q_dot = q_dot;
    derivatives->w_dot = w_dot;
    for (int i = 0; i < 4; i++) {
        derivatives->rpm_dot[i] = rpm_dot[i];
    }
}

static inline void step(State* initial, StateDerivative* deriv, float dt, State* output) {
    output->pos = add3(initial->pos, scalmul3(deriv->vel, dt));
    output->vel = add3(initial->vel, scalmul3(deriv->v_dot, dt));
    output->quat = add_quat(initial->quat, scalmul_quat(deriv->q_dot, dt));
    output->omega = add3(initial->omega, scalmul3(deriv->w_dot, dt));
    for (int i = 0; i < 4; i++) {
        output->rpms[i] = initial->rpms[i] + deriv->rpm_dot[i] * dt;
    }
    quat_normalize(&output->quat);
}

static inline void rk4_step(State* state, Params* params, float* actions, float dt) {
    StateDerivative k1, k2, k3, k4;
    State temp_state;

    compute_derivatives(state, params, actions, &k1);

    step(state, &k1, dt * 0.5f, &temp_state);
    compute_derivatives(&temp_state, params, actions, &k2);

    step(state, &k2, dt * 0.5f, &temp_state);
    compute_derivatives(&temp_state, params, actions, &k3);

    step(state, &k3, dt, &temp_state);
    compute_derivatives(&temp_state, params, actions, &k4);

    float dt_6 = dt / 6.0f;

    state->pos.x += (k1.vel.x + 2.0f * k2.vel.x + 2.0f * k3.vel.x + k4.vel.x) * dt_6;
    state->pos.y += (k1.vel.y + 2.0f * k2.vel.y + 2.0f * k3.vel.y + k4.vel.y) * dt_6;
    state->pos.z += (k1.vel.z + 2.0f * k2.vel.z + 2.0f * k3.vel.z + k4.vel.z) * dt_6;

    state->vel.x += (k1.v_dot.x + 2.0f * k2.v_dot.x + 2.0f * k3.v_dot.x + k4.v_dot.x) * dt_6;
    state->vel.y += (k1.v_dot.y + 2.0f * k2.v_dot.y + 2.0f * k3.v_dot.y + k4.v_dot.y) * dt_6;
    state->vel.z += (k1.v_dot.z + 2.0f * k2.v_dot.z + 2.0f * k3.v_dot.z + k4.v_dot.z) * dt_6;

    state->quat.w += (k1.q_dot.w + 2.0f * k2.q_dot.w + 2.0f * k3.q_dot.w + k4.q_dot.w) * dt_6;
    state->quat.x += (k1.q_dot.x + 2.0f * k2.q_dot.x + 2.0f * k3.q_dot.x + k4.q_dot.x) * dt_6;
    state->quat.y += (k1.q_dot.y + 2.0f * k2.q_dot.y + 2.0f * k3.q_dot.y + k4.q_dot.y) * dt_6;
    state->quat.z += (k1.q_dot.z + 2.0f * k2.q_dot.z + 2.0f * k3.q_dot.z + k4.q_dot.z) * dt_6;

    state->omega.x += (k1.w_dot.x + 2.0f * k2.w_dot.x + 2.0f * k3.w_dot.x + k4.w_dot.x) * dt_6;
    state->omega.y += (k1.w_dot.y + 2.0f * k2.w_dot.y + 2.0f * k3.w_dot.y + k4.w_dot.y) * dt_6;
    state->omega.z += (k1.w_dot.z + 2.0f * k2.w_dot.z + 2.0f * k3.w_dot.z + k4.w_dot.z) * dt_6;

    for (int i = 0; i < 4; i++) {
        state->rpms[i] +=
            (k1.rpm_dot[i] + 2.0f * k2.rpm_dot[i] + 2.0f * k3.rpm_dot[i] + k4.rpm_dot[i]) * dt_6;
    }

    quat_normalize(&state->quat);
}

static inline void move_drone(Drone* drone, float* actions) {
    clamp4(actions, -1.0f, 1.0f);

    for (int s = 0; s < ACTION_SUBSTEPS; s++) {
        rk4_step(&drone->state, &drone->params, actions, DT);

        clamp3(&drone->state.vel, -drone->params.max_vel, drone->params.max_vel);
        clamp3(&drone->state.omega, -drone->params.max_omega, drone->params.max_omega);

        for (int i = 0; i < 4; i++) {
            drone->state.rpms[i] = clampf(drone->state.rpms[i], 0.0f, drone->params.max_rpm);
        }
    }
}

static inline void reset_rings(unsigned int* rng, Target* ring_buffer, int num_rings) {
    ring_buffer[0] = rndring(rng, RING_RADIUS);

    // ensure rings are spaced at least 2*ring_radius apart
    for (int i = 1; i < num_rings; i++) {
        do {
            ring_buffer[i] = rndring(rng, RING_RADIUS);
        } while (norm3(sub3(ring_buffer[i].pos, ring_buffer[i - 1].pos)) < 2.0f * RING_RADIUS);
    }
}

static inline Drone* nearest_drone(Drone* agent, Drone* others, int num_agents) {
    float min_dist = FLT_MAX;
    Drone* nearest = NULL;

    for (int i = 0; i < num_agents; i++) {
        Drone* other = &others[i];
        if (other == agent) continue;

        float dist = norm3(sub3(agent->state.pos, other->state.pos));

        if (dist < min_dist) {
            min_dist = dist;
            nearest = other;
        }
    }

    return nearest;
}

static inline int check_ring(Drone* drone, Target* ring) {
    // previous dot product negative if on the 'entry' side of the ring's plane
    float prev_dot = dot3(sub3(drone->prev_pos, ring->pos), ring->normal);
    float new_dot = dot3(sub3(drone->state.pos, ring->pos), ring->normal);

    bool valid_dir = (prev_dot < 0.0f && new_dot > 0.0f);
    bool invalid_dir = (prev_dot > 0.0f && new_dot < 0.0f);

    // if we have crossed the plane of the ring
    if (valid_dir || invalid_dir) {
        // find intesection with ring's plane
        Vec3 dir = sub3(drone->state.pos, drone->prev_pos);
        float denom = dot3(ring->normal, dir);
        if (fabsf(denom) < 1e-9f) return 0;

        float t = -prev_dot / denom;
        Vec3 intersection = add3(drone->prev_pos, scalmul3(dir, t));
        float dist = norm3(sub3(intersection, ring->pos));

        if (dist < (ring->radius - 0.5f) && valid_dir) {
            return 1;
        } else if (dist < ring->radius + 0.5f) {
            return -1;
        }
    }

    return 0;
}

static inline bool check_collision(Drone* agent, Drone* others, int num_agents) {
    if (num_agents <= 1) return false;

    Drone* nearest = nearest_drone(agent, others, num_agents);
    Vec3 to_nearest = sub3(agent->state.pos, nearest->state.pos);
    float nearest_dist = norm3(to_nearest);

    return nearest_dist < 0.1f;
}

float hover_potential(Drone* agent, float hover_dist, float hover_omega, float hover_vel) {
    float dist = norm3(sub3(agent->target->pos, agent->state.pos));
    float vel = norm3(agent->state.vel);
    float omega = norm3(agent->state.omega);

    float d = 1.0f / (1.0f + dist / hover_dist);
    float v = 1.0f / (1.0f + vel / hover_vel);
    float w = 1.0f / (1.0f + omega / hover_omega);

    return d * (0.7f + 0.15f * v + 0.15f * w);
}

float check_hover(Drone* agent, float hover_dist, float hover_omega, float hover_vel) {
    float dist = norm3(sub3(agent->target->pos, agent->state.pos));
    float vel = norm3(agent->state.vel);
    float omega = norm3(agent->state.omega);

    float d = dist / (hover_dist * 10.0f);
    float v = vel / (hover_vel * 10.0f);
    float w = omega / (hover_omega * 10.0f);

    float score = 1.0f - 0.7f * d - 0.15f * v - 0.15f * w;
    return score > 0.0f ? score : 0.0f;
}

void compute_drone_observations(Drone* agent, float* observations) {
    int idx = 0;

    // choose the hemisphere with w >= 0
    // to avoid observation sign ambiguity
    Quat q = agent->state.quat;
    //if (q.w < 0.0f) {q.w=-q.w; q.x=-q.x; q.y=-q.y; q.z=-q.z;}

    Quat q_inv = quat_inverse(q);
    Vec3 linear_vel_body = quat_rotate(q_inv, agent->state.vel);
    Vec3 to_target_world = sub3(agent->target->pos, agent->state.pos);
    Vec3 to_target = quat_rotate(q_inv, to_target_world);

    // we should probably clamp the overall velocity
    float denom = agent->params.max_vel * 1.7320508f; // sqrt(3)
    observations[idx++] = linear_vel_body.x / denom;
    observations[idx++] = linear_vel_body.y / denom;
    observations[idx++] = linear_vel_body.z / denom;

    observations[idx++] = agent->state.omega.x / agent->params.max_omega;
    observations[idx++] = agent->state.omega.y / agent->params.max_omega;
    observations[idx++] = agent->state.omega.z / agent->params.max_omega;

    observations[idx++] = q.w;
    observations[idx++] = q.x;
    observations[idx++] = q.y;
    observations[idx++] = q.z;

    // this is body frame so we have to be careful about scaling
    // because distances are relative to the drone orientation
    observations[idx++] = tanhf(to_target.x * 0.1f);
    observations[idx++] = tanhf(to_target.y * 0.1f);
    observations[idx++] = tanhf(to_target.z * 0.1f);

    observations[idx++] = tanhf(to_target.x * 10.0f);
    observations[idx++] = tanhf(to_target.y * 10.0f);
    observations[idx++] = tanhf(to_target.z * 10.0f);

    Vec3 normal_body = quat_rotate(q_inv, agent->target->normal);
    observations[idx++] = normal_body.x;
    observations[idx++] = normal_body.y;
    observations[idx++] = normal_body.z;

    // rpms should always be last in the obs
    observations[idx++] = agent->state.rpms[0] / agent->params.max_rpm;
    observations[idx++] = agent->state.rpms[1] / agent->params.max_rpm;
    observations[idx++] = agent->state.rpms[2] / agent->params.max_rpm;
    observations[idx++] = agent->state.rpms[3] / agent->params.max_rpm;
}
