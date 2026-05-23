#include "drone.h"
#include "m4d_deployment_runtime.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    float* data;
    size_t len;
    size_t cap;
} FloatVec;

static void float_vec_push(FloatVec* vec, float value) {
    if (vec->len == vec->cap) {
        size_t next = vec->cap == 0 ? 1024 : vec->cap * 2;
        float* data = (float*)realloc(vec->data, next * sizeof(float));
        if (data == NULL) {
            fprintf(stderr, "Failed to grow metric sample buffer to %zu floats\n", next);
            exit(1);
        }
        vec->data = data;
        vec->cap = next;
    }
    vec->data[vec->len++] = value;
}

static int cmp_float(const void* a, const void* b) {
    float fa = *(const float*)a;
    float fb = *(const float*)b;
    return (fa > fb) - (fa < fb);
}

static float percentile(FloatVec* vec, float q) {
    if (vec->len == 0) return 0.0f;
    qsort(vec->data, vec->len, sizeof(float), cmp_float);
    float idx_f = q * (float)(vec->len - 1);
    size_t idx = (size_t)(idx_f + 0.5f);
    if (idx >= vec->len) idx = vec->len - 1;
    return vec->data[idx];
}

void c_close_client(Client* client) {
    (void)client;
}

static float env_float(const char* key, float fallback) {
    const char* value = getenv(key);
    return value == NULL ? fallback : (float)atof(value);
}

static int env_action_mode(const char* key, int fallback) {
    const char* value = getenv(key);
    if (value == NULL) return fallback;
    if (strcmp(value, "1") == 0 || strcmp(value, "normalized") == 0 ||
        strcmp(value, "normalized_thrust") == 0) {
        return M4D_ACTION_NORMALIZED_THRUST;
    }
    if (strcmp(value, "0") == 0 || strcmp(value, "hover_trim") == 0) {
        return M4D_ACTION_HOVER_TRIM;
    }
    fprintf(stderr, "Unknown M4D_ACTION_MODE='%s'; valid: 0, hover_trim, 1, normalized_thrust\n",
            value);
    exit(2);
}

static float slew_limited_value(float raw, float prev, int has_prev, float da_max) {
    if (!has_prev || da_max <= 0.0f) {
        return raw;
    }
    float delta = raw - prev;
    if (delta > da_max) delta = da_max;
    if (delta < -da_max) delta = -da_max;
    return prev + delta;
}

static void apply_slew_limiter_for_agent(float* actions, const float* prev_actions,
                                         int has_prev, float da_max) {
    if (!has_prev || da_max <= 0.0f) {
        return;
    }
    for (int a = 0; a < 4; a++) {
        actions[a] = slew_limited_value(actions[a], prev_actions[a], 1, da_max);
    }
}

static float mean_abs_delta4(const float* a, const float* b) {
    float delta = 0.0f;
    for (int i = 0; i < 4; i++) {
        delta += fabsf(a[i] - b[i]);
    }
    return 0.25f * delta;
}

static float post_lag_motor_jump_for_agent(const Drone* agent, const float* reset_actions,
                                           const float* continued_actions) {
    Drone reset_drone = *agent;
    Drone continued_drone = *agent;
    float reset_local[4] = {
        reset_actions[0],
        reset_actions[1],
        reset_actions[2],
        reset_actions[3],
    };
    float continued_local[4] = {
        continued_actions[0],
        continued_actions[1],
        continued_actions[2],
        continued_actions[3],
    };
    move_drone(&reset_drone, reset_local);
    move_drone(&continued_drone, continued_local);
    float inv_max_rpm = 1.0f / fmaxf(agent->params.max_rpm, 1e-6f);
    float jump = 0.0f;
    for (int i = 0; i < 4; i++) {
        jump += fabsf(reset_drone.state.rpms[i] - continued_drone.state.rpms[i]) * inv_max_rpm;
    }
    return 0.25f * jump;
}

static void configure_common(DroneEnv* env, int num_agents) {
    env->num_agents = num_agents;
    env->max_rings = 10;
    env->task = HOVER;

    env->alpha_dist = 0.782192f;
    env->alpha_hover = 0.071445f;
    env->alpha_shaping = 3.9754f;
    env->alpha_omega = 0.00135588f;
    env->alpha_omega_xy = 0.00135588f;
    env->alpha_omega_z = 0.00135588f;
    env->alpha_omega_z_sq = 0.0025f;
    env->alpha_omega_z_mult = 5.0f;

    env->hover_target_dist = 5.0f;
    env->oob_radius = 12.0f;
    env->hover_dist = 0.1f;
    env->hover_omega = 0.1f;
    env->hover_vel = 0.1f;

    env->action_scale = 0.5f;
    env->action_mode = M4D_ACTION_HOVER_TRIM;
    env->normalized_thrust_min = 0.0f;
    env->normalized_thrust_max = 1.0f;
    env->reset_pos_scale = 1.0f;
    env->reset_yaw_range = 3.14159f;
    env->reset_vel_max = 0.2f;
    env->action_latency = 0.0f;
    env->sensor_noise = 0.0f;
    env->dr_authority_gated = 0.0f;
    env->dr_usable_t2w_min = 0.0f;
    env->dr_usable_t2w_max = 0.0f;
    env->dr_mass_min = 1.0f;
    env->dr_mass_max = 1.0f;
    env->dr_inertia_min = 1.0f;
    env->dr_inertia_max = 1.0f;
    env->dr_motor_thrust_min = 1.0f;
    env->dr_motor_thrust_max = 1.0f;
    env->dr_motor_tau_min = BASE_K_MOT;
    env->dr_motor_tau_max = BASE_K_MOT;
    env->dr_yaw_torque_min = 1.0f;
    env->dr_yaw_torque_max = 1.0f;
    env->dr_linear_drag_min = 1.0f;
    env->dr_linear_drag_max = 1.0f;
    env->dr_angular_damping_min = 1.0f;
    env->dr_angular_damping_max = 1.0f;
}

static void configure_baseline(DroneEnv* env, int num_agents) {
    configure_common(env, num_agents);

    env->domain_randomization = 0.0f;
    env->dr_mass = 0.0f;
    env->dr_inertia = 0.0f;
    env->dr_k_thrust = 0.0f;
    env->dr_linear_drag = 0.0f;
    env->dr_yaw_drag = 0.0f;
    env->dr_motor_lag = 0.0f;
    env->dr_com_xy = 0.0f;
    env->dr_com_z = 0.0f;
}

static void configure_nominal_normalized(DroneEnv* env, int num_agents) {
    configure_baseline(env, num_agents);

    env->alpha_omega_z_mult = 5.0f;
    env->action_scale = 1.0f;
    env->action_mode = M4D_ACTION_NORMALIZED_THRUST;
    env->normalized_thrust_min = 0.0f;
    env->normalized_thrust_max = 0.85f;
    env->reset_yaw_range = 3.14159f;
    env->reset_vel_max = 0.2f;
}

static void configure_dr_light(DroneEnv* env, int num_agents) {
    configure_common(env, num_agents);

    env->domain_randomization = 1.0f;
    env->dr_mass = 0.05f;
    env->dr_inertia = 0.10f;
    env->dr_k_thrust = 0.10f;
    env->dr_linear_drag = 0.20f;
    env->dr_yaw_drag = 0.20f;
    env->dr_motor_lag = 0.05f;
    env->dr_com_xy = 0.01f;
    env->dr_com_z = 0.0f;
}

static void configure_dr_narrow20(DroneEnv* env, int num_agents) {
    configure_common(env, num_agents);

    env->domain_randomization = 1.0f;
    env->dr_mass = 0.20f;
    env->dr_inertia = 0.20f;
    env->dr_k_thrust = 0.20f;
    env->dr_linear_drag = 0.20f;
    env->dr_yaw_drag = 0.20f;
    env->dr_motor_lag = 0.20f;
    env->dr_com_xy = 0.01f;
    env->dr_com_z = 0.01f;

    env->alpha_omega_z_mult = 5.0f;
    env->action_scale = 1.0f;
    env->action_mode = M4D_ACTION_NORMALIZED_THRUST;
    env->normalized_thrust_min = 0.0f;
    env->normalized_thrust_max = 0.85f;
    env->reset_yaw_range = 3.14159f;
    env->reset_vel_max = 0.2f;
}

static void configure_dr_medium(DroneEnv* env, int num_agents) {
    configure_common(env, num_agents);

    env->domain_randomization = 1.0f;
    env->dr_mass = 0.10f;
    env->dr_inertia = 0.20f;
    env->dr_k_thrust = 0.20f;
    env->dr_linear_drag = 0.40f;
    env->dr_yaw_drag = 0.40f;
    env->dr_motor_lag = 0.15f;
    env->dr_com_xy = 0.02f;
    env->dr_com_z = 0.03f;

    env->action_scale = 0.7f;
    env->action_latency = 0.01f;
    env->sensor_noise = 0.01f;
}

static void configure_dr_hard(DroneEnv* env, int num_agents) {
    configure_common(env, num_agents);

    env->domain_randomization = 1.0f;
    env->dr_mass = 0.20f;
    env->dr_inertia = 0.40f;
    env->dr_k_thrust = 0.40f;
    env->dr_linear_drag = 0.80f;
    env->dr_yaw_drag = 0.80f;
    env->dr_motor_lag = 0.25f;
    env->dr_com_xy = 0.04f;
    env->dr_com_z = 0.05f;

    env->action_scale = 0.7f;
    env->action_latency = 0.02f;
    env->sensor_noise = 0.02f;
}

static void configure_dr_hard_small(DroneEnv* env, int num_agents) {
    configure_dr_hard(env, num_agents);

    env->reset_pos_scale = 0.75f;
    env->reset_vel_max = 0.15f;
}

static void configure_dr_family_v1(DroneEnv* env, int num_agents) {
    configure_common(env, num_agents);

    env->domain_randomization = 1.0f;
    env->dr_mass = 0.25f;
    env->dr_inertia = 0.50f;
    env->dr_k_thrust = 0.35f;
    env->dr_linear_drag = 0.75f;
    env->dr_yaw_drag = 0.75f;
    env->dr_motor_lag = 0.50f;
    env->dr_com_xy = 0.025f;
    env->dr_com_z = 0.015f;

    env->alpha_omega_z_mult = 5.0f;
    env->action_scale = 1.0f;
    env->action_mode = M4D_ACTION_NORMALIZED_THRUST;
    env->normalized_thrust_min = 0.0f;
    env->normalized_thrust_max = 1.0f;
    env->reset_yaw_range = 3.14159f;
    env->reset_vel_max = 0.2f;
    env->action_latency = 0.01f;
    env->sensor_noise = 0.01f;
}

static void configure_dr_family_v05(DroneEnv* env, int num_agents) {
    configure_common(env, num_agents);

    env->domain_randomization = 1.0f;
    env->dr_authority_gated = 1.0f;
    env->dr_usable_t2w_min = 2.2f;
    env->dr_usable_t2w_max = 3.8f;
    env->dr_mass_min = 0.85f;
    env->dr_mass_max = 1.15f;
    env->dr_inertia_min = 0.70f;
    env->dr_inertia_max = 1.40f;
    env->dr_motor_thrust_min = 0.90f;
    env->dr_motor_thrust_max = 1.10f;
    env->dr_motor_tau_min = 0.08f;
    env->dr_motor_tau_max = 0.20f;
    env->dr_yaw_torque_min = 0.80f;
    env->dr_yaw_torque_max = 1.25f;
    env->dr_com_xy = 0.015f;
    env->dr_com_z = 0.010f;
    env->dr_linear_drag_min = 0.50f;
    env->dr_linear_drag_max = 1.50f;
    env->dr_angular_damping_min = 0.50f;
    env->dr_angular_damping_max = 1.50f;

    env->alpha_omega_z_mult = 5.0f;
    env->action_scale = 1.0f;
    env->action_mode = M4D_ACTION_NORMALIZED_THRUST;
    env->normalized_thrust_min = 0.0f;
    env->normalized_thrust_max = 0.85f;
    env->reset_yaw_range = 3.14159f;
    env->reset_vel_max = 0.2f;
    env->action_latency = 0.0f;
    env->sensor_noise = 0.0f;
}

static void configure_dr_family_v1a(DroneEnv* env, int num_agents) {
    configure_common(env, num_agents);

    env->domain_randomization = 1.0f;
    env->dr_authority_gated = 1.0f;
    env->dr_usable_t2w_min = 2.0f;
    env->dr_usable_t2w_max = 4.2f;
    env->dr_mass_min = 0.80f;
    env->dr_mass_max = 1.25f;
    env->dr_inertia_min = 0.60f;
    env->dr_inertia_max = 1.60f;
    env->dr_motor_thrust_min = 0.85f;
    env->dr_motor_thrust_max = 1.15f;
    env->dr_motor_tau_min = 0.06f;
    env->dr_motor_tau_max = 0.24f;
    env->dr_yaw_torque_min = 0.75f;
    env->dr_yaw_torque_max = 1.30f;
    env->dr_com_xy = 0.025f;
    env->dr_com_z = 0.015f;
    env->dr_linear_drag_min = 0.25f;
    env->dr_linear_drag_max = 2.00f;
    env->dr_angular_damping_min = 0.50f;
    env->dr_angular_damping_max = 2.00f;

    env->alpha_omega_z_mult = 5.0f;
    env->action_scale = 1.0f;
    env->action_mode = M4D_ACTION_NORMALIZED_THRUST;
    env->normalized_thrust_min = 0.0f;
    env->normalized_thrust_max = 0.85f;
    env->reset_yaw_range = 3.14159f;
    env->reset_vel_max = 0.2f;
    env->action_latency = 0.0f;
    env->sensor_noise = 0.0f;
}

static void configure_dr_family_v1a_super_large(DroneEnv* env, int num_agents) {
    configure_dr_family_v1a(env, num_agents);

    env->dr_usable_t2w_min = 2.6f;
    env->dr_usable_t2w_max = 3.4f;
    env->dr_mass_min = 1.18f;
    env->dr_mass_max = 1.25f;
    env->dr_inertia_min = 1.35f;
    env->dr_inertia_max = 1.60f;
    env->dr_motor_thrust_min = 0.92f;
    env->dr_motor_thrust_max = 1.08f;
    env->dr_motor_tau_min = 0.14f;
    env->dr_motor_tau_max = 0.22f;
    env->dr_com_xy = 0.015f;
    env->dr_com_z = 0.010f;
    env->reset_pos_scale = 0.25f;
}

static void configure_dr_family_v1a_ultra_large(DroneEnv* env, int num_agents) {
    configure_dr_family_v1a(env, num_agents);

    env->dr_usable_t2w_min = 3.0f;
    env->dr_usable_t2w_max = 4.0f;
    env->dr_mass_min = 1.45f;
    env->dr_mass_max = 1.80f;
    env->dr_inertia_min = 2.00f;
    env->dr_inertia_max = 3.20f;
    env->dr_motor_thrust_min = 0.95f;
    env->dr_motor_thrust_max = 1.05f;
    env->dr_motor_tau_min = 0.16f;
    env->dr_motor_tau_max = 0.24f;
    env->dr_com_xy = 0.012f;
    env->dr_com_z = 0.008f;
    env->reset_pos_scale = 0.20f;
    env->reset_vel_max = 0.1f;
}

static void configure_dr_family_v1a_ultra_large_low_authority(DroneEnv* env, int num_agents) {
    configure_dr_family_v1a(env, num_agents);

    env->dr_usable_t2w_min = 1.55f;
    env->dr_usable_t2w_max = 1.95f;
    env->dr_mass_min = 1.45f;
    env->dr_mass_max = 1.80f;
    env->dr_inertia_min = 2.00f;
    env->dr_inertia_max = 3.20f;
    env->dr_motor_thrust_min = 0.85f;
    env->dr_motor_thrust_max = 0.98f;
    env->dr_motor_tau_min = 0.20f;
    env->dr_motor_tau_max = 0.28f;
    env->dr_com_xy = 0.018f;
    env->dr_com_z = 0.010f;
    env->reset_pos_scale = 0.20f;
    env->reset_vel_max = 0.1f;
}

static void configure_dr_family_v1a2_mix(DroneEnv* env, int num_agents) {
    configure_dr_family_v1a(env, num_agents);
    env->dr_profile_mix = 1.0f;
}

static void configure_dr_edgefix_capped_mix(DroneEnv* env, int num_agents) {
    configure_dr_family_v1a(env, num_agents);
    env->dr_profile_mix = 2.0f;
}

static void configure_low_authority_holdout(DroneEnv* env, int num_agents) {
    configure_dr_family_v05(env, num_agents);

    env->dr_usable_t2w_min = 1.50f;
    env->dr_usable_t2w_max = 1.80f;
}

static void configure_edge_low_authority(DroneEnv* env, int num_agents) {
    configure_dr_family_v05(env, num_agents);

    env->dr_usable_t2w_min = 2.00f;
    env->dr_usable_t2w_max = 2.15f;
    env->dr_motor_tau_min = 0.06f;
    env->dr_motor_tau_max = 0.20f;
}

static void configure_motor_tau_high_holdout(DroneEnv* env, int num_agents) {
    configure_dr_family_v05(env, num_agents);

    env->dr_motor_tau_min = 0.20f;
    env->dr_motor_tau_max = 0.35f;
}

static void configure_edge_high_tau(DroneEnv* env, int num_agents) {
    configure_dr_family_v05(env, num_agents);

    env->dr_usable_t2w_min = 2.20f;
    env->dr_usable_t2w_max = 3.80f;
    env->dr_motor_tau_min = 0.22f;
    env->dr_motor_tau_max = 0.24f;
}

static void configure_mass_high_holdout(DroneEnv* env, int num_agents) {
    configure_dr_family_v05(env, num_agents);

    env->dr_mass_min = 1.15f;
    env->dr_mass_max = 1.35f;
}

static void recompute_hover_rpms(Drone* agent) {
    float trim[4];
    hover_trim_thrusts(&agent->params, trim);
    for (int i = 0; i < 4; i++) {
        agent->state.rpms[i] = thrust_to_rpm_i(&agent->params, i, trim[i]);
    }
}

static void apply_fixed_motor_profile(Drone* agent, const float scales[4], float k_thrust_mult,
                                      float normalized_thrust_max) {
    Params* p = &agent->params;
    p->action_mode = M4D_ACTION_NORMALIZED_THRUST;
    p->action_scale = 1.0f;
    p->normalized_thrust_min = 0.0f;
    p->normalized_thrust_max = normalized_thrust_max;
    p->k_thrust = BASE_K_THRUST * k_thrust_mult;
    p->k_thrust_mult = k_thrust_mult;
    p->mass_mult = p->mass / BASE_MASS;
    for (int i = 0; i < 4; i++) {
        p->motor_thrust_scale[i] = scales[i];
    }
    recompute_hover_rpms(agent);
}

static void apply_small_airframe_profile(Drone* agent) {
    Params* p = &agent->params;
    const float mass_mult = 0.65f;
    const float inertia_mult = 0.55f;
    const float arm_mult = 0.78f;

    p->mass = BASE_MASS * mass_mult;
    p->mass_mult = mass_mult;
    p->ixx = BASE_IXX * inertia_mult;
    p->iyy = BASE_IYY * inertia_mult;
    p->izz = BASE_IZZ * inertia_mult;
    p->ixx_mult = inertia_mult;
    p->iyy_mult = inertia_mult;
    p->izz_mult = inertia_mult;
    p->arm_len = BASE_ARM_LEN * arm_mult;
    for (int i = 0; i < 4; i++) {
        p->motor_x[i] *= arm_mult;
        p->motor_y[i] *= arm_mult;
    }
    recompute_hover_rpms(agent);
}

static void apply_eval_profile_after_reset(DroneEnv* env, Drone* agent, const char* config) {
    (void)env;
    if (strcmp(config, "mixed_motors_mild") == 0) {
        const float scales[4] = {1.00f, 0.92f, 0.98f, 1.05f};
        apply_fixed_motor_profile(agent, scales, 1.0f, 0.85f);
    } else if (strcmp(config, "3plus1_mismatch") == 0 ||
               strcmp(config, "three_plus_one_mismatch") == 0) {
        const float scales[4] = {1.00f, 0.92f, 0.92f, 0.92f};
        apply_fixed_motor_profile(agent, scales, 1.0f, 0.85f);
    } else if (strcmp(config, "capped_high_thrust") == 0 ||
               strcmp(config, "mad_bsc_capped") == 0) {
        const float scales[4] = {1.0f, 1.0f, 1.0f, 1.0f};
        apply_fixed_motor_profile(agent, scales, 2.91f, 0.50f);
    } else if (strcmp(config, "hard_small") == 0) {
        apply_small_airframe_profile(agent);
    }
}

static void apply_eval_profile_all(DroneEnv* env, const char* config) {
    for (int i = 0; i < env->num_agents; i++) {
        apply_eval_profile_after_reset(env, &env->agents[i], config);
    }
    compute_observations(env);
}

static void configure_env(DroneEnv* env, const char* config, int num_agents) {
    if (strcmp(config, "baseline") == 0) {
        configure_baseline(env, num_agents);
    } else if (strcmp(config, "nominal") == 0) {
        configure_nominal_normalized(env, num_agents);
    } else if (strcmp(config, "light") == 0) {
        configure_dr_light(env, num_agents);
    } else if (strcmp(config, "narrow20") == 0) {
        configure_dr_narrow20(env, num_agents);
    } else if (strcmp(config, "medium") == 0) {
        configure_dr_medium(env, num_agents);
    } else if (strcmp(config, "hard") == 0) {
        configure_dr_hard(env, num_agents);
    } else if (strcmp(config, "hard_small") == 0) {
        configure_dr_hard_small(env, num_agents);
    } else if (strcmp(config, "family_v1") == 0 ||
               strcmp(config, "family_v1_holdout_raw") == 0) {
        configure_dr_family_v1(env, num_agents);
    } else if (strcmp(config, "family_v05") == 0 ||
               strcmp(config, "family_v0.5") == 0 ||
               strcmp(config, "family_v0.5_authority_gated") == 0) {
        configure_dr_family_v05(env, num_agents);
    } else if (strcmp(config, "family_v1a") == 0 ||
               strcmp(config, "family_v1a_authority_gated") == 0) {
        configure_dr_family_v1a(env, num_agents);
    } else if (strcmp(config, "family_v1a_super_large") == 0) {
        configure_dr_family_v1a_super_large(env, num_agents);
    } else if (strcmp(config, "family_v1a_ultra_large") == 0) {
        configure_dr_family_v1a_ultra_large(env, num_agents);
    } else if (strcmp(config, "family_v1a_ultra_large_low_authority") == 0) {
        configure_dr_family_v1a_ultra_large_low_authority(env, num_agents);
    } else if (strcmp(config, "family_v1a2_mix") == 0 ||
               strcmp(config, "family_v1a.2_mix") == 0 ||
               strcmp(config, "v1a2_mix") == 0 ||
               strcmp(config, "v1a.2_mix") == 0) {
        configure_dr_family_v1a2_mix(env, num_agents);
    } else if (strcmp(config, "edgefix_capped_mix") == 0 ||
               strcmp(config, "v1_edgefix_capped_mix") == 0) {
        configure_dr_edgefix_capped_mix(env, num_agents);
    } else if (strcmp(config, "low_authority_holdout") == 0) {
        configure_low_authority_holdout(env, num_agents);
    } else if (strcmp(config, "edge_low_authority") == 0) {
        configure_edge_low_authority(env, num_agents);
    } else if (strcmp(config, "motor_tau_high_holdout") == 0) {
        configure_motor_tau_high_holdout(env, num_agents);
    } else if (strcmp(config, "edge_high_tau") == 0) {
        configure_edge_high_tau(env, num_agents);
    } else if (strcmp(config, "mass_high_holdout") == 0) {
        configure_mass_high_holdout(env, num_agents);
    } else if (strcmp(config, "mixed_motors_mild") == 0 ||
               strcmp(config, "3plus1_mismatch") == 0 ||
               strcmp(config, "three_plus_one_mismatch") == 0 ||
               strcmp(config, "capped_high_thrust") == 0 ||
               strcmp(config, "mad_bsc_capped") == 0) {
        configure_nominal_normalized(env, num_agents);
    } else {
        fprintf(stderr,
                "Unknown config '%s'; valid: baseline, nominal, light, narrow20, medium, "
                "hard, hard_small, family_v1, family_v1_holdout_raw, family_v05, "
                "family_v0.5_authority_gated, family_v1a, "
                "family_v1a_authority_gated, family_v1a_super_large, "
                "family_v1a_ultra_large, family_v1a_ultra_large_low_authority, "
                "v1a.2_mix, edgefix_capped_mix, "
                "edge_low_authority, edge_high_tau, low_authority_holdout, motor_tau_high_holdout, "
                "mass_high_holdout, mixed_motors_mild, "
                "3plus1_mismatch, capped_high_thrust\n",
                config);
        exit(2);
    }
}

static void count_nonfinite_array(const float* values, size_t n, long* count) {
    for (size_t i = 0; i < n; i++) {
        if (!isfinite(values[i])) {
            *count += 1;
        }
    }
}

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr,
                "Usage: %s WEIGHTS.bin [episodes] "
                "[baseline|nominal|light|narrow20|medium|hard|family_v1|family_v05|holdout] [action_scale] "
                "[num_agents]\n",
                argv[0]);
        return 2;
    }

    const char* weights_path = argv[1];
    int target_episodes = argc >= 3 ? atoi(argv[2]) : 20;
    const char* config = argc >= 4 ? argv[3] : "baseline";
    int has_action_scale = argc >= 5;
    float action_scale = has_action_scale ? (float)atof(argv[4]) : 0.0f;
    int num_agents = argc >= 6 ? atoi(argv[5]) : 64;
    if (target_episodes <= 0) {
        target_episodes = 20;
    }
    if (num_agents <= 0) {
        num_agents = 64;
    }

    DroneEnv* env = (DroneEnv*)calloc(1, sizeof(DroneEnv));
    configure_env(env, config, num_agents);
    if (has_action_scale) {
        env->action_scale = action_scale;
    }
    env->action_mode = env_action_mode("M4D_ACTION_MODE", env->action_mode);
    env->normalized_thrust_min =
        env_float("M4D_NORMALIZED_THRUST_MIN", env->normalized_thrust_min);
    env->normalized_thrust_max =
        env_float("M4D_NORMALIZED_THRUST_MAX", env->normalized_thrust_max);
    int sample_actions = getenv("M4D_SAMPLE_ACTIONS") != NULL;
    unsigned int policy_seed =
        getenv("M4D_POLICY_SEED") ? (unsigned int)atoi(getenv("M4D_POLICY_SEED")) : 0u;
    int trace_steps = getenv("M4D_TRACE_STEPS") ? atoi(getenv("M4D_TRACE_STEPS")) : 0;
    int reset_state_interval = getenv("M4D_RESET_STATE_INTERVAL")
                                   ? atoi(getenv("M4D_RESET_STATE_INTERVAL"))
                                   : M4D_DEPLOY_DEFAULT_RESET_INTERVAL;
    float slew_da_max = env_float("M4D_SLEW_DA_MAX", 0.0f);
    env->rng = getenv("M4D_ENV_SEED") ? (unsigned int)atoi(getenv("M4D_ENV_SEED")) : 42u;
    srand(policy_seed);
    if (sample_actions) {
        fprintf(stderr,
                "M4D_SAMPLE_ACTIONS is ignored by the deployment runtime; deterministic mean "
                "actions are used.\n");
    }
    const char* episode_csv_path = getenv("M4D_EPISODE_CSV");
    FILE* episode_csv = NULL;
    if (episode_csv_path != NULL && episode_csv_path[0] != '\0') {
        episode_csv = fopen(episode_csv_path, "w");
        if (episode_csv == NULL) {
            perror("Failed to open M4D_EPISODE_CSV");
            return 1;
        }
        fprintf(episode_csv,
                "episode,agent,terminal,episode_length,episode_return,mass_scale,"
                "i_scale_mean,i_scale_min,i_scale_max,effective_t2w,"
                "usable_t2w_true,hover_fraction,thrust_margin,"
                "min_motor_scale,mean_motor_scale,max_motor_scale,max_motor_tau,"
                "motor_thrust_scale_min,motor_thrust_scale_max,motor_tau_max,"
                "com_offset_norm,obs_delay,act_delay,battery_sag,sensor_noise,action_cap,"
                "thrust_limit_scale,mean_abs_action,action_saturation,"
                "mean_abs_delta_action,episode_action_saturation,"
                "episode_mean_abs_delta_action,episode_reset_jump,max_dist,max_omega\n");
    }

    printf("config=%s action_scale=%.6f action_mode=%d normalized_thrust_min=%.6f "
           "normalized_thrust_max=%.6f num_agents=%d deterministic=%d sample_actions=%d "
           "policy_seed=%u env_seed=%u trace_steps=%d reset_state_interval=%d "
           "slew_da_max=%.6f\n",
           config, env->action_scale, env->action_mode, env->normalized_thrust_min,
           env->normalized_thrust_max, env->num_agents, 1, 0, policy_seed, env->rng, trace_steps,
           reset_state_interval, slew_da_max);

    const size_t obs_size = 23;
    env->observations = (float*)calloc(env->num_agents * obs_size, sizeof(float));
    env->actions = (float*)calloc(env->num_agents * 4, sizeof(float));
    env->rewards = (float*)calloc(env->num_agents, sizeof(float));
    env->terminals = (float*)calloc(env->num_agents, sizeof(float));

    M4DDeploymentRuntime policy;
    if (m4d_deploy_init(&policy, weights_path, env->num_agents, reset_state_interval,
                        env->action_scale) != 0) {
        return 1;
    }

    init(env);
    c_reset(env);

    float* episode_returns = (float*)calloc(env->num_agents, sizeof(float));
    int* episode_lengths = (int*)calloc(env->num_agents, sizeof(int));
    int print_episodes = getenv("M4D_PRINT_EPISODES") != NULL;
    int completed = 0;
    int oob_count = 0;
    int timeout_count = 0;
    long nan_inf_count = 0;
    double sum_return = 0.0;
    double sum_sq_return = 0.0;
    double sum_len = 0.0;
    double action_delta_sum = 0.0;
    double reset_jump_sum = 0.0;
    double post_slew_reset_jump_sum = 0.0;
    double post_lag_motor_jump_sum = 0.0;
    float reset_jump_max = 0.0f;
    long action_delta_count = 0;
    long reset_jump_count = 0;
    long post_slew_reset_jump_count = 0;
    long post_lag_motor_jump_count = 0;
    float min_return = 0.0f;
    float max_return = 0.0f;
    float* prev_actions = (float*)calloc((size_t)env->num_agents * 4, sizeof(float));
    unsigned char* has_prev_action = (unsigned char*)calloc(env->num_agents, sizeof(unsigned char));
    float* continued_actions = (float*)calloc((size_t)env->num_agents * 4, sizeof(float));
    float* continued_limited_actions =
        (float*)calloc((size_t)env->num_agents * 4, sizeof(float));
    float* state_backup = (float*)calloc((size_t)M4D_DEPLOY_NUM_LAYERS * env->num_agents *
                                             M4D_DEPLOY_HIDDEN_SIZE,
                                         sizeof(float));
    float* ep_action_abs_sum = (float*)calloc(env->num_agents, sizeof(float));
    float* ep_action_saturation_sum = (float*)calloc(env->num_agents, sizeof(float));
    float* ep_action_delta_sum = (float*)calloc(env->num_agents, sizeof(float));
    int* ep_action_delta_count = (int*)calloc(env->num_agents, sizeof(int));
    float* ep_reset_jump_sum = (float*)calloc(env->num_agents, sizeof(float));
    int* ep_reset_jump_count = (int*)calloc(env->num_agents, sizeof(int));
    float* ep_max_dist = (float*)calloc(env->num_agents, sizeof(float));
    float* ep_max_omega = (float*)calloc(env->num_agents, sizeof(float));
    float* snap_mass_scale = (float*)calloc(env->num_agents, sizeof(float));
    float* snap_i_scale_mean = (float*)calloc(env->num_agents, sizeof(float));
    float* snap_i_scale_min = (float*)calloc(env->num_agents, sizeof(float));
    float* snap_i_scale_max = (float*)calloc(env->num_agents, sizeof(float));
    float* snap_effective_t2w = (float*)calloc(env->num_agents, sizeof(float));
    float* snap_usable_t2w_true = (float*)calloc(env->num_agents, sizeof(float));
    float* snap_hover_fraction = (float*)calloc(env->num_agents, sizeof(float));
    float* snap_thrust_margin = (float*)calloc(env->num_agents, sizeof(float));
    float* snap_min_motor_scale = (float*)calloc(env->num_agents, sizeof(float));
    float* snap_mean_motor_scale = (float*)calloc(env->num_agents, sizeof(float));
    float* snap_max_motor_scale = (float*)calloc(env->num_agents, sizeof(float));
    float* snap_max_motor_tau = (float*)calloc(env->num_agents, sizeof(float));
    float* snap_motor_tau_max = (float*)calloc(env->num_agents, sizeof(float));
    float* snap_com_offset_norm = (float*)calloc(env->num_agents, sizeof(float));
    float* snap_action_cap = (float*)calloc(env->num_agents, sizeof(float));
    FloatVec dist_samples = {0};
    FloatVec omega_samples = {0};
    FloatVec action_delta_samples = {0};
    FloatVec reset_jump_samples = {0};
    FloatVec post_slew_reset_jump_samples = {0};
    FloatVec post_lag_motor_jump_samples = {0};
    int step = 0;

    apply_eval_profile_all(env, config);

    while (completed < target_episodes) {
        count_nonfinite_array(env->observations, (size_t)env->num_agents * obs_size,
                              &nan_inf_count);
        for (int i = 0; i < env->num_agents; i++) {
            Drone* agent = &env->agents[i];
            float dist = norm3(sub3(agent->target->pos, agent->state.pos));
            float omega = norm3(agent->state.omega);
            float_vec_push(&dist_samples, dist);
            float_vec_push(&omega_samples, omega);
            if (dist > ep_max_dist[i]) ep_max_dist[i] = dist;
            if (omega > ep_max_omega[i]) ep_max_omega[i] = omega;

            Params* p = &agent->params;
            float i_min = fminf(p->ixx_mult, fminf(p->iyy_mult, p->izz_mult));
            float i_max = fmaxf(p->ixx_mult, fmaxf(p->iyy_mult, p->izz_mult));
            float max_cmd = clampf(p->normalized_thrust_max, p->normalized_thrust_min, 1.0f);
            snap_mass_scale[i] = p->mass_mult;
            snap_i_scale_mean[i] = (p->ixx_mult + p->iyy_mult + p->izz_mult) / 3.0f;
            snap_i_scale_min[i] = i_min;
            snap_i_scale_max[i] = i_max;
            snap_effective_t2w[i] =
                total_max_motor_thrust(p) / fmaxf(p->mass * p->gravity, 1e-6f);
            snap_usable_t2w_true[i] = max_cmd * snap_effective_t2w[i];
            snap_hover_fraction[i] =
                snap_usable_t2w_true[i] > 1e-6f ? 1.0f / snap_usable_t2w_true[i] : 0.0f;
            snap_thrust_margin[i] = snap_usable_t2w_true[i] - 1.0f;
            float min_motor_scale = p->motor_thrust_scale[0];
            float max_motor_scale = p->motor_thrust_scale[0];
            float sum_motor_scale = 0.0f;
            float max_motor_tau = p->motor_tau[0];
            for (int m = 0; m < 4; m++) {
                if (p->motor_thrust_scale[m] < min_motor_scale) {
                    min_motor_scale = p->motor_thrust_scale[m];
                }
                if (p->motor_thrust_scale[m] > max_motor_scale) {
                    max_motor_scale = p->motor_thrust_scale[m];
                }
                if (p->motor_tau[m] > max_motor_tau) {
                    max_motor_tau = p->motor_tau[m];
                }
                sum_motor_scale += p->motor_thrust_scale[m];
            }
            snap_min_motor_scale[i] = min_motor_scale;
            snap_mean_motor_scale[i] = sum_motor_scale * 0.25f;
            snap_max_motor_scale[i] = max_motor_scale;
            snap_max_motor_tau[i] = max_motor_tau;
            snap_motor_tau_max[i] = max_motor_tau;
            snap_com_offset_norm[i] = sqrtf(p->com_x * p->com_x + p->com_y * p->com_y
                                            + p->com_z * p->com_z);
            snap_action_cap[i] = max_cmd;
        }

        if (reset_state_interval > 0 && policy.step > 0 &&
            policy.step % reset_state_interval == 0) {
            size_t state_count = (size_t)M4D_DEPLOY_NUM_LAYERS * env->num_agents *
                                 M4D_DEPLOY_HIDDEN_SIZE;
            memcpy(state_backup, policy.state, state_count * sizeof(float));
            m4d_deploy_forward_no_reset(&policy, env->observations, continued_actions);
            memcpy(policy.state, state_backup, state_count * sizeof(float));
            m4d_deploy_reset_state(&policy);
            m4d_deploy_forward_no_reset(&policy, env->observations, env->actions);
            policy.step += 1;

            for (int i = 0; i < env->num_agents; i++) {
                float* reset_action = &env->actions[4 * i];
                float* continued_action = &continued_actions[4 * i];
                float* continued_limited_action = &continued_limited_actions[4 * i];
                float* prev_action = &prev_actions[4 * i];

                float jump = mean_abs_delta4(reset_action, continued_action);
                for (int a = 0; a < 4; a++) {
                    continued_limited_action[a] = slew_limited_value(
                        continued_action[a], prev_action[a], has_prev_action[i], slew_da_max);
                    reset_action[a] = slew_limited_value(
                        reset_action[a], prev_action[a], has_prev_action[i], slew_da_max);
                }
                float post_slew_jump = mean_abs_delta4(reset_action, continued_limited_action);
                float post_lag_motor_jump = post_lag_motor_jump_for_agent(
                    &env->agents[i], reset_action, continued_limited_action);

                reset_jump_sum += jump;
                if (jump > reset_jump_max) reset_jump_max = jump;
                reset_jump_count += 1;
                float_vec_push(&reset_jump_samples, jump);
                post_slew_reset_jump_sum += post_slew_jump;
                post_slew_reset_jump_count += 1;
                float_vec_push(&post_slew_reset_jump_samples, post_slew_jump);
                post_lag_motor_jump_sum += post_lag_motor_jump;
                post_lag_motor_jump_count += 1;
                float_vec_push(&post_lag_motor_jump_samples, post_lag_motor_jump);
                ep_reset_jump_sum[i] += jump;
                ep_reset_jump_count[i] += 1;
            }
        } else {
            m4d_deploy_forward(&policy, env->observations, env->actions);
            for (int i = 0; i < env->num_agents; i++) {
                apply_slew_limiter_for_agent(&env->actions[4 * i], &prev_actions[4 * i],
                                             has_prev_action[i], slew_da_max);
            }
        }

        count_nonfinite_array(env->actions, (size_t)env->num_agents * 4, &nan_inf_count);
        for (int i = 0; i < env->num_agents; i++) {
            float action_abs = 0.0f;
            float saturation = 0.0f;
            for (int a = 0; a < 4; a++) {
                float abs_action = fabsf(env->actions[4 * i + a]);
                action_abs += abs_action;
                if (abs_action >= 0.99f) saturation += 1.0f;
            }
            ep_action_abs_sum[i] += action_abs * 0.25f;
            ep_action_saturation_sum[i] += saturation * 0.25f;

            if (has_prev_action[i]) {
                float delta = 0.0f;
                for (int a = 0; a < 4; a++) {
                    delta += fabsf(env->actions[4 * i + a] - prev_actions[4 * i + a]);
                }
                float mean_delta = 0.25f * delta;
                action_delta_sum += mean_delta;
                action_delta_count += 1;
                float_vec_push(&action_delta_samples, mean_delta);
                ep_action_delta_sum[i] += mean_delta;
                ep_action_delta_count[i] += 1;
            }
            for (int a = 0; a < 4; a++) {
                prev_actions[4 * i + a] = env->actions[4 * i + a];
            }
            has_prev_action[i] = 1;
        }

        if (step < trace_steps) {
            Drone* agent = &env->agents[0];
            float obs_min = env->observations[0];
            float obs_max = env->observations[0];
            for (int j = 1; j < 23; j++) {
                if (env->observations[j] < obs_min) {
                    obs_min = env->observations[j];
                }
                if (env->observations[j] > obs_max) {
                    obs_max = env->observations[j];
                }
            }
            Vec3 to_target = sub3(agent->target->pos, agent->state.pos);
            float dist = norm3(to_target);
            printf("trace_pre step=%d obs_min=%.6f obs_max=%.6f "
                   "action=[%.6f %.6f %.6f %.6f] pos=[%.6f %.6f %.6f] "
                   "target=[%.6f %.6f %.6f] dist=%.6f vel=[%.6f %.6f %.6f] "
                   "omega=[%.6f %.6f %.6f] rpm=[%.3f %.3f %.3f %.3f]\n",
                   step, obs_min, obs_max, env->actions[0], env->actions[1], env->actions[2],
                   env->actions[3], agent->state.pos.x, agent->state.pos.y,
                   agent->state.pos.z, agent->target->pos.x, agent->target->pos.y,
                   agent->target->pos.z, dist, agent->state.vel.x, agent->state.vel.y,
                   agent->state.vel.z, agent->state.omega.x, agent->state.omega.y,
                   agent->state.omega.z, agent->state.rpms[0], agent->state.rpms[1],
                   agent->state.rpms[2], agent->state.rpms[3]);
        }

        c_step(env);
        count_nonfinite_array(env->rewards, env->num_agents, &nan_inf_count);

        if (step < trace_steps) {
            Drone* agent = &env->agents[0];
            Vec3 to_target = sub3(agent->target->pos, agent->state.pos);
            float dist = norm3(to_target);
            printf("trace_post step=%d reward=%.6f terminal=%.0f pos=[%.6f %.6f %.6f] "
                   "dist=%.6f vel=[%.6f %.6f %.6f] omega=[%.6f %.6f %.6f] "
                   "rpm=[%.3f %.3f %.3f %.3f]\n",
                   step, env->rewards[0], env->terminals[0], agent->state.pos.x,
                   agent->state.pos.y, agent->state.pos.z, dist, agent->state.vel.x,
                   agent->state.vel.y, agent->state.vel.z, agent->state.omega.x,
                   agent->state.omega.y, agent->state.omega.z, agent->state.rpms[0],
                   agent->state.rpms[1], agent->state.rpms[2], agent->state.rpms[3]);
        }
        step += 1;

        int reset_profile_changed = 0;
        for (int i = 0; i < env->num_agents; i++) {
            episode_returns[i] += env->rewards[i];
            episode_lengths[i] += 1;

            if (env->terminals[i]) {
                int timeout = episode_lengths[i] >= HORIZON;
                timeout_count += timeout;
                oob_count += !timeout;
                sum_return += episode_returns[i];
                sum_sq_return += (double)episode_returns[i] * (double)episode_returns[i];
                sum_len += episode_lengths[i];
                if (completed == 0 || episode_returns[i] < min_return) {
                    min_return = episode_returns[i];
                }
                if (completed == 0 || episode_returns[i] > max_return) {
                    max_return = episode_returns[i];
                }

                if (print_episodes) {
                    printf("episode=%d agent=%d len=%d return=%.6f terminal=%s\n",
                           completed + 1, i, episode_lengths[i], episode_returns[i],
                           timeout ? "timeout" : "oob");
                }

                completed += 1;
                if (episode_csv != NULL) {
                    float length = fmaxf((float)episode_lengths[i], 1.0f);
                    float mean_abs_action_ep = ep_action_abs_sum[i] / length;
                    float saturation_ep = ep_action_saturation_sum[i] / length;
                    float mean_delta_ep = ep_action_delta_count[i] > 0
                        ? ep_action_delta_sum[i] / (float)ep_action_delta_count[i]
                        : 0.0f;
                    float reset_jump_ep = ep_reset_jump_count[i] > 0
                        ? ep_reset_jump_sum[i] / (float)ep_reset_jump_count[i]
                        : 0.0f;
                    fprintf(episode_csv,
                            "%d,%d,%s,%d,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,"
                            "%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,"
                            "%.9g,%.9g,%.9g,%s,%.9g,%s,%.9g,%.9g,%.9g,%.9g,"
                            "%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g\n",
                            completed, i, timeout ? "timeout" : "oob", episode_lengths[i],
                            episode_returns[i], snap_mass_scale[i], snap_i_scale_mean[i],
                            snap_i_scale_min[i], snap_i_scale_max[i], snap_effective_t2w[i],
                            snap_usable_t2w_true[i], snap_hover_fraction[i],
                            snap_thrust_margin[i], snap_min_motor_scale[i],
                            snap_mean_motor_scale[i], snap_max_motor_scale[i],
                            snap_max_motor_tau[i], snap_min_motor_scale[i],
                            snap_max_motor_scale[i], snap_motor_tau_max[i], snap_com_offset_norm[i],
                            "", env->action_latency, "", env->sensor_noise, snap_action_cap[i],
                            snap_action_cap[i], mean_abs_action_ep, saturation_ep,
                            mean_delta_ep, saturation_ep, mean_delta_ep, reset_jump_ep,
                            ep_max_dist[i], ep_max_omega[i]);
                }
                episode_returns[i] = 0.0f;
                episode_lengths[i] = 0;
                has_prev_action[i] = 0;
                ep_action_abs_sum[i] = 0.0f;
                ep_action_saturation_sum[i] = 0.0f;
                ep_action_delta_sum[i] = 0.0f;
                ep_action_delta_count[i] = 0;
                ep_reset_jump_sum[i] = 0.0f;
                ep_reset_jump_count[i] = 0;
                ep_max_dist[i] = 0.0f;
                ep_max_omega[i] = 0.0f;
                apply_eval_profile_after_reset(env, &env->agents[i], config);
                reset_profile_changed = 1;

                if (completed >= target_episodes) {
                    break;
                }
            }
        }
        if (reset_profile_changed) {
            compute_observations(env);
        }
    }

    double mean_return = sum_return / (double)completed;
    double variance = sum_sq_return / (double)completed - mean_return * mean_return;
    if (variance < 0.0) {
        variance = 0.0;
    }
    double std_return = sqrt(variance);
    double timeout_rate = (double)timeout_count / (double)completed;
    double oob_rate = (double)oob_count / (double)completed;
    double mean_len = sum_len / (double)completed;
    float p95_dist = percentile(&dist_samples, 0.95f);
    float p99_dist = percentile(&dist_samples, 0.99f);
    float p95_omega = percentile(&omega_samples, 0.95f);
    float p99_omega = percentile(&omega_samples, 0.99f);
    float p95_action_delta = percentile(&action_delta_samples, 0.95f);
    float p95_reset_jump = percentile(&reset_jump_samples, 0.95f);
    float p95_post_slew_reset_jump = percentile(&post_slew_reset_jump_samples, 0.95f);
    float p95_post_lag_motor_jump = percentile(&post_lag_motor_jump_samples, 0.95f);
    double mean_abs_delta_action =
        action_delta_count > 0 ? action_delta_sum / (double)action_delta_count : 0.0;
    double reset32_action_jump =
        reset_jump_count > 0 ? reset_jump_sum / (double)reset_jump_count : 0.0;
    double post_slew_reset_jump =
        post_slew_reset_jump_count > 0
            ? post_slew_reset_jump_sum / (double)post_slew_reset_jump_count
            : 0.0;
    double post_lag_motor_jump =
        post_lag_motor_jump_count > 0
            ? post_lag_motor_jump_sum / (double)post_lag_motor_jump_count
            : 0.0;

    printf("summary episodes=%d mean_return=%.6f std_return=%.6f min_return=%.6f "
           "max_return=%.6f timeout_rate=%.6f oob_rate=%.6f mean_len=%.6f\n",
           completed, mean_return, std_return, min_return, max_return, timeout_rate, oob_rate,
           mean_len);
    if (env->log.n > 0.0f) {
        float n = env->log.n;
        printf("env_log n=%.0f episode_return=%.6f score=%.6f perf=%.6f oob=%.6f "
               "timeout=%.6f ema_dist=%.6f ema_vel=%.6f ema_omega=%.6f "
               "ema_omega_z=%.6f mean_abs_action=%.6f action_saturation_frac=%.6f "
               "motor_clip_low_frac=%.6f motor_clip_high_frac=%.6f\n",
               n, env->log.episode_return / n, env->log.score / n, env->log.perf / n,
               env->log.oob / n, env->log.timeout / n, env->log.ema_dist / n,
               env->log.ema_vel / n, env->log.ema_omega / n, env->log.ema_omega_z / n,
               env->log.mean_abs_action / n, env->log.action_saturation_frac / n,
               env->log.motor_clip_low_frac / n, env->log.motor_clip_high_frac / n);
    }
    printf("smoothness mean_abs_delta_action=%.6f mean_abs_delta_action_p95=%.6f "
           "reset32_action_jump=%.6f reset32_action_jump_p95=%.6f "
           "reset32_action_jump_max=%.6f reset32_action_jump_count=%ld\n",
           mean_abs_delta_action, p95_action_delta, reset32_action_jump, p95_reset_jump,
           reset_jump_max, reset_jump_count);
    printf("slew pre_slew_reset_jump=%.6f pre_slew_reset_jump_p95=%.6f "
           "post_slew_reset_jump=%.6f post_slew_reset_jump_p95=%.6f "
           "post_lag_motor_jump=%.6f post_lag_motor_jump_p95=%.6f "
           "slew_da_max=%.6f\n",
           reset32_action_jump, p95_reset_jump, post_slew_reset_jump, p95_post_slew_reset_jump,
           post_lag_motor_jump, p95_post_lag_motor_jump, slew_da_max);
    printf("percentiles p95_dist=%.6f p99_dist=%.6f p95_omega=%.6f p99_omega=%.6f "
           "samples=%zu nan_inf_count=%ld\n",
           p95_dist, p99_dist, p95_omega, p99_omega, dist_samples.len, nan_inf_count);
    printf("csv,%s,%.6f,%d,%d,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n", config,
           env->action_scale, env->num_agents, completed, mean_return, std_return, min_return,
           max_return, timeout_rate, oob_rate, mean_len);
    printf("csv_extended,%s,%.6f,%d,%d,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,"
           "%.6f,%.6f,%.6f,%.6f,%ld\n",
           config, env->action_scale, env->num_agents, completed, oob_rate, timeout_rate,
           env->log.n > 0.0f ? env->log.ema_dist / env->log.n : 0.0f,
           env->log.n > 0.0f ? env->log.ema_vel / env->log.n : 0.0f,
           env->log.n > 0.0f ? env->log.ema_omega_z / env->log.n : 0.0f,
           env->log.n > 0.0f ? env->log.mean_abs_action / env->log.n : 0.0f,
           env->log.n > 0.0f ? env->log.action_saturation_frac / env->log.n : 0.0f,
           mean_abs_delta_action, p95_dist, p99_dist, p95_omega, p99_omega, nan_inf_count);
    printf("csv_clean,%s,%.6f,%d,%d,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,"
           "%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%ld\n",
           config, env->action_scale, env->num_agents, completed, oob_rate, timeout_rate,
           env->log.n > 0.0f ? env->log.ema_dist / env->log.n : 0.0f,
           env->log.n > 0.0f ? env->log.ema_vel / env->log.n : 0.0f,
           env->log.n > 0.0f ? env->log.ema_omega_z / env->log.n : 0.0f,
           env->log.n > 0.0f ? env->log.mean_abs_action / env->log.n : 0.0f,
           env->log.n > 0.0f ? env->log.action_saturation_frac / env->log.n : 0.0f,
           mean_abs_delta_action, p95_action_delta, reset32_action_jump, p95_reset_jump,
           p95_dist, p99_dist, p95_omega, p99_omega, nan_inf_count);
    printf("csv_slew,%s,%.6f,%d,%d,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,"
           "%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%ld,%.6f,%.6f,%.6f,"
           "%.6f,%.6f,%.6f,%.6f\n",
           config, env->action_scale, env->num_agents, completed, oob_rate, timeout_rate,
           env->log.n > 0.0f ? env->log.ema_dist / env->log.n : 0.0f,
           env->log.n > 0.0f ? env->log.ema_vel / env->log.n : 0.0f,
           env->log.n > 0.0f ? env->log.ema_omega_z / env->log.n : 0.0f,
           env->log.n > 0.0f ? env->log.mean_abs_action / env->log.n : 0.0f,
           env->log.n > 0.0f ? env->log.action_saturation_frac / env->log.n : 0.0f,
           mean_abs_delta_action, p95_action_delta, reset32_action_jump, p95_reset_jump,
           p95_dist, p99_dist, p95_omega, p99_omega, nan_inf_count, slew_da_max,
           reset32_action_jump, p95_reset_jump, post_slew_reset_jump, p95_post_slew_reset_jump,
           post_lag_motor_jump, p95_post_lag_motor_jump);

    c_close(env);
    m4d_deploy_close(&policy);
    if (episode_csv != NULL) {
        fclose(episode_csv);
    }
    free(env->observations);
    free(env->actions);
    free(env->rewards);
    free(env->terminals);
    free(episode_returns);
    free(episode_lengths);
    free(prev_actions);
    free(has_prev_action);
    free(continued_actions);
    free(continued_limited_actions);
    free(state_backup);
    free(ep_action_abs_sum);
    free(ep_action_saturation_sum);
    free(ep_action_delta_sum);
    free(ep_action_delta_count);
    free(ep_reset_jump_sum);
    free(ep_reset_jump_count);
    free(ep_max_dist);
    free(ep_max_omega);
    free(snap_mass_scale);
    free(snap_i_scale_mean);
    free(snap_i_scale_min);
    free(snap_i_scale_max);
    free(snap_effective_t2w);
    free(snap_usable_t2w_true);
    free(snap_hover_fraction);
    free(snap_thrust_margin);
    free(snap_min_motor_scale);
    free(snap_mean_motor_scale);
    free(snap_max_motor_scale);
    free(snap_max_motor_tau);
    free(snap_motor_tau_max);
    free(snap_com_offset_norm);
    free(snap_action_cap);
    free(dist_samples.data);
    free(omega_samples.data);
    free(action_delta_samples.data);
    free(reset_jump_samples.data);
    free(post_slew_reset_jump_samples.data);
    free(post_lag_motor_jump_samples.data);
    free(env);
    return 0;
}
