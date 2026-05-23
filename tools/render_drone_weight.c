#include "drone.h"
#include "m4d_deployment_runtime.h"
#include "render.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static float render_env_float(const char* key, float fallback) {
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

static void configure_common(DroneEnv* env) {
    env->num_agents = 4;
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
}

static void configure_baseline(DroneEnv* env) {
    configure_common(env);

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

static void configure_nominal_normalized(DroneEnv* env) {
    configure_baseline(env);

    env->alpha_omega_z_mult = 5.0f;
    env->action_scale = 1.0f;
    env->action_mode = M4D_ACTION_NORMALIZED_THRUST;
    env->normalized_thrust_min = 0.0f;
    env->normalized_thrust_max = 0.85f;
    env->reset_yaw_range = 3.14159f;
    env->reset_vel_max = 0.2f;
}

static void configure_dr_light(DroneEnv* env) {
    configure_common(env);

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

static void configure_dr_medium(DroneEnv* env) {
    configure_common(env);

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

static void configure_dr_hard(DroneEnv* env) {
    configure_common(env);

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

static void configure_dr_hard_small(DroneEnv* env) {
    configure_dr_hard(env);

    env->num_agents = 4;
    env->reset_pos_scale = 0.75f;
    env->reset_vel_max = 0.15f;
}

static void configure_dr_family_v05(DroneEnv* env) {
    configure_common(env);

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

static void configure_dr_family_v1a(DroneEnv* env) {
    configure_common(env);

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

static void configure_dr_family_v1a_super_large(DroneEnv* env) {
    configure_dr_family_v1a(env);

    env->num_agents = 8;
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

static void configure_dr_family_v1a_ultra_large(DroneEnv* env) {
    configure_dr_family_v1a(env);

    env->num_agents = 8;
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

static void configure_dr_family_v1a_ultra_large_low_authority(DroneEnv* env) {
    configure_dr_family_v1a(env);

    env->num_agents = 8;
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

static void apply_render_profile_after_reset(DroneEnv* env, Drone* agent, const char* config) {
    (void)env;
    if (strcmp(config, "capped_high_thrust") == 0 ||
        strcmp(config, "mad_bsc_capped") == 0) {
        const float scales[4] = {1.0f, 1.0f, 1.0f, 1.0f};
        apply_fixed_motor_profile(agent, scales, 2.91f, 0.50f);
    } else if (strcmp(config, "hard_small") == 0) {
        apply_small_airframe_profile(agent);
    }
}

static void apply_render_profile_all(DroneEnv* env, const char* config) {
    for (int i = 0; i < env->num_agents; i++) {
        apply_render_profile_after_reset(env, &env->agents[i], config);
    }
    compute_observations(env);
}

static void configure_env(DroneEnv* env, const char* config) {
    if (strcmp(config, "baseline") == 0) {
        configure_baseline(env);
    } else if (strcmp(config, "light") == 0) {
        configure_dr_light(env);
    } else if (strcmp(config, "medium") == 0) {
        configure_dr_medium(env);
    } else if (strcmp(config, "hard") == 0) {
        configure_dr_hard(env);
    } else if (strcmp(config, "hard_small") == 0) {
        configure_dr_hard_small(env);
    } else if (strcmp(config, "family_v05") == 0 ||
               strcmp(config, "family_v0.5_authority_gated") == 0) {
        configure_dr_family_v05(env);
    } else if (strcmp(config, "family_v1a") == 0 ||
               strcmp(config, "family_v1a_authority_gated") == 0) {
        configure_dr_family_v1a(env);
    } else if (strcmp(config, "family_v1a_super_large") == 0) {
        configure_dr_family_v1a_super_large(env);
    } else if (strcmp(config, "family_v1a_ultra_large") == 0) {
        configure_dr_family_v1a_ultra_large(env);
    } else if (strcmp(config, "family_v1a_ultra_large_low_authority") == 0) {
        configure_dr_family_v1a_ultra_large_low_authority(env);
    } else if (strcmp(config, "capped_high_thrust") == 0 ||
               strcmp(config, "mad_bsc_capped") == 0) {
        configure_nominal_normalized(env);
    } else {
        fprintf(stderr,
                "Unknown config '%s'; valid: baseline, light, medium, hard, hard_small, family_v05, "
                "family_v0.5_authority_gated, family_v1a, family_v1a_authority_gated, "
                "family_v1a_super_large, family_v1a_ultra_large, "
                "family_v1a_ultra_large_low_authority, capped_high_thrust\n",
                config);
        exit(2);
    }
}

static float mean_last_returns(const float* returns, int count) {
    int n = count < 20 ? count : 20;
    if (n == 0) {
        return 0.0f;
    }

    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        sum += returns[i];
    }
    return sum / (float)n;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr,
                "Usage: %s WEIGHTS.bin [frames] [baseline|light|medium|hard] [action_scale]\\n",
                argv[0]);
        return 2;
    }

    const char* weights_path = argv[1];
    int frames = argc >= 3 ? atoi(argv[2]) : 0;
    const char* config = argc >= 4 ? argv[3] : "medium";
    int has_action_scale = argc >= 5;
    float action_scale = has_action_scale ? (float)atof(argv[4]) : 0.0f;
    int reset_state_interval = getenv("M4D_RESET_STATE_INTERVAL")
                                   ? atoi(getenv("M4D_RESET_STATE_INTERVAL"))
                                   : 32;

    srand(42);

    DroneEnv* env = (DroneEnv*)calloc(1, sizeof(DroneEnv));
    configure_env(env, config);
    if (has_action_scale) {
        env->action_scale = action_scale;
    }
    env->action_mode = env_action_mode("M4D_ACTION_MODE", env->action_mode);
    env->normalized_thrust_min =
        render_env_float("M4D_NORMALIZED_THRUST_MIN", env->normalized_thrust_min);
    env->normalized_thrust_max =
        render_env_float("M4D_NORMALIZED_THRUST_MAX", env->normalized_thrust_max);
    printf("config=%s action_scale=%.6f action_mode=%d normalized_thrust_min=%.6f "
           "normalized_thrust_max=%.6f reset_state_interval=%d\n",
           config, env->action_scale, env->action_mode, env->normalized_thrust_min,
           env->normalized_thrust_max, reset_state_interval);
    fflush(stdout);

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
    apply_render_profile_all(env, config);
    c_render(env);
    SetTargetFPS(60);

    int frame = 0;
    int completed_episodes = 0;
    float episode_returns[4] = {0};
    int episode_lengths[4] = {0};
    float last_returns[20] = {0};
    while (!WindowShouldClose()) {
        m4d_deploy_forward(&policy, env->observations, env->actions);
        c_step(env);

        for (int i = 0; i < env->num_agents; i++) {
            episode_returns[i] += env->rewards[i];
            episode_lengths[i] += 1;

            if (env->terminals[i]) {
                int slot = completed_episodes % 20;
                last_returns[slot] = episode_returns[i];
                completed_episodes += 1;

                printf("episode=%d agent=%d len=%d return=%.6f last20_mean=%.6f\n",
                       completed_episodes, i, episode_lengths[i], episode_returns[i],
                       mean_last_returns(last_returns,
                                         completed_episodes < 20 ? completed_episodes : 20));
                fflush(stdout);

                episode_returns[i] = 0.0f;
                episode_lengths[i] = 0;
                apply_render_profile_after_reset(env, &env->agents[i], config);
                compute_observations(env);
            }
        }

        c_render(env);

        if (frames > 0 && ++frame >= frames) {
            break;
        }
    }

    c_close(env);
    m4d_deploy_close(&policy);
    free(env->observations);
    free(env->actions);
    free(env->rewards);
    free(env->terminals);
    free(env);
    return 0;
}
