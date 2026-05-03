#include "drone.h"
#include "puffernet.h"
#include "render.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

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

static void configure_env(DroneEnv* env, const char* config) {
    if (strcmp(config, "baseline") == 0) {
        configure_baseline(env);
    } else if (strcmp(config, "light") == 0) {
        configure_dr_light(env);
    } else if (strcmp(config, "medium") == 0) {
        configure_dr_medium(env);
    } else {
        fprintf(stderr, "Unknown config '%s'; valid: baseline, light, medium\n", config);
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
                "Usage: %s WEIGHTS.bin [frames] [baseline|light|medium] [action_scale]\\n",
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
    printf("config=%s action_scale=%.6f reset_state_interval=%d\n", config, env->action_scale,
           reset_state_interval);
    fflush(stdout);

    const size_t obs_size = 23;
    env->observations = (float*)calloc(env->num_agents * obs_size, sizeof(float));
    env->actions = (float*)calloc(env->num_agents * 4, sizeof(float));
    env->rewards = (float*)calloc(env->num_agents, sizeof(float));
    env->terminals = (float*)calloc(env->num_agents, sizeof(float));

    Weights* weights = load_weights(weights_path);
    if (weights == NULL) {
        return 1;
    }

    int logit_sizes[4] = {1, 1, 1, 1};
    PufferNet* net = make_puffernet(weights, env->num_agents, obs_size, 128, 3, logit_sizes, 4);

    init(env);
    c_reset(env);
    c_render(env);
    SetTargetFPS(60);

    int frame = 0;
    int completed_episodes = 0;
    float episode_returns[4] = {0};
    int episode_lengths[4] = {0};
    float last_returns[20] = {0};
    while (!WindowShouldClose()) {
        if (reset_state_interval > 0 && frame % reset_state_interval == 0) {
            memset(net->mingru->state, 0,
                   (size_t)net->mingru->num_layers * net->mingru->batch_size *
                       net->mingru->hidden_size * sizeof(float));
        }
        forward_puffernet(net, env->observations, env->actions);
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
            }
        }

        c_render(env);

        if (frames > 0 && ++frame >= frames) {
            break;
        }
    }

    c_close(env);
    free_puffernet(net);
    free(weights);
    free(env->observations);
    free(env->actions);
    free(env->rewards);
    free(env->terminals);
    free(env);
    return 0;
}
