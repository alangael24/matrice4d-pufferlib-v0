#include "drone.h"
#include "m4d_deployment_runtime.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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

static void configure_env(DroneEnv* env, const char* config, int num_agents) {
    if (strcmp(config, "baseline") == 0) {
        configure_baseline(env, num_agents);
    } else if (strcmp(config, "light") == 0) {
        configure_dr_light(env, num_agents);
    } else if (strcmp(config, "medium") == 0) {
        configure_dr_medium(env, num_agents);
    } else if (strcmp(config, "hard") == 0) {
        configure_dr_hard(env, num_agents);
    } else {
        fprintf(stderr, "Unknown config '%s'; valid: baseline, light, medium, hard\n", config);
        exit(2);
    }
}

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr,
                "Usage: %s WEIGHTS.bin [episodes] [baseline|light|medium|hard] [action_scale] "
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
    srand(policy_seed);
    if (sample_actions) {
        fprintf(stderr,
                "M4D_SAMPLE_ACTIONS is ignored by the deployment runtime; deterministic mean "
                "actions are used.\n");
    }

    printf("config=%s action_scale=%.6f action_mode=%d normalized_thrust_min=%.6f "
           "normalized_thrust_max=%.6f num_agents=%d deterministic=%d sample_actions=%d "
           "policy_seed=%u trace_steps=%d reset_state_interval=%d\n",
           config, env->action_scale, env->action_mode, env->normalized_thrust_min,
           env->normalized_thrust_max, env->num_agents, 1, 0, policy_seed, trace_steps,
           reset_state_interval);

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
    double sum_return = 0.0;
    double sum_sq_return = 0.0;
    double sum_len = 0.0;
    float min_return = 0.0f;
    float max_return = 0.0f;
    int step = 0;

    while (completed < target_episodes) {
        m4d_deploy_forward(&policy, env->observations, env->actions);

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
                episode_returns[i] = 0.0f;
                episode_lengths[i] = 0;

                if (completed >= target_episodes) {
                    break;
                }
            }
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
    printf("csv,%s,%.6f,%d,%d,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n", config,
           env->action_scale, env->num_agents, completed, mean_return, std_return, min_return,
           max_return, timeout_rate, oob_rate, mean_len);

    c_close(env);
    m4d_deploy_close(&policy);
    free(env->observations);
    free(env->actions);
    free(env->rewards);
    free(env->terminals);
    free(episode_returns);
    free(episode_lengths);
    free(env);
    return 0;
}
