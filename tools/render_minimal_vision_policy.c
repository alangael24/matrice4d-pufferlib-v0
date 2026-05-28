#include "drone.h"

#define M4D_DEPLOY_OBS_SIZE DRONE_OBS_SIZE
#define M4D_DEPLOY_ALIGN_WEIGHTS 1
#include "m4d_deployment_runtime.h"
#include "render.h"

#include <stdio.h>
#include <stdlib.h>

static float getenv_float(const char* name, float fallback) {
    const char* value = getenv(name);
    return value != NULL ? strtof(value, NULL) : fallback;
}

static int getenv_int(const char* name, int fallback) {
    const char* value = getenv(name);
    return value != NULL ? atoi(value) : fallback;
}

static void configure_minimal_vision_baseline(DroneEnv* env) {
    env->num_agents = 1;
    env->max_rings = getenv_int("M4D_MAX_RINGS", 10);
    env->task = (DroneTask)getenv_int("M4D_TASK", HOVER);

    int race = env->task == RACE;
    env->alpha_dist = getenv_float("M4D_ALPHA_DIST", 1.0f);
    env->alpha_hover = getenv_float("M4D_ALPHA_HOVER", race ? 0.0f : 0.02f);
    env->alpha_shaping = getenv_float("M4D_ALPHA_SHAPING", race ? 0.0f : 1.0f);
    env->alpha_omega = 0.00135588f;
    env->alpha_omega_xy = 0.0015f;
    env->alpha_omega_z = 0.0015f;
    env->alpha_omega_z_sq = 0.0025f;
    env->alpha_omega_z_mult = 1.0f;
    env->alpha_action_delta = 0.002f;

    env->hover_target_dist = getenv_float("M4D_TARGET_DIST", race ? 6.0f : 5.0f);
    env->oob_radius = getenv_float("M4D_OOB_RADIUS", 18.0f);
    env->hover_dist = 0.1f;
    env->hover_omega = 0.1f;
    env->hover_vel = 0.1f;

    env->domain_randomization = 0.0f;
    env->action_scale = 1.0f;
    env->action_mode = M4D_ACTION_NORMALIZED_THRUST;
    env->normalized_thrust_min = 0.0f;
    env->normalized_thrust_max = 0.85f;
    env->reset_pos_scale = 0.20f;
    env->reset_yaw_range = 0.60f;
    env->reset_vel_max = 0.0f;
    env->action_latency = 0.0f;
    env->sensor_noise = 0.0f;

    env->minimal_vision_enabled = getenv_float("M4D_MINIMAL_VISION_ENABLED", 1.0f);
    env->minimal_vision_only = getenv_float("M4D_MINIMAL_VISION_ONLY", 0.0f);
    env->minimal_vision_mask_target = getenv_float("M4D_MINIMAL_VISION_MASK_TARGET", 1.0f);
    env->minimal_vision_spawn_visible_target =
        getenv_float("M4D_MINIMAL_VISION_SPAWN_VISIBLE_TARGET", 1.0f);
    env->minimal_vision_fov = 2.0943951f;
    env->minimal_vision_vfov = 1.3962634f;
    env->minimal_vision_sigma = getenv_float("M4D_MINIMAL_VISION_SIGMA", race ? 0.22f : 0.45f);
    env->minimal_vision_depth_gain = getenv_float("M4D_MINIMAL_VISION_DEPTH_GAIN", 0.08f);
    env->minimal_vision_noise = getenv_float("M4D_MINIMAL_VISION_NOISE", 0.02f);
    env->minimal_vision_distractors = getenv_float("M4D_MINIMAL_VISION_DISTRACTORS", 0.0f);
    env->minimal_vision_gate_mask = getenv_float("M4D_MINIMAL_VISION_GATE_MASK", 0.0f);
    env->race_track_mode = getenv_float("RACE_TRACK_MODE", 0.0f);
    env->race_course_yaw_delta = getenv_float("RACE_COURSE_YAW_DELTA", 0.0f);
    env->race_course_pitch_delta = getenv_float("RACE_COURSE_PITCH_DELTA", 0.0f);
    env->race_course_pitch_limit = getenv_float("RACE_COURSE_PITCH_LIMIT", 0.0f);
    env->race_course_spacing_min = getenv_float("RACE_COURSE_SPACING_MIN", 0.0f);
    env->race_course_spacing_max = getenv_float("RACE_COURSE_SPACING_MAX", 0.0f);
    env->race_course_dz_max = getenv_float("RACE_COURSE_DZ_MAX", 0.0f);
    env->race_reset_start_prob = getenv_float("RACE_RESET_START_PROB", 0.0f);
    env->race_reset_t_min = getenv_float("RACE_RESET_T_MIN", 0.0f);
    env->race_reset_t_max = getenv_float("RACE_RESET_T_MAX", 0.0f);
    env->race_reset_lateral = getenv_float("RACE_RESET_LATERAL", 0.0f);
    env->race_reset_yaw_error_frac = getenv_float("RACE_RESET_YAW_ERROR_FRAC", 0.0f);
    env->race_reset_speed_min = getenv_float("RACE_RESET_SPEED_MIN", 0.0f);
    env->race_reset_speed_max = getenv_float("RACE_RESET_SPEED_MAX", 0.0f);
}

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s policy.bin [frames]\n", argv[0]);
        return 2;
    }

    const char* weights_path = argv[1];
    int frames = argc >= 3 ? atoi(argv[2]) : 0;
    int no_render = getenv("M4D_NO_RENDER") != NULL;
    int reset_state_interval = getenv("M4D_RESET_STATE_INTERVAL")
                                   ? atoi(getenv("M4D_RESET_STATE_INTERVAL"))
                                   : 0;

    srand(44);

    DroneEnv* env = (DroneEnv*)calloc(1, sizeof(DroneEnv));
    configure_minimal_vision_baseline(env);
    env->rng = 44u;

    env->observations = (float*)calloc(env->num_agents * DRONE_OBS_SIZE, sizeof(float));
    env->actions = (float*)calloc(env->num_agents * 4, sizeof(float));
    env->rewards = (float*)calloc(env->num_agents, sizeof(float));
    env->terminals = (float*)calloc(env->num_agents, sizeof(float));

    M4DDeploymentRuntime policy;
    if (m4d_deploy_init(&policy, weights_path, env->num_agents, reset_state_interval,
                        env->action_scale) != 0) {
        return 1;
    }

    printf("minimal_vision_baseline policy=%s task=%d obs=%d reset_state_interval=%d\n",
           weights_path, env->task, DRONE_OBS_SIZE, reset_state_interval);
    fflush(stdout);

    init(env);
    c_reset(env);
    if (!no_render) {
        c_render(env);
        SetTargetFPS(60);
    }

    int frame = 0;
    float episode_return = 0.0f;
    int episode_length = 0;
    int episode = 0;

    while ((no_render && (frames <= 0 || frame < frames)) ||
           (!no_render && !WindowShouldClose())) {
        m4d_deploy_forward(&policy, env->observations, env->actions);
        c_step(env);

        episode_return += env->rewards[0];
        episode_length += 1;

        if (env->terminals[0] > 0.0f) {
            printf("episode=%d len=%d return=%.6f\n", ++episode, episode_length,
                   episode_return);
            fflush(stdout);
            episode_return = 0.0f;
            episode_length = 0;
            m4d_deploy_reset_state(&policy);
        }

        if (!no_render) {
            c_render(env);
        }

        if (frames > 0 && ++frame >= frames) {
            break;
        }
    }

    float n = env->log.n > 1e-6f ? env->log.n : 1.0f;
    printf(
        "summary episodes=%.0f rings_passed=%.6f ring_collisions=%.6f collisions=%.6f "
        "oob=%.6f timeout=%.6f lap_complete=%.6f episode_return=%.6f "
        "episode_length=%.6f target_in_fov_frac=%.6f retina_energy=%.6f "
        "bearing_error_to_target=%.6f distance_to_target=%.6f gate_index_at_oob=%.6f\n",
        env->log.n,
        env->log.rings_passed / n,
        env->log.ring_collision / n,
        env->log.collisions / n,
        env->log.oob / n,
        env->log.timeout / n,
        env->log.lap_complete / n,
        env->log.episode_return / n,
        env->log.episode_length / n,
        env->log.target_in_fov_frac / n,
        env->log.retina_energy / n,
        env->log.bearing_error_to_target / n,
        env->log.distance_to_target / n,
        env->log.oob_diag_count > 1e-6f ? env->log.gate_index_at_oob / env->log.oob_diag_count : 0.0f);
    fflush(stdout);

    c_close(env);
    m4d_deploy_close(&policy);
    free(env->observations);
    free(env->actions);
    free(env->rewards);
    free(env->terminals);
    free(env);
    return 0;
}
