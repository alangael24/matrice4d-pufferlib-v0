#include "drone.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

void c_close_client(Client* client) {
    (void)client;
}

static float getenv_float(const char* name, float fallback) {
    const char* value = getenv(name);
    return value != NULL ? strtof(value, NULL) : fallback;
}

static void configure_race_geometry_env(DroneEnv* env) {
    env->num_agents = 1;
    env->max_rings = 8;
    env->task = RACE;
    env->alpha_dist = 0.5f;
    env->alpha_hover = 0.0f;
    env->alpha_shaping = 0.0f;
    env->alpha_omega_xy = 0.004f;
    env->alpha_omega_z = 0.004f;
    env->alpha_omega_z_sq = 0.006f;
    env->alpha_omega_z_mult = 1.5f;
    env->alpha_action_delta = 0.01f;
    env->hover_target_dist = 4.0f;
    env->oob_radius = 18.0f;
    env->hover_dist = 0.1f;
    env->hover_omega = 0.1f;
    env->hover_vel = 0.1f;
    env->domain_randomization = 0.0f;
    env->action_mode = M4D_ACTION_NORMALIZED_THRUST;
    env->normalized_thrust_min = 0.0f;
    env->normalized_thrust_max = 0.80f;
    env->action_scale = 1.0f;
    env->minimal_vision_enabled = 1.0f;
    env->minimal_vision_only = 0.0f;
    env->minimal_vision_mask_target = 1.0f;
    env->minimal_vision_spawn_visible_target = 1.0f;
    env->minimal_vision_fov = 2.0943951f;
    env->minimal_vision_vfov = 1.3962634f;
    env->minimal_vision_sigma = 0.22f;
    env->minimal_vision_depth_gain = 0.08f;
    env->race_track_mode = getenv_float("RACE_TRACK_MODE", 0.0f);
    env->race_course_yaw_delta = getenv_float("RACE_COURSE_YAW_DELTA", 0.0f);
    env->race_course_pitch_delta = getenv_float("RACE_COURSE_PITCH_DELTA", 0.0f);
    env->race_course_pitch_limit = getenv_float("RACE_COURSE_PITCH_LIMIT", 0.0f);
    env->race_course_spacing_min = getenv_float("RACE_COURSE_SPACING_MIN", 0.0f);
    env->race_course_spacing_max = getenv_float("RACE_COURSE_SPACING_MAX", 0.0f);
    env->race_course_dz_max = getenv_float("RACE_COURSE_DZ_MAX", 0.0f);
}

int main(int argc, char** argv) {
    int tracks = argc > 1 ? atoi(argv[1]) : 4096;
    float margin = 1.0f;

    DroneEnv env = {0};
    configure_race_geometry_env(&env);
    env.rng = 44u;
    env.observations = (float*)calloc((size_t)env.num_agents * DRONE_OBS_SIZE, sizeof(float));
    env.actions = (float*)calloc((size_t)env.num_agents * 4, sizeof(float));
    env.rewards = (float*)calloc((size_t)env.num_agents, sizeof(float));
    env.terminals = (float*)calloc((size_t)env.num_agents, sizeof(float));
    if (env.observations == NULL || env.actions == NULL || env.rewards == NULL || env.terminals == NULL) {
        fprintf(stderr, "allocation failed\n");
        return 2;
    }

    init(&env);

    float max_gate_norm = 0.0f;
    float max_segment = 0.0f;
    float max_wrap_segment = 0.0f;
    int long_segments = 0;
    int long_wrap_segments = 0;

    for (int t = 0; t < tracks; t++) {
        c_reset(&env);
        Drone* agent = &env.agents[0];
        for (int i = 0; i < agent->buffer_size; i++) {
            float gate_norm = norm3(agent->buffer[i].pos);
            if (gate_norm > max_gate_norm) max_gate_norm = gate_norm;
            if (i + 1 < agent->buffer_size) {
                float segment = norm3(sub3(agent->buffer[i + 1].pos, agent->buffer[i].pos));
                if (segment > max_segment) max_segment = segment;
                if (segment > env.oob_radius - margin) long_segments += 1;
            }
        }
        if (agent->buffer_size > 1) {
            float wrap_segment = norm3(sub3(agent->buffer[0].pos,
                                            agent->buffer[agent->buffer_size - 1].pos));
            if (wrap_segment > max_wrap_segment) max_wrap_segment = wrap_segment;
            if (wrap_segment > env.oob_radius - margin) long_wrap_segments += 1;
        }
    }

    printf("tracks=%d gates=%d oob_radius=%.3f margin=%.3f\n",
           tracks, env.max_rings, env.oob_radius, margin);
    printf("max_gate_position_norm=%.6f\n", max_gate_norm);
    printf("max_segment_between_consecutive_gates=%.6f\n", max_segment);
    printf("segments_over_oob_minus_margin=%d\n", long_segments);
    printf("max_wrap_segment_last_to_first=%.6f\n", max_wrap_segment);
    printf("wrap_segments_over_oob_minus_margin=%d\n", long_wrap_segments);

    return long_segments == 0 ? 0 : 1;
}
