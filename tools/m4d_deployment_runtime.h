#pragma once

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define M4D_DEPLOY_OBS_SIZE 23
#define M4D_DEPLOY_NUM_ACTIONS 4
#define M4D_DEPLOY_HIDDEN_SIZE 128
#define M4D_DEPLOY_NUM_LAYERS 3
#define M4D_DEPLOY_DECODER_OUTPUTS (M4D_DEPLOY_NUM_ACTIONS + 1)
#define M4D_DEPLOY_DEFAULT_RESET_INTERVAL 32
#define M4D_DEPLOY_DEFAULT_ACTION_SCALE 0.7f

typedef struct {
    int num_agents;
    int step;
    int reset_state_interval;
    float action_scale;

    float* weights;
    size_t num_weights;
    size_t weight_capacity;
    float* encoder_w;
    float* decoder_w;
    float* logstd;
    float* mingru_w[M4D_DEPLOY_NUM_LAYERS];

    float* state;
    float* x;
    float* next_x;
    float* combined;
} M4DDeploymentRuntime;

static inline float m4d_deploy_sigmoid(float x) {
    float z = expf(-fabsf(x));
    return x >= 0.0f ? 1.0f / (1.0f + z) : z / (1.0f + z);
}

static inline float m4d_deploy_clampf(float v, float lo, float hi) {
    if (v < lo) return lo;
    if (v > hi) return hi;
    return v;
}

static inline size_t m4d_deploy_expected_weights(void) {
    return (size_t)M4D_DEPLOY_HIDDEN_SIZE * M4D_DEPLOY_OBS_SIZE
         + (size_t)M4D_DEPLOY_DECODER_OUTPUTS * M4D_DEPLOY_HIDDEN_SIZE
         + (size_t)M4D_DEPLOY_NUM_ACTIONS
         + (size_t)M4D_DEPLOY_NUM_LAYERS * 3 * M4D_DEPLOY_HIDDEN_SIZE * M4D_DEPLOY_HIDDEN_SIZE;
}

static inline size_t m4d_deploy_align8(size_t value) {
    return (value + 7u) & ~((size_t)7u);
}

static inline float* m4d_deploy_take_aligned(M4DDeploymentRuntime* rt, size_t* off,
                                             size_t count) {
    // Mirrors the native backend allocation contract: tensor data pointers are
    // 16-byte aligned, while saved checkpoints contain only total_elems floats.
    float* out = rt->weights + *off;
    *off += count;
    *off = m4d_deploy_align8(*off);
    return out;
}

static inline int m4d_deploy_load_weights(M4DDeploymentRuntime* rt, const char* path) {
    FILE* file = fopen(path, "rb");
    if (file == NULL) {
        perror("Error opening weight file");
        return -1;
    }

    if (fseek(file, 0, SEEK_END) != 0) {
        fclose(file);
        return -1;
    }
    long bytes = ftell(file);
    if (bytes < 0) {
        fclose(file);
        return -1;
    }
    rewind(file);

    if ((bytes % (long)sizeof(float)) != 0) {
        fprintf(stderr, "Weight file size is not a multiple of float size: %ld\n", bytes);
        fclose(file);
        return -1;
    }

    rt->num_weights = (size_t)bytes / sizeof(float);
    size_t expected = m4d_deploy_expected_weights();
    if (rt->num_weights != expected) {
        fprintf(stderr, "Unexpected weight count: got %zu expected %zu\n", rt->num_weights, expected);
        fclose(file);
        return -1;
    }

    rt->weight_capacity = rt->num_weights + 7;
    rt->weights = (float*)calloc(rt->weight_capacity, sizeof(float));
    if (rt->weights == NULL) {
        fclose(file);
        return -1;
    }

    size_t read = fread(rt->weights, sizeof(float), rt->num_weights, file);
    fclose(file);
    if (read != rt->num_weights) {
        fprintf(stderr, "Failed to read weight file: read %zu expected %zu\n", read, rt->num_weights);
        return -1;
    }

    size_t off = 0;
    rt->encoder_w = m4d_deploy_take_aligned(
        rt, &off, (size_t)M4D_DEPLOY_HIDDEN_SIZE * M4D_DEPLOY_OBS_SIZE);
    rt->decoder_w = m4d_deploy_take_aligned(
        rt, &off, (size_t)M4D_DEPLOY_DECODER_OUTPUTS * M4D_DEPLOY_HIDDEN_SIZE);
    rt->logstd = m4d_deploy_take_aligned(rt, &off, M4D_DEPLOY_NUM_ACTIONS);
    for (int l = 0; l < M4D_DEPLOY_NUM_LAYERS; l++) {
        rt->mingru_w[l] = m4d_deploy_take_aligned(
            rt, &off, (size_t)3 * M4D_DEPLOY_HIDDEN_SIZE * M4D_DEPLOY_HIDDEN_SIZE);
    }

    if (off > rt->weight_capacity) {
        fprintf(stderr, "Aligned weight layout exceeds padded capacity: off=%zu capacity=%zu\n",
                off, rt->weight_capacity);
        return -1;
    }
    return 0;
}

static inline int m4d_deploy_init(M4DDeploymentRuntime* rt, const char* path, int num_agents,
                                  int reset_state_interval, float action_scale) {
    memset(rt, 0, sizeof(*rt));
    rt->num_agents = num_agents > 0 ? num_agents : 1;
    rt->reset_state_interval = reset_state_interval;
    rt->action_scale = action_scale;

    if (m4d_deploy_load_weights(rt, path) != 0) {
        return -1;
    }

    size_t state_count = (size_t)M4D_DEPLOY_NUM_LAYERS * rt->num_agents * M4D_DEPLOY_HIDDEN_SIZE;
    size_t hidden_count = (size_t)rt->num_agents * M4D_DEPLOY_HIDDEN_SIZE;
    size_t combined_count = (size_t)rt->num_agents * 3 * M4D_DEPLOY_HIDDEN_SIZE;
    rt->state = (float*)calloc(state_count, sizeof(float));
    rt->x = (float*)calloc(hidden_count, sizeof(float));
    rt->next_x = (float*)calloc(hidden_count, sizeof(float));
    rt->combined = (float*)calloc(combined_count, sizeof(float));
    if (rt->state == NULL || rt->x == NULL || rt->next_x == NULL || rt->combined == NULL) {
        return -1;
    }
    return 0;
}

static inline void m4d_deploy_reset_state(M4DDeploymentRuntime* rt) {
    memset(rt->state, 0,
           (size_t)M4D_DEPLOY_NUM_LAYERS * rt->num_agents * M4D_DEPLOY_HIDDEN_SIZE *
               sizeof(float));
}

static inline void m4d_deploy_matmul(const float* input, const float* weights, float* output,
                                     int batch, int in_dim, int out_dim) {
    for (int b = 0; b < batch; b++) {
        for (int o = 0; o < out_dim; o++) {
            float sum = 0.0f;
            const float* w = weights + (size_t)o * in_dim;
            const float* x = input + (size_t)b * in_dim;
            for (int i = 0; i < in_dim; i++) {
                sum += x[i] * w[i];
            }
            output[(size_t)b * out_dim + o] = sum;
        }
    }
}

static inline void m4d_deploy_forward_no_reset(M4DDeploymentRuntime* rt, const float* obs,
                                               float* actions) {
    m4d_deploy_matmul(obs, rt->encoder_w, rt->x, rt->num_agents, M4D_DEPLOY_OBS_SIZE,
                      M4D_DEPLOY_HIDDEN_SIZE);

    float* x = rt->x;
    float* out = rt->next_x;
    for (int l = 0; l < M4D_DEPLOY_NUM_LAYERS; l++) {
        m4d_deploy_matmul(x, rt->mingru_w[l], rt->combined, rt->num_agents,
                          M4D_DEPLOY_HIDDEN_SIZE, 3 * M4D_DEPLOY_HIDDEN_SIZE);
        float* state_l = rt->state + (size_t)l * rt->num_agents * M4D_DEPLOY_HIDDEN_SIZE;
        for (int b = 0; b < rt->num_agents; b++) {
            float* cb = rt->combined + (size_t)b * 3 * M4D_DEPLOY_HIDDEN_SIZE;
            float* sb = state_l + (size_t)b * M4D_DEPLOY_HIDDEN_SIZE;
            float* xb = x + (size_t)b * M4D_DEPLOY_HIDDEN_SIZE;
            float* ob = out + (size_t)b * M4D_DEPLOY_HIDDEN_SIZE;
            for (int h = 0; h < M4D_DEPLOY_HIDDEN_SIZE; h++) {
                float hidden = cb[h];
                float gate = cb[M4D_DEPLOY_HIDDEN_SIZE + h];
                float proj = cb[2 * M4D_DEPLOY_HIDDEN_SIZE + h];
                float hidden_tilde = hidden >= 0.0f ? hidden + 0.5f : m4d_deploy_sigmoid(hidden);
                float gate_sigmoid = m4d_deploy_sigmoid(gate);
                float mingru_out = sb[h] + gate_sigmoid * (hidden_tilde - sb[h]);
                float proj_sigmoid = m4d_deploy_sigmoid(proj);
                ob[h] = proj_sigmoid * mingru_out + (1.0f - proj_sigmoid) * xb[h];
                sb[h] = mingru_out;
            }
        }
        float* tmp = x;
        x = out;
        out = tmp;
    }

    for (int b = 0; b < rt->num_agents; b++) {
        for (int a = 0; a < M4D_DEPLOY_NUM_ACTIONS; a++) {
            float sum = 0.0f;
            const float* w = rt->decoder_w + (size_t)a * M4D_DEPLOY_HIDDEN_SIZE;
            const float* xb = x + (size_t)b * M4D_DEPLOY_HIDDEN_SIZE;
            for (int h = 0; h < M4D_DEPLOY_HIDDEN_SIZE; h++) {
                sum += xb[h] * w[h];
            }
            actions[(size_t)b * M4D_DEPLOY_NUM_ACTIONS + a] = sum;
        }
    }
}

static inline void m4d_deploy_forward(M4DDeploymentRuntime* rt, const float* obs, float* actions) {
    if (rt->reset_state_interval > 0 && rt->step % rt->reset_state_interval == 0) {
        m4d_deploy_reset_state(rt);
    }
    m4d_deploy_forward_no_reset(rt, obs, actions);
    rt->step += 1;
}

static inline void m4d_deploy_scale_actions(const M4DDeploymentRuntime* rt, const float* raw,
                                            float* scaled) {
    for (int i = 0; i < rt->num_agents * M4D_DEPLOY_NUM_ACTIONS; i++) {
        float clipped = m4d_deploy_clampf(raw[i], -1.0f, 1.0f);
        scaled[i] = m4d_deploy_clampf(clipped * rt->action_scale, -1.0f, 1.0f);
    }
}

static inline void m4d_deploy_close(M4DDeploymentRuntime* rt) {
    free(rt->weights);
    free(rt->state);
    free(rt->x);
    free(rt->next_x);
    free(rt->combined);
    memset(rt, 0, sizeof(*rt));
}
