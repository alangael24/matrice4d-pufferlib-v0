#include "m4d_deployment_runtime.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>

static int read_float_file(const char* path, float* values, size_t count) {
    FILE* file = fopen(path, "rb");
    if (file == NULL) {
        perror("Error opening observation file");
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

    char* text = (char*)calloc((size_t)bytes + 1, sizeof(char));
    if (text == NULL) {
        fclose(file);
        return -1;
    }
    size_t read = fread(text, 1, (size_t)bytes, file);
    fclose(file);
    if (read != (size_t)bytes) {
        free(text);
        return -1;
    }

    char* p = text;
    size_t n = 0;
    while (*p != '\0' && n < count) {
        while (*p == ' ' || *p == '\n' || *p == '\r' || *p == '\t' || *p == ',') {
            p++;
        }
        if (*p == '\0') {
            break;
        }
        errno = 0;
        char* end = p;
        float v = strtof(p, &end);
        if (end == p || errno == ERANGE) {
            fprintf(stderr, "Failed to parse float near byte offset %ld in %s\n",
                    (long)(p - text), path);
            free(text);
            return -1;
        }
        values[n++] = v;
        p = end;
    }

    free(text);
    if (n != count) {
        fprintf(stderr, "Observation file has %zu floats; expected %zu\n", n, count);
        return -1;
    }
    return 0;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr,
                "Usage: %s WEIGHTS.bin OBS_FILE [steps] [num_agents] [reset_interval] "
                "[action_scale]\n",
                argv[0]);
        return 2;
    }

    const char* weights_path = argv[1];
    const char* obs_path = argv[2];
    int steps = argc >= 4 ? atoi(argv[3]) : 40;
    int num_agents = argc >= 5 ? atoi(argv[4]) : 3;
    int reset_interval = argc >= 6 ? atoi(argv[5]) : M4D_DEPLOY_DEFAULT_RESET_INTERVAL;
    float action_scale = argc >= 7 ? (float)atof(argv[6]) : M4D_DEPLOY_DEFAULT_ACTION_SCALE;
    if (steps <= 0 || num_agents <= 0) {
        fprintf(stderr, "steps and num_agents must be positive\n");
        return 2;
    }

    M4DDeploymentRuntime rt;
    if (m4d_deploy_init(&rt, weights_path, num_agents, reset_interval, action_scale) != 0) {
        return 1;
    }

    size_t obs_count = (size_t)steps * num_agents * M4D_DEPLOY_OBS_SIZE;
    float* obs = (float*)calloc(obs_count, sizeof(float));
    float* raw = (float*)calloc((size_t)num_agents * M4D_DEPLOY_NUM_ACTIONS, sizeof(float));
    float* scaled = (float*)calloc((size_t)num_agents * M4D_DEPLOY_NUM_ACTIONS, sizeof(float));
    if (obs == NULL || raw == NULL || scaled == NULL) {
        m4d_deploy_close(&rt);
        free(obs);
        free(raw);
        free(scaled);
        return 1;
    }

    if (read_float_file(obs_path, obs, obs_count) != 0) {
        m4d_deploy_close(&rt);
        free(obs);
        free(raw);
        free(scaled);
        return 1;
    }

    printf("kind,step,agent,raw0,raw1,raw2,raw3,scaled0,scaled1,scaled2,scaled3\n");
    for (int step = 0; step < steps; step++) {
        const float* step_obs = obs + (size_t)step * num_agents * M4D_DEPLOY_OBS_SIZE;
        m4d_deploy_forward(&rt, step_obs, raw);
        m4d_deploy_scale_actions(&rt, raw, scaled);

        for (int agent = 0; agent < num_agents; agent++) {
            float* r = raw + (size_t)agent * M4D_DEPLOY_NUM_ACTIONS;
            float* s = scaled + (size_t)agent * M4D_DEPLOY_NUM_ACTIONS;
            printf("action,%d,%d,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g\n", step, agent,
                   r[0], r[1], r[2], r[3], s[0], s[1], s[2], s[3]);
        }
    }

    m4d_deploy_close(&rt);
    free(obs);
    free(raw);
    free(scaled);
    return 0;
}
