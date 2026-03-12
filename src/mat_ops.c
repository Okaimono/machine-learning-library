#include "mat_ops.h"

#include <math.h>
#include <stdlib.h>

static inline f32 relu_scalar(f32 x) {
    return (x > 0.0f) ? x : 0.0f;
}

f32* relu_inplace(f32* v, int n) {
    f32* out = malloc(n * sizeof(f32));
    for (int i = 0; i < n; i++) {
        out[i] = relu_scalar(v[i]);
    }
    return out;
}

static inline f32 sigmoid(f32 x) {
    return 1.0f/ (1.0f + expf(-x));
}

f32* apply_sigmoid(f32* v, u32 size) {
    f32* out = malloc(size * sizeof(f32));
    for (int i = 0; i < size; i++) {
        out[i] = sigmoid(v[i]);
    }
    return out;
}