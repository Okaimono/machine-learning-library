#include "mat_ops.h"

#include <math.h>

static inline f32 relu_scalar(f32 x) {
    return (x > 0.0f) ? x : 0.0f;
}

void relu_inplace(f32* v, int n) {
    for (int i = 0; i < n; i++) {
        v[i] = relu_scalar(v[i]);
    }
}

static inline f32 sigmoid(f32 x) {
    return 1.0f/ (1.0f + expf(-x));
}

void apply_sigmoid(f32* vec, u32 size) {
    for (u32 i = 0; i < size; i++) {
        vec[i] = sigmoid(vec[i]);
    }
}