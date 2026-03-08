#pragma once
#include "matrix.h"

typedef struct {
    matrix* W1;
    matrix* B1;
    matrix* W2;
    matrix* B2;
} xor_net;

xor_net* xor_net_create();
void xor_net_destroy(xor_net* net);

void xor_net_train(xor_net* net, f32* input, f32 y_true, f32 lr);

f32 xor_net_predict(xor_net* net, f32* input);