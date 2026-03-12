#include "nn.h"
#include "matrix.h"
#include "mat_ops.h"

#include <stdlib.h>
#include <stdio.h>

xor_net* xor_net_create() {
    xor_net* net = calloc(1, sizeof(xor_net));

    const u32 rows_W = 4;
    const u32 cols_W = 2;
    const u32 rows_B = 4;
    const u32 cols_B = 1;

    f32 val_W1[] = {
        -0.7f,  0.2f,
        0.4f, -0.6f,
        -0.1f,  0.8f,
        0.6f, -0.4f
    };
    
    f32 val_B1[] = { 0.0f, 0.0f, 0.0f, 0.0f };
    f32 val_W2[] = { 0.2f, -0.6f, 0.5f, -0.1f };
    f32 val_B2[] = { 0.0f };

    net->W1 = mat_create(4, 2);  mat_fill_array(net->W1, val_W1);
    net->B1 = mat_create(4, 1);  mat_fill_array(net->B1, val_B1);
    net->W2 = mat_create(1, 4);  mat_fill_array(net->W2, val_W2);
    net->B2 = mat_create(1, 1);  mat_fill_array(net->B2, val_B2);

    return net;
}

void xor_net_destroy(xor_net* net) {
    mat_destroy(net->W1);
    mat_destroy(net->W2);
    mat_destroy(net->B1);
    mat_destroy(net->B2);
    free(net);
}
/*
In order to scale our matrix, 
we need to go, for every 
*/

void xor_net_train(xor_net* net, f32* input, f32 y_true, f32 lr) {
    // ============= FORWARD =================

    // (Z1) LINEAR TRANSFORMATION
    f32* tempZ1 = mat_mul_vec(net->W1, input);
    f32* Z1   = mat_add_vec(net->B1, tempZ1);

    // (A1) NONLINEAR TRANSFORMATION
    f32* A1   = apply_sigmoid(Z1, 4);

    // (Z2) LINEAR TRANSFORMATION
    f32* tempZ2 = mat_mul_vec(net->W2, A1); 
    f32* Z2   = mat_add_vec(net->B2, tempZ2);

    // (A2) NONLINEAR TRANSFORMATION
    f32* A2   = apply_sigmoid(Z2, 1);

    // =============== BACKPROP ==============

    f32 dA2 = A2[0] - y_true;
    f32 dZ2 = dA2 * A2[0] * (1 - A2[0]);

    for (int i = 0; i < 4; i++) {
        net->W2->data[i] -= A1[i] * dZ2 * lr;
    }
    net->B2->data[0] -= dZ2 * lr;

    for (int i = 0; i < 4; i++) {
        // Solve for derivative of dA1/dZ1 + factor in standard error
        f32 dL_dA1 = net->W2->data[i] * dZ2;
        f32 dA1_dZ1 = A1[i] * (1 - A1[i]);
        f32 dL_dZ1 = dA1_dZ1 * dL_dA1;

        // Now apply for each part of our linear transformation, do the derivative * standard error 
        for (int j = 0; j < 2; j++) {
            net->W1->data[i * 2 + j] -= input[j] * dL_dZ1 * lr;
        }
        net->B1->data[i] -= dL_dZ1 * lr;
    }

    free(tempZ1);
    free(tempZ2);
    free(Z1);
    free(Z2);
    free(A1);
    free(A2);
}

f32 xor_net_predict(xor_net* net, f32* input) {
    f32* tempZ1 = mat_mul_vec(net->W1, input);
    f32* Z1     = mat_add_vec(net->B1, tempZ1);
    f32* A1     = apply_sigmoid(Z1, 4);

    f32* tempZ2 = mat_mul_vec(net->W2, A1);
    f32* Z2     = mat_add_vec(net->B2, tempZ2);
    f32* A2     = apply_sigmoid(Z2, 1);

    f32 result = A2[0];

    free(tempZ1);
    free(tempZ2);
    free(Z1);
    free(Z2);
    free(A1);
    free(A2);

    return result;
}