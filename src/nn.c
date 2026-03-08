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

void xor_net_train(xor_net* net, f32* input, f32 y_true, f32 lr) {
    f32* Z1 = mat_mul_vec(net->W1, input);
    f32* A1 = mat_add_vec(net->B1, Z1);
    apply_sigmoid(A1, 4);

    f32* Z2 = mat_mul_vec(net->W2, A1);     // dot product
    f32* A2 = mat_add_vec(net->B2, Z2);     // add bias
    apply_sigmoid(A2, 1);                   // sigmoid the single output value

    f32 dA2 = A2[0] - y_true;

    // step 2: sigmoid derivative at output
    f32 dZ2 = dA2 * A2[0] * (1.0f - A2[0]);

    // step 3: update W2 and B2
    for (u32 i = 0; i < 4; i++) {
        net->W2->data[i] -= lr * dZ2 * A1[i];
    }
    net->B2->data[0] -= lr * dZ2;

    // step 4: propagate error back to layer 1
    f32 dA1[4];
    for (u32 i = 0; i < 4; i++) {
        dA1[i] = net->W2->data[i] * dZ2;
    }

    // step 5: sigmoid derivative at layer 1
    f32 dZ1[4];
    for (u32 i = 0; i < 4; i++) {
        dZ1[i] = dA1[i] * A1[i] * (1.0f - A1[i]);
    }

    // step 6: update W1 and B1
    for (u32 i = 0; i < 4; i++) {
        for (u32 j = 0; j < 2; j++) {
            net->W1->data[i * 2 + j] -= lr * dZ1[i] * input[j];
        }
        net->B1->data[i] -= lr * dZ1[i];
    }

    free(Z1);
    free(A1);
    free(Z2);
    free(A2);
}

f32 xor_net_predict(xor_net* net, f32* input) {
    f32* Z1 = mat_mul_vec(net->W1, input);
    f32* A1 = mat_add_vec(net->B1, Z1);
    apply_sigmoid(A1, 4);

    f32* Z2 = mat_mul_vec(net->W2, A1);
    f32* A2 = mat_add_vec(net->B2, Z2);
    apply_sigmoid(A2, 1);

    f32 result = A2[0];

    free(Z1);
    free(A1);
    free(Z2);
    free(A2);

    return result;
}