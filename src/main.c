#include "matrix.h"
#include "mat_ops.h"
#include "nn.h"

#include <stdio.h>
#include <stdlib.h>

int main(void) {
    xor_net* net = xor_net_create();

    f32 inputs[4][2] = {
        {0.0f, 0.0f},
        {0.0f, 1.0f},
        {1.0f, 0.0f},
        {1.0f, 1.0f}
    };

    f32 labels[4] = {0.0f, 1.0f, 1.0f, 0.0f};

    for (int epoch = 0; epoch < 100000; epoch++) {
        int i = rand() % 4;
        xor_net_train(net, inputs[i], labels[i], 0.1f);
    }

    printf("[0,0] -> %f\n", xor_net_predict(net, (f32[]){0.0f, 0.0f}));
    printf("[0,1] -> %f\n", xor_net_predict(net, (f32[]){0.0f, 1.0f}));
    printf("[1,0] -> %f\n", xor_net_predict(net, (f32[]){1.0f, 0.0f}));
    printf("[1,1] -> %f\n", xor_net_predict(net, (f32[]){1.0f, 1.0f}));

    xor_net_destroy(net);

    return 0;
}