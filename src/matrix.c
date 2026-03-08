#include "matrix.h"

#include <stdlib.h> // malloc, calloc, free

matrix* mat_create(u32 rows, u32 cols) {
    matrix* mat = (matrix*)malloc(sizeof(matrix));
    mat->data = (f32*)calloc(rows * cols, sizeof(f32));
    mat->rows = rows;
    mat->cols = cols;
    return mat;
}

void mat_destroy(matrix* mat) {
    free(mat->data);
    free(mat);
}

void mat_fill_array(matrix* mat, f32* values) {
    u32 size = mat->rows * mat->cols;
    f32* data = mat->data;
    for (u32 i = 0; i < size; i++) {
        data[i] = values[i];
    }
}

matrix* mat_multiply(matrix* a, matrix* b) {
    matrix* out = mat_create(a->rows, b->cols);

    for (u32 i = 0; i < a->rows; i++) {
        for (u32 j = 0; j < b->cols; j++) {
            for (u32 k = 0; k < a->cols; k++) {
                out->data[i * out->cols + j] +=
                    a->data[i * a->cols + k] *
                    b->data[k * b->cols + j];
            }
        }
    }

    return out;
}

f32* mat_mul_vec(const matrix* W, const f32* x) {
    f32* y = (f32*)calloc(W->rows, sizeof(f32));

    for (u32 rows = 0; rows < W->rows; rows++) {
        for (u32 cols = 0; cols < W->cols; cols++) {
            y[rows] += x[cols] * W->data[rows * W->cols + cols];
        }
    }

    return y;
}

f32* mat_add_vec(const matrix* B, const f32* x) {
    f32* y = (f32*)calloc(B->rows, sizeof(f32));

    for (u32 rows = 0; rows < B->rows; rows++) {
        y[rows] = x[rows] + B->data[rows];
    }

    return y;
}

/*
RANDOM DAYDREAM THOUGHTS ABOUT AI HERE:

so the error is not that the math is to complex but that 
we would need an o( to the power of terrifying  n). or make a new model 
that has more capabilities to scale differently, 
so its more adjustable and flexible. i guess the error here is, 
maybe theres a chance like with intellect, u can abstract to crazy amounts, 
and this gets rid of the rigidness that exists. with ai models, 
they can't control how abstract it thinks or how rigid it is. 
so we need to control it so that way it is most rigid and grounded in humans interests, 
then adjust cognitive firepower and make it abstract it in specific directions
enabiling creative firepower, rather than putting makeup on it.
the issue is ai models are guilded. they hold internal structural flaws
that disable the ability for it to grow in the proper direction it needs to.
instead, the architectural disadvantage is being covered with makeup.
this should result in major issues and bottlenecks that should appear soon.
*/