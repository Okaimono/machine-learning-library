#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

typedef uint32_t u32;
typedef float f32;
typedef struct {
    u32 rows, cols;
    f32* data;    
} matrix;

matrix* mat_create(u32 rows, u32 cols) {
    matrix* mat = malloc(sizeof(matrix));
    mat->data = malloc(sizeof(f32) * rows * cols);
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
    f32* data = out->data;
    for (u32 i = 0; i < a->rows; i++) {
        for (u32 j = 0; j < b->cols; j++) {
            for (u32 k = 0; k < a->cols; k++) {
                out->data[i * out->cols + j] += a->data[i * a->cols + k] * b->data[k * b->cols + j];
            }
        }
    } 
    return out;
}

int main(void) {
    f32 val_1[4] = {3, 2, 2, 1};
    f32 val_2[4] = {4, 2, 5, 6};

    matrix* first_matrix = mat_create(2, 2);
    matrix* second_matrix = mat_create(2, 2); 
    mat_fill_array(first_matrix, val_1);
    mat_fill_array(second_matrix, val_2);
 
    matrix* composite_matrix = mat_multiply(first_matrix, second_matrix);

    for (u32 i = 0; i < composite_matrix->rows; i++) {
        for (u32 j = 0; j < composite_matrix->cols; j++) {
            printf("%f ", composite_matrix->data[i * composite_matrix->cols + j]);
        }
        printf("\n");
    }

    mat_destroy(first_matrix);
    mat_destroy(second_matrix);
    mat_destroy(composite_matrix);

    return 0;  
}
