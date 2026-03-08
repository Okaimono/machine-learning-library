#pragma once

#include <stdint.h>

// Types
typedef uint32_t u32;
typedef float f32;

// Matrix (row-major)
typedef struct {
    u32 rows, cols;
    f32* data;
} matrix;

// API
matrix* mat_create(u32 rows, u32 cols);
void    mat_destroy(matrix* mat);

void    mat_fill_array(matrix* mat, f32* values);

matrix* mat_multiply(matrix* a, matrix* b);

// W: (rows, cols), x: (cols) -> returns y: (rows)
// Caller owns returned pointer; must free().
f32*    mat_mul_vec(const matrix* W, const f32* x);
f32*    mat_add_vec(const matrix* B, const f32* x);