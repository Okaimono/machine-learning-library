#pragma once
#include "matrix.h"

f32* apply_sigmoid(f32* vec, u32 size);

f32* relu_inplace(f32* v, int n);