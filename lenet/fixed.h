#ifndef FIXED_H
#define FIXED_H

#include "types.h"

#define M 9
#define N 6
#define MAX ((1 << 15) - 1)
#define MIN (~MAX)
#define ONE (1 << N)
#define K (1 << (N - 1))

#define F_LIT(f) (fixed)(f * ONE)
#define F_TO_FLOAT(f) (float)(f) / ONE 
#define F_ADD(a, b) f_add(a, b)
#define F_MUL(a, b) f_mul(a, b)
#define F_LT(a, b) a < b

extern inline fixed f_add(fixed a, fixed b) {
    lint tmp;

    tmp = (lint)a + (lint)b;
    if (tmp > MAX)
        tmp = MAX;
    if (tmp < MIN)
        tmp = MIN;

    return (fixed)tmp;
};

extern inline fixed f_mul(fixed a, fixed b) {
    lint tmp;

    tmp = (lint)a * (lint)b;
    tmp += K;
    tmp >>= N;
    if (tmp > MAX) 
        tmp = MAX;
    if (tmp < MIN) 
        tmp = MIN;
    return (fixed)tmp;
};

#endif