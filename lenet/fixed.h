#ifndef FIXED_H
#define FIXED_H

#define M 7
#define N 8
#define MAX ((1 << 15) - 1)
#define MIN (~MAX)
#define ONE (1 << N)
#define K (1 << (N - 1))

#define F_LIT(f) (fixed)(f * ONE)
#define F_TO_FLOAT(f) (float)(f) / ONE 
#define F_ADD(a, b) f_add(a, b)
#define F_MUL(a, b) f_mul(a, b)

typedef signed short fixed;
typedef signed int lint;

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