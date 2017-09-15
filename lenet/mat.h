#ifndef MAT_H
#define MAT_H

#define NUMARGS(...)  (sizeof((uint[]){__VA_ARGS__})/sizeof(uint))
#define reshape(mat, ...) (Reshape(mat, NUMARGS(__VA_ARGS__), (uint[]){__VA_ARGS__}))
#define constrain(mat, ...) (Constrain(mat, NUMARGS(__VA_ARGS__), (uint[]){__VA_ARGS__}))
#define get(mat, ...) (Get(mat, NUMARGS(__VA_ARGS__), (uint[]){__VA_ARGS__}))
#define set(mat, val, ...) (Set(mat, val, NUMARGS(__VA_ARGS__), (uint[]){__VA_ARGS__}))

typedef unsigned short uint;

typedef struct {
	uint dims[10];
	uint len_dims;
	uint constraints[10];
	uint len_constraints;
	uint constraints_offset;
	uint (*offset_calc)(void *, uint, uint[]);
	float *data;
} mat;

uint get_dim(mat *m, uint axis);

void Reshape(mat *m, uint len, uint dims[]);

void Constrain(mat *m, uint len, uint idxs[]);

void unconstrain(mat *m);

float Get(mat *m, uint len, uint idxs[]);

void Set(mat *m, float val, uint len, uint idxs[]);

#endif