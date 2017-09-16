#ifndef MAT_H
#define MAT_H

#define NUMARGS(...)  (sizeof((uint[]){__VA_ARGS__})/sizeof(uint))
#define MAT_RESHAPE(m, ...) (mat_reshape(m, NUMARGS(__VA_ARGS__), 			\
								(uint[]){__VA_ARGS__}))
#define MAT_CONSTRAIN(m, ...) (mat_constrain(m, NUMARGS(__VA_ARGS__), 		\
								(uint[]){__VA_ARGS__}))
#define MAT_UNCONSTRAIN(m) (mat_unconstrain(m))
#define MAT_GET(m, ...) (													\
	(NUMARGS(__VA_ARGS__) == 1) ? 											\
		*(m->data + + m->constraints_offset + ((uint[]){__VA_ARGS__})[0]) :	\
	(NUMARGS(__VA_ARGS__) == 2) ? 											\
		*(m->data + + m->constraints_offset + ((uint[]){__VA_ARGS__})[0] * 	\
			m->dims[m->len_dims - 1] + ((uint[]){__VA_ARGS__})[1]):			\
	(NUMARGS(__VA_ARGS__) == 3) ? 											\
		*(m->data + + m->constraints_offset + ((uint[]){__VA_ARGS__})[0] *	\
			m->dims[m->len_dims - 1] * m->dims[m->len_dims - 2] +			\
			m->dims[m->len_dims - 1] * ((uint[]){__VA_ARGS__})[1] + 		\
			((uint[]){__VA_ARGS__})[3]):									\
	mat_get(m, NUMARGS(__VA_ARGS__), (uint[]){__VA_ARGS__}))
#define MAT_SET(m, val, ...) (												\
	(NUMARGS(__VA_ARGS__) == 1) ? 											\
		*(m->data + + m->constraints_offset +								\
			((uint[]){__VA_ARGS__})[0]) = val :								\
	(NUMARGS(__VA_ARGS__) == 2) ? 											\
		*(m->data + + m->constraints_offset + ((uint[]){__VA_ARGS__})[0] * 	\
			m->dims[m->len_dims - 1] + ((uint[]){__VA_ARGS__})[1]) = val :	\
	(NUMARGS(__VA_ARGS__) == 3) ? 											\
		*(m->data + + m->constraints_offset + ((uint[]){__VA_ARGS__})[0] *	\
			m->dims[m->len_dims - 1] * m->dims[m->len_dims - 2] +			\
			m->dims[m->len_dims - 1] * ((uint[]){__VA_ARGS__})[1] + 		\
			((uint[]){__VA_ARGS__})[3]) = val :								\
	mat_set(m, val, NUMARGS(__VA_ARGS__), (uint[]){__VA_ARGS__}))

#define MAT_GET_DIM(m, axis) (mat_get_dim(m, axis))

typedef unsigned short uint;

typedef struct {
	uint dims[10];
	uint len_dims;
	uint constraints[10];
	uint len_constraints;
	uint constraints_offset;
	float *data;
} mat;

uint mat_get_dim(mat *m, uint axis);

void mat_reshape(mat *m, uint len, uint dims[]);

void mat_constrain(mat *m, uint len, uint idxs[]);

void mat_unconstrain(mat *m);

float mat_get(mat *m, uint len, uint idxs[]);

void mat_set(mat *m, float val, uint len, uint idxs[]);

#endif