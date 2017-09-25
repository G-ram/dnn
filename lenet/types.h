#ifndef TYPES_H
#define TYPES_H

typedef signed short fixed;
typedef signed int lint;
typedef unsigned short uint;

typedef struct {
	uint dims[10];
	uint len_dims;
	uint constraints[10];
	uint len_constraints;
	uint constraints_offset;
	fixed *data;
} mat;

#endif