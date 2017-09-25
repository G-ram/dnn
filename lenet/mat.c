#include "mat.h" 
#include <stdio.h>

uint mat_get_dim(mat *m, uint axis) {
	return m->dims[axis + m->len_constraints];
}

void mat_reshape(mat *m, uint len, uint dims[]) {
	m->len_dims = len;
	m->len_constraints = 0;
	m->constraints_offset = 0;
	for(uint i = 0; i < len; i ++) {
		m->dims[i] = dims[i];
	}
}

void mat_constrain(mat *m, uint len, uint idxs[]) {
	for(uint i = 0; i < len; i ++) {
		m->constraints[i] = idxs[i];
	}	
	m->len_constraints = len;
	uint offset = 0;
	for(uint i = 0; i < len; i ++) {
		uint factor = 1;
		uint factor_idx = m->len_dims - i;
		for(short j = factor_idx - 1; j > 0; j --) {
			factor *= m->dims[j];
		}
		offset += factor * idxs[i];
	}
	m->constraints_offset = offset;
}

void mat_unconstrain(mat *m) {
	m->len_constraints = 0;
	m->constraints_offset = 0;
}

uint _offset_calc(void *_m, uint len, uint idxs[]) {
	mat *m = (mat *)_m;
	uint offset = 0;
	for(uint i = 0; i < len; i ++) {
		uint factor = 1;
		uint factor_idx = m->len_dims - i;
		for(short j = factor_idx - 1; j > m->len_constraints; j --) {
			factor *= m->dims[j];
		}
		offset += factor * idxs[i];
	}
	return offset;
}

fixed mat_get(mat *m, uint len, uint idxs[]) {
	return *(m->data + _offset_calc(m, len, idxs) + m->constraints_offset);
}

void mat_set(mat *m, fixed val, uint len, uint idxs[]) {
	*(m->data + _offset_calc(m, len, idxs) + m->constraints_offset) = val;
}