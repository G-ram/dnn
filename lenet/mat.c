#include "mat.h" 
#include <stdio.h>

uint get_dim(mat *m, uint axis) {
	return m->dims[axis + m->len_constraints];
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

uint _offset_calc1D(void *_m, uint len, uint idxs[]) {
	 return idxs[0];
}

uint _offset_calc2D(void *_m, uint len, uint idxs[]) {
	mat *m = (mat *)_m;
	if(len == 1) {
		return _offset_calc1D(m, len, idxs);
	}
	uint col_idx = m->len_dims - 1;
	return idxs[0] * m->dims[col_idx] + idxs[1];	
}

uint _offset_calc3D(void *_m, uint len, uint idxs[]) {
	mat *m = (mat *)_m;
	if(len == 2){
		return _offset_calc2D(m, len, idxs);
	} else if(len == 1) {
		return _offset_calc1D(m, len, idxs);
	}
	uint row_idx = m->len_dims - 2;
	uint col_idx = m->len_dims - 1;
	return idxs[0] * m->dims[row_idx] * m->dims[col_idx] + 
			idxs[1] * m->dims[col_idx] + idxs[2];
}

void Reshape(mat *m, uint len, uint dims[]) {
	m->len_dims = len;
	m->len_constraints = 0;
	m->constraints_offset = 0;
	for(uint i = 0; i < len; i ++) {
		m->dims[i] = dims[i];
	}
	if(len == 1){
		m->offset_calc = _offset_calc1D;
	} else if(len == 2) {
		m->offset_calc = _offset_calc2D;
	} else if(len == 3) {
		m->offset_calc = _offset_calc3D;
	} else {
		m->offset_calc = _offset_calc;
	}
}

void Constrain(mat *m, uint len, uint idxs[]) {
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

void unconstrain(mat *m) {
	m->len_constraints = 0;
	m->constraints_offset = 0;
}

float Get(mat *m, uint len, uint idxs[]) {
	return *(m->data + m->offset_calc(m, len, idxs) + m->constraints_offset);
}

void Set(mat *m, float val, uint len, uint idxs[]) {
	*(m->data + m->offset_calc(m, len, idxs) + m->constraints_offset) = val;
}