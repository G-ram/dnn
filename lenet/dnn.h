#ifndef DNN_H
#define DNN_H

#include "headers/conv1.h"
#include "headers/conv2.h"
#include "headers/pr.h"
#include "headers/pred.h"
#include "headers/input.h"
#include "types.h"
#include "mat.h"
#include "fixed.h"

uint infer(float src[28][28]);
void conv1_layer(mat *src, mat *dest);
void pool1_layer(mat *src, mat *dest);
void conv2_layer(mat *src, mat *dest);
void pool2_layer(mat *src, mat *dest);
void pr_layer(mat *src, mat *dest);
void relu_layer(mat *src, mat *dest);
void pred_layer(mat *src, mat *dest);

void convolve2d(mat *src, uint size, fixed filter[][size],mat *dest);

void convolve3d(mat *src, uint size, fixed filter[][size][size], mat *dest);

void mul_vector(uint rows, uint cols, fixed mat_data[][cols], mat *vector, 
	mat *dest);

void sparse_mul_vector(uint rows, fixed mat_data[], uint mat_idx[], 
	uint mat_ptr[], mat *vector, mat *dest);

void bias2d(mat *src, fixed bias, mat *dest);

void bias1d(mat *src, fixed bias[], mat *dest);

void pool(mat *src, uint size, uint stride, mat *dest);

void relu(mat *src, mat *dest);

#endif
