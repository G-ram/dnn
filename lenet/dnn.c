#include "dnn.h"
#include <stdio.h>
#include <math.h>

int main() {
	uint correct = 0;
	for(uint i = 0; i < num; i ++) {
		uint label = infer(input[i]);
		if(label == labels[i])
			correct++;

		printf("Label: %d Actual: %d\n", label, labels[i]);
	}
	printf("Correct: %d / %d Percentage: %f \n", correct, num, 
		(float) correct / (float) num * 100.);
	return 0;
}

uint infer(float src[28][28]) {
	// Init two buffers
	float data1[20 * 24 * 24], data2[100 * 8 * 8];
	mat buf1, buf2;
	mat *b1 = &buf1;
	b1->data = data1;
	mat *b2 = &buf2;
	b2->data = data2;

	MAT_RESHAPE(b2, 28, 28);
	for(uint i = 0; i < 28; i ++) {
		for(uint j = 0; j < 28; j ++) {
			MAT_SET(b2, src[i][j], i, j);
		}
	}

	MAT_RESHAPE(b1, 20, 24, 24);
	conv1_layer(b2, b1);

	MAT_RESHAPE(b2, 20, 12, 12);
	pool1_layer(b1, b2);

	MAT_RESHAPE(b1, 100, 8, 8);
	conv2_layer(b2, b1);

	MAT_RESHAPE(b2, 100, 4, 4);
	pool2_layer(b1, b2);

	MAT_RESHAPE(b1, 500);
	pr_layer(b2, b1);

	MAT_RESHAPE(b2, 500);
	relu_layer(b1, b2);

	MAT_RESHAPE(b1, 10);
	pred_layer(b2, b1);

	float max = 0.;
	uint idx = 0;
	for(uint i = 0; i < 10; i ++) {
		float prob = MAT_GET(b1, i);
		if(max < prob) {
			idx = i;
			max = prob;
		}
		// printf("%d => %f\n", i, prob);
	}
	return idx;
}

void conv1_layer(mat *src, mat *dest){
	float data[24 * 24];
	mat it;
	mat *inter = &it;
	inter->data = data;
	MAT_RESHAPE(inter, 24, 24);
	for(uint i = 0; i < 20; i ++) {
		convolve2d(src, 5, conv1_w[i], inter);
		MAT_CONSTRAIN(dest, i);
		bias2d(inter, conv1_b[i], dest);
		MAT_UNCONSTRAIN(dest);
	}
}

void pool1_layer(mat *src, mat *dest){
	for(uint i = 0; i < 20; i ++) {
		MAT_CONSTRAIN(src, i);
		MAT_CONSTRAIN(dest, i);
		pool(src, 2, 2, dest);
		MAT_UNCONSTRAIN(src);
		MAT_UNCONSTRAIN(dest);
	}
}

void conv2_layer(mat *src, mat *dest){
	float data[8 * 8];
	mat it;
	mat *inter = &it;
	inter->data = data;
	MAT_RESHAPE(inter, 8, 8);
	for(uint i = 0; i < 100; i ++) {
		convolve3d(src, 5, conv2_w[i], inter);
		MAT_CONSTRAIN(dest, i);
		bias2d(inter, conv2_b[i], dest);
		MAT_UNCONSTRAIN(dest);
	}
}

void pool2_layer(mat *src, mat *dest){
	for(uint i = 0; i < 100; i ++) {
		MAT_CONSTRAIN(src, i);
		MAT_CONSTRAIN(dest, i);
		pool(src, 2, 2, dest);
		MAT_UNCONSTRAIN(src);
		MAT_UNCONSTRAIN(dest);
	}
}

void pr_layer(mat *src, mat *dest) {
	// First we need to collapse src to a 1D vector
	MAT_RESHAPE(src, 100 * 4 * 4);

	float data[500];
	mat it;
	mat *inter = &it;
	inter->data = data;
	MAT_RESHAPE(inter, 500);
	sparse_mul_vector(500, pr_w, pr_idx, pr_ptr, src, inter);

	bias1d(inter, pr_b, dest);
}

void relu_layer(mat *src, mat *dest) {
	relu(src, dest);
}

void pred_layer(mat *src, mat *dest) {
	float data[10];
	mat it;
	mat *inter = &it;
	inter->data = data;
	MAT_RESHAPE(inter, 10);
	mul_vector(10, 500, pred_w, src, inter);

	bias1d(inter, pred_b, dest);
}

void convolve2d(mat *src, uint size, float filter[][size], mat *dest) {
	uint rows = MAT_GET_DIM(src, 0);
	uint cols = MAT_GET_DIM(src, 1);
	uint drows = MAT_GET_DIM(dest, 0);
	uint dcols = MAT_GET_DIM(dest, 1);
	for(uint i = 0; i < drows; i ++) {
		for(uint j = 0; j < dcols; j ++) {
			MAT_SET(dest, 0, i, j);
			for(int k = 0; k < size; k ++) {
				int irow_idx = i + k;
				for(int l = 0; l < size; l ++) {
					int icol_idx = j + l;
					if(irow_idx >= 0 && irow_idx < rows && icol_idx >= 0 && 
						icol_idx < cols) {
						float w = MAT_GET(dest, i, j) + 
							MAT_GET(src, irow_idx, icol_idx) * filter[k][l];
						MAT_SET(dest, w, i, j);
					}
				}
			}
		}
	}
}

void convolve3d(mat *src, uint size, float filter[][size][size], mat *dest) {
	uint layers = MAT_GET_DIM(src, 0);
	uint drows = MAT_GET_DIM(dest, 0);
	uint dcols = MAT_GET_DIM(dest, 1);
	for(uint i = 0; i < drows; i ++) {
		for(uint j = 0; j < dcols; j ++) {
			MAT_SET(dest, 0, i, j);
		}
	}
	float data[drows * dcols];
	mat it;
	mat *inter = &it;
	inter->data = data;
	MAT_RESHAPE(inter, drows, dcols);
	for(uint i = 0; i < layers; i++) {
		MAT_CONSTRAIN(src, i);
		convolve2d(src, size, filter[i], inter);
		MAT_UNCONSTRAIN(src);
		for(uint j = 0; j < drows; j ++) {
			for(uint k = 0; k < dcols; k ++) {
				float w = MAT_GET(dest, j, k) + MAT_GET(inter, j, k);
				MAT_SET(dest, w, j, k);
			}
		}
	}
}

void mul_vector(uint rows, uint cols, float mat_data[][cols], mat *vector, 
	mat *dest) {
	for(uint i = 0; i < rows; i ++) {
		MAT_SET(dest, 0, i);
		for(uint j = 0; j < cols; j ++) {
			float w = MAT_GET(dest, i) + mat_data[i][j] * MAT_GET(vector, j); 
			MAT_SET(dest, w, i);
		}
	}
}

void sparse_mul_vector(uint rows, float mat_data[], uint mat_idx[], 
	uint mat_ptr[], mat *vector, mat *dest) {
	for(uint i = 0; i < rows; i ++) {
		MAT_SET(dest, 0, i);
		for(uint j = mat_ptr[i]; j < mat_ptr[i + 1]; j ++) {
			float w = MAT_GET(dest, i) + mat_data[j] * MAT_GET(vector, mat_idx[j]); 
			MAT_SET(dest, w, i);
		}
	}
}

void bias2d(mat *src, float bias, mat *dest) {
	uint rows = MAT_GET_DIM(src, 0); 
	uint cols = MAT_GET_DIM(src, 1);
	for(uint i = 0; i < rows; i ++) {
		for(uint j = 0; j < cols; j ++) {
			MAT_SET(dest, MAT_GET(src, i, j) + bias, i, j);
		}
	}
}

void bias1d(mat *src, float bias[], mat *dest) {
	uint rows = MAT_GET_DIM(src, 0); 
	for(uint i = 0; i < rows; i ++) {
		MAT_SET(dest, MAT_GET(src, i) + bias[i], i);
	}
}

void pool(mat *src, uint size, uint stride, mat *dest) {
	uint rows = MAT_GET_DIM(src, 0);
	uint cols = MAT_GET_DIM(src, 1);
	for(uint i = 0; i < rows; i += stride) {
		for(uint j = 0; j < cols; j += stride) {
			float max = MAT_GET(src, i, j);
			for(uint k = 0; k < size; k ++) {
				for(uint l = 0; l < size; l ++) {
					float val = MAT_GET(src, i + k, j + l);
					if(max < val)
						max = val;
				}
			}
			MAT_SET(dest, max, i / stride, j / stride);
		}
	}
}

void relu(mat *src, mat *dest) {
	uint rows = MAT_GET_DIM(src, 0);
	float max = 0.0;
	for(uint i = 0; i < rows; i ++) {
		max = MAT_GET(src, i);
		MAT_SET(dest, max, i);
		if(max < 0.0)
			MAT_SET(dest, 0.0, i);
	}
}