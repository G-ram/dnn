#include "dnn.h"
#include <stdio.h>

int main() {
	infer(input);
	return 0;
}

float infer(float src[28][28]) {
	float inter1[20][24][24];
	float inter2[20][12][12];
	conv1_layer(src, inter1);
	pool1_layer(inter1, inter2);
	float inter3[100][8][8];
	float inter4[100][4][4];
	conv2_layer(inter2, inter3);
	pool2_layer(inter3, inter4);
	float inter5[500];
	float inter6[500];
	fc3_layer(inter4, inter5);
	relu_layer(inter5, inter6);
	float dest[10];
	pred_layer(inter6, dest);
	for(uint i = 0; i < 10; i ++) {
		printf("Output: %d => %f\n", i, dest[i]);
	}
	return 0.0;
}

void conv1_layer(float src[28][28], float dest[20][24][24]){
	for(uint i = 0; i < 20; i ++) {
		float inter[24][24];
		convolve2d(28, 28, src, 5, conv1_w[i], 24, 24, inter);
		bias2d(24, 24, inter, conv1_b[i], 24, 24, dest[i]);
	}
}

void pool1_layer(float src[20][24][24], float dest[20][12][12]){
	for(uint i = 0; i < 20; i ++) {
		pool(24, 24, src[i], 2, 2, 12, 12, dest[i]);
	}
}

void conv2_layer(float src[20][12][12], float dest[100][8][8]){
	for(uint i = 0; i < 100; i ++) {
		float inter[8][8];
		convolve3d(12, 12, 20, src, 5, conv2_w[i], 8, 8, inter);
		bias2d(8, 8, inter, conv2_b[i], 8, 8, dest[i]);
	}
}

void pool2_layer(float src[100][8][8], float dest[100][4][4]){
	for(uint i = 0; i < 100; i ++) {
		pool(8, 8, src[i], 2, 2, 4, 4, dest[i]);
	}
}

void fc3_layer(float src[100][4][4], float dest[500]) {
	// First we need to collapse src to a 1D vector
	float inter1[100 * 4 * 4][1];
	for(uint i = 0; i < 100; i ++) {
		for(uint j = 0; j < 4; j ++) {
			for(uint k = 0; k < 4; k ++) {
				inter1[i * 4 * 4 + 4 * j + k][0] = src[i][j][k];
			}
		}
	}

	float inter2[500][1];
	mul(1600, 1, inter1, 500, 1600, fc3_w, 500, 1, inter2);

	float inter3[500];
	for(uint i = 0; i < 500; i ++) {
		inter3[i] = inter2[i][0]; 
	}
	bias1d(500, inter3, fc3_b, dest);
}

void relu_layer(float src[500], float dest[500]) {
	relu(500, src, dest);
}

void pred_layer(float src[500], float dest[10]) {
	float inter1[500][1];
	for(uint i = 0; i < 500; i ++) {
		inter1[i][0] = src[i];
	}

	float inter2[10][1];
	mul(500, 1, inter1, 10, 500, pred_w, 500, 1, inter2);

	float inter3[10];
	for(uint i = 0; i < 10; i ++) {
		inter3[i] = inter2[i][0]; 
	}
	bias1d(10, inter3, pred_b, dest);
}

void convolve2d(uint rows, uint cols, float src[][cols], uint size, 
	float filter[][size], uint drows, uint dcols, float dest[][dcols]) {
	int k_half = size / 2;
	for(uint i = 0; i < dcols; i ++) {
		for(uint j = 0; j < drows; j ++) {
			dest[i][j] = 0;
			for(int k = -k_half; k <= k_half; k ++) {
				int frow_idx = k + k_half;
				int irow_idx = i + k;
				for(int l = -k_half; l <= k_half; l ++) {
					int fcol_idx = l + k_half;
					int icol_idx = i + k;
					if(irow_idx >= 0 && irow_idx < rows && icol_idx >= 0 && icol_idx < cols)
						dest[i][j] += src[irow_idx][icol_idx] * filter[frow_idx][fcol_idx];
				}
			}
		}
	}
}

void convolve3d(uint rows, uint cols, uint layers, float src[][rows][cols], 
	uint size, float filter[][size][size], uint drows, uint dcols, 
	float dest[][dcols]) {
	for(uint i = 0; i < drows; i ++) {
		for(uint j = 0; j < dcols; j ++) {
			dest[i][j] = 0;
		}
	}
	for(uint i = 0; i < layers; i++) {
		float inter[drows][dcols];
		convolve2d(rows, cols, src[i], size, filter[i], drows, dcols, inter);
		for(uint j = 0; j < drows; j ++) {
			for(uint k = 0; k < cols; k ++) {
				dest[j][k] += inter[j][k];
			}
		}
	}
}

void mul(uint rows, uint cols, float src[][cols], uint frows, uint fcols, 
	float filter[][fcols], uint drows, uint dcols, float dest[][dcols]) {
	for(uint i = 0; i < cols; i ++) {
		for(uint j = 0; j < fcols; j ++) {
			for(uint k = 0; k < frows; k++) {
				dest[i][j] += src[i][k] * filter[k][j];
			}
		}
	}
}

void bias2d(uint rows, uint cols, float src[][cols], float bias2d, uint drows, 
	uint dcols, float dest[][dcols]) {
	for(uint i = 0; i < cols; i ++) {
		for(uint j = 0; j < rows; j ++) {
			dest[i][j] += bias2d;
		}
	}
}

void bias1d(uint rows, float src[], float bias[], float dest[]) {
	for(uint i = 0; i < rows; i ++) {
		dest[i] = src[i] + bias[i];
	}
}

void pool(uint rows, uint cols, float src[][cols], uint size, 
	uint stride, uint drows, uint dcols, float dest[][dcols]) {
	for(uint i = 0; i < dcols; i += stride) {
		for(uint j = 0; j < drows; j += stride) {
			float max = -1.0;
			for(uint k = 0; k < size; k ++) {
				for(uint l = 0; l < size; l ++) {
					if(max < src[i + k][j + l]) {
						max = src[i + k][j + l];
					}
				}
			}
			dest[i / stride][j / stride] = max;
		}
	}
}

void relu(uint rows, float src[], float dest[]) {
	for(uint i = 0; i < rows; i ++) {
		dest[i] = src[i];
		if(src[i] < 0.0) {
			dest[i] = 0.0;
		}
	}
}