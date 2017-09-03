#include "dnn.h"

int main() {
	return 0;
}

void convolve(float **src, uint sw, uint sh, float **filter, uint fw, uint fh, uint stride, float **dest) {
	uint dw = (sw / fw) * fw;
	uint dh = (sh / fh) * fh;
	for(uint i = 0; i < dh; i += stride) {
		for(uint j = 0; j < dw; j += stride) {
			dest[i][j] = 0;
			for(uint k = 0; k < fh; k ++) {
				for(uint l = 0; l < fw; l ++) {
					dest[i][j] += src[i + k][j + l] * filter[k][l];
				}
			}
		}
	}
}

void mul(float **src, uint sw, uint sh, float **filter, uint fw, uint fh, float **dest) {
	if(sw != fh)
		return;
	
	for(uint i = 0; i < sh; i ++) {
		for(uint j = 0; j < sw; j ++) {
			dest[i][j] += src[i][j] * filter[j][i];
		}
	}
}

void bias(float **src, uint sw, uint sh, float bias, float **dest) {
	for(uint i = 0; i < sh; i ++) {
		for(uint j = 0; j < sw; j ++) {
			dest[i][j] += bias;
		}
	}
}

void pool(float **src, uint sw, uint sh, uint fw, uint fh, uint stride, float **dest) {
	uint dw = (sw / fw) * fw;
	uint dh = (sh / fh) * fh;
	for(uint i = 0; i < dh; i += stride) {
		for(uint j = 0; j < dw; j += stride) {
			float max = -1.0;
			for(uint k = 0; l < fh; k ++) {
				for(uint l = 0; l < fw; l ++) {
					if(max < src[i + k][j + l]) {
						max = src[i + k];
					}
				}
			}
			dest[i / stride][j / stride] = max;
		}
	}
}