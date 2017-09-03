#include "conv1.h"
#include "conv2.h"
#include "fc3.h"
#include "pred.h"

typedef unsigned int uint;


void convolve(float **src, uint sw, uint sh, float **filter, uint fw, uint fh, 
	uint stride, float **dest);
void mul(float **src, uint sw, uint sh, float **filter, uint fw, uint fh, 
	float **dest);
void bias(float **src, uint sw, uint sh, float bias, float **dest);
void pool(float **src, uint sw, uint sh, uint fw, uint fh, uint stride, 
	float **dest);
