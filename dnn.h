#include "conv1.h"
#include "conv2.h"
#include "fc3.h"
#include "pred.h"

#define IMG_WIDTH 28
#define IMG_HEIGHT 28

typedef unsigned int uint;

float input[IMG_WIDTH][IMG_HEIGHT] = {};

float infer(float src[28][28]);
void conv1_layer(float src[28][28], float dest[20][24][24]);
void pool1_layer(float src[20][24][24], float dest[20][12][12]);
void conv2_layer(float src[20][12][12], float dest[100][8][8]);
void pool2_layer(float src[100][8][8], float dest[100][4][4]);
void fc3_layer(float src[100][4][4], float dest[500]);
void relu_layer(float src[500], float dest[500]);
void pred_layer(float src[500], float dest[10]);

void convolve2d(uint rows, uint cols, float src[][cols], uint frows, uint fcols, 
	float filter[][fcols], uint drows, uint dcols, float dest[][dcols]);
void convolve3d(uint rows, uint cols, uint layers, float src[][rows][cols], 
	uint frows, uint fcols, uint flayers, float filter[][frows][fcols], 
	uint drows, uint dcols, float dest[][dcols]);
void mul(uint rows, uint cols, float src[][cols], uint frows, uint fcols, 
	float filter[][fcols], uint drows, uint dcols, float dest[][dcols]);
void bias2d(uint rows, uint cols, float src[][cols], float bias, uint drows,
	uint dcols, float dest[][dcols]);
void bias1d(uint rows, float src[], float bias[], float dest[]);
void pool(uint rows, uint cols, float src[][cols], uint frows, uint fcols,
	uint stride, uint drows, uint dcols, float dest[][dcols]);
void relu(uint rows, float src[], float dest[]);
