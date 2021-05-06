/*
params.cuh
*/ 
// experiment1 for simple_example from Ohtsuka's book
#include <math.h>
#ifndef PARAMS_CUH
#define PARAMS_CUH

/*#define TIME 400
#define ITERATIONS 1
#define HORIZON 20
#define DIM_OF_PARAMETERS 2
#define DIM_OF_STATES 2
#define NUM_OF_CONSTRAINTS 2
#define DIM_OF_WEIGHT_MATRIX 3
#define DIM_OF_INPUT 1*/

// For Control Cart and Single Pole
#define TIME 700
#define ITERATIONS 10
#define HORIZON 25  //50
#define DIM_OF_PARAMETERS 7
#define DIM_OF_STATES 4
#define NUM_OF_CONSTRAINTS 4
#define DIM_OF_WEIGHT_MATRIX 5
#define DIM_OF_INPUT 1

/* NUM_OF_SAMPLES について HORIZON^2 * (2/3) + 10程度に設定  ←　これ以下だと、２次曲面フィットで特異行列を扱うことになる　*/
#define NUM_OF_SAMPLES 8000
#define NUM_OF_ELITES  13// [4+3ln(20)]/2程度を確保　（←CMA−ESの類推より）
#define THREAD_PER_BLOCKS 10


const int sizeOfParaboloidElements = 351; //1326
const int addTermForLSM = 1149; //sizeOfParaboloidElements + addTermForLSM = THREAD_PER_BLOCKSの定数倍になるように加算する項  4000 - sizeOfParaboloidElements くらい
const float neighborVar = 0.8;
const float interval = 0.01;
const float variance = 2.0;
const float invBarrier = 500;

#endif
// const int 
