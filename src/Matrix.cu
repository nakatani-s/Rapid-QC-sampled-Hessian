/* Matrix.cu */
#include "../include/Matrix.cuh" 

void printMatrix(int m, int n, float*A, int lda, const char* name)
{
    for(int row = 0 ; row < m ; row++){
        for(int col = 0 ; col < n ; col++){
            float Areg = A[row + col*lda];
            printf("%s(%d,%d) = %f\n", name, row+1, col+1, Areg);
            //printf("%s[%d] = %f\n", name, row + col*lda, Areg);
        }
    }
}

unsigned int countBlocks(unsigned int a, unsigned int b) {
	unsigned int num;
	num = a / b;
	if (a < b || a % b > 0)
		num++;
	return num;
}

__global__ void setup_Identity_Matrix(float *IdMat)
{
    unsigned int id = threadIdx.x + blockDim.x * blockIdx.x;
    if(threadIdx.x == blockIdx.x)
    {
        IdMat[id] = 1.0f;
        //values[threadIdx.x] = 1.0f;
    }else{
        IdMat[id] = 0.0f;
        //values[threadIdx.x] = 0.0f;
    }
    __syncthreads();
}

__global__ void setup_Identity_Matrix_overMaxThread(float *IdMat, int Ydimention)
{
    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    int iy = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int id = iy * Ydimention + ix;
    if(ix == iy)
    {
        IdMat[id] = 1.0f;
        //values[threadIdx.x] = 1.0f;
    }else{
        IdMat[id] = 0.0f;
        //values[threadIdx.x] = 0.0f;
    }
    __syncthreads();
}
__global__ void multiply_matrix(float *OutMatrix, float voc, float *InMatrix)
{
    unsigned int id = threadIdx.x + blockDim.x * blockIdx.x;
    OutMatrix[id] = voc * InMatrix[id];
    //printf("OutMatrix[%d] == %f = %f * %f\n",id, OutMatrix[id], voc, InMatrix[id]);
    __syncthreads();
}

__global__ void copy_inputSequences(InputVector *outInput, float *temp)
{
    unsigned int id = threadIdx.x + blockDim.x * blockIdx.x;
    for(int i = 0; i < HORIZON; i++){
        outInput[id].Input[i] = temp[i];
    }
    __syncthreads();
}