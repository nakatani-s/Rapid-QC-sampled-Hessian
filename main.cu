/*
  Least Squares Method: LSM 最小二乗法
  QuadraticHyperPlane:  QHP ２次超平面
*/ 

#include<iostream>
#include <stdio.h>
#include <fstream>
#include <math.h>
#include <cuda.h>
#include <time.h>
#include <curand_kernel.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <errno.h>
#include <string.h>
#include <unistd.h>
#include <iomanip>

#include "include/params.cuh"
#include "include/init.cuh"
#include "include/DataStructure.cuh"
#include "include/MCMPC.cuh"
#include "include/LSM_QuadHyperPlane.cuh"
#include "include/Matrix.cuh"
#include "include/costFunction.cuh"

#define CHECK(call)                                                  \
{                                                                    \
    const cudaError_t error = call;                                  \
    if (error != cudaSuccess)                                        \
    {                                                                \
        printf("Error: %s:%d, ", __FILE__, __LINE__);                \
        printf("code:%d, reason: %s\n", error,                       \
                cudaGetErrorString(error));                          \
        exit(1);                                                     \
    }                                                                \
}
#define CHECK_CUBLAS(call,str)                                                        \
{                                                                                     \
    if ( call != CUBLAS_STATUS_SUCCESS)                                               \
    {                                                                                 \
        printf("CUBLAS Error: %s : %s %d\n", str, __FILE__, __LINE__);                \
        exit(1);                                                                      \
    }                                                                                 \
}

#define CHECK_CUSOLVER(call,str)                                                      \
{                                                                                     \
    if ( call != CUSOLVER_STATUS_SUCCESS)                                             \
    {                                                                                 \
        printf("CUBLAS Error: %s : %s %d\n", str, __FILE__, __LINE__);                \
        exit(1);                                                                      \
    }                                                                                 \
}





int main(int argc, char **argv)
{
    /* 行列演算ライブラリを使用するために宣言 */
    cusolverDnHandle_t cusolverH = NULL;
    // cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR; 
    CHECK_CUSOLVER( cusolverDnCreate(&cusolverH),"Failed to Create cusolver handle");

    cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;

    /* データ書き込み用ファイルの定義 */
    FILE *fp;
    time_t timeValue;
    struct tm *timeObject;
    time( &timeValue );
    timeObject = localtime( &timeValue );
    char filename1[35];
    sprintf(filename1,"data_system_%d%d_%d%d.txt",timeObject->tm_mon + 1, timeObject->tm_mday, timeObject->tm_hour,timeObject->tm_min);
    fp = fopen(filename1,"w");


    /* ホスト・デバイスで使用するベクトルの宣言 */
    float hostParams[DIM_OF_PARAMETERS], hostState[DIM_OF_STATES], hostConstraint[NUM_OF_CONSTRAINTS], hostWeightMatrix[DIM_OF_WEIGHT_MATRIX];
    float *deviceParams, *deviceState, *deviceConstraint, *deviceWeightMatrix;
    initialize_host_vector(hostParams, hostState, hostConstraint, hostWeightMatrix);
    cudaMalloc(&deviceParams, sizeof(float) * DIM_OF_PARAMETERS);
    cudaMalloc(&deviceState, sizeof(float) * DIM_OF_STATES);
    cudaMalloc(&deviceConstraint, sizeof(float) * NUM_OF_CONSTRAINTS);
    cudaMalloc(&deviceWeightMatrix, sizeof(float) * DIM_OF_WEIGHT_MATRIX);
    cudaMemcpy(deviceParams, hostParams, sizeof(float) * DIM_OF_PARAMETERS, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceState, hostState, sizeof(float) * DIM_OF_STATES, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceConstraint, hostConstraint, sizeof(float) * NUM_OF_CONSTRAINTS, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceWeightMatrix, hostWeightMatrix, sizeof(float)* DIM_OF_WEIGHT_MATRIX, cudaMemcpyHostToDevice);

    /* GPUの設定用パラメータ */
    unsigned int numBlocks, randomBlocks, randomNums, Blocks, dimHessian, numUnknownParamQHP, numUnknownParamHessian;
    unsigned int paramsSizeQuadHyperPlane;
    randomNums = NUM_OF_SAMPLES * (DIM_OF_INPUT + 1) * HORIZON;
    randomBlocks = countBlocks(randomNums, THREAD_PER_BLOCKS);
    numBlocks = countBlocks(NUM_OF_SAMPLES, THREAD_PER_BLOCKS);
    Blocks = numBlocks;
    dimHessian = HORIZON * HORIZON;
    // numUnknownParamQHP = count_QHP_Parameters( HORIZON );
    numUnknownParamQHP = sizeOfParaboloidElements;
    numUnknownParamHessian = numUnknownParamQHP - (HORIZON + 1);
    paramsSizeQuadHyperPlane = numUnknownParamQHP; //ホライズンの大きさに併せて、局所サンプルのサイズを決定
    paramsSizeQuadHyperPlane = paramsSizeQuadHyperPlane + addTermForLSM;
    dim3 block(2,2);
    dim3 grid((numUnknownParamQHP + block.x - 1)/ block.x, (numUnknownParamQHP + block.y -1) / block.y);
    printf("#NumBlocks = %d\n", numBlocks);
    printf("#NumBlocks = %d\n", numUnknownParamQHP);

    /* GPUで乱数生成するために使用する乱数の種 */
    curandState *deviceRandomSeed;
    cudaMalloc((void **)&deviceRandomSeed, randomNums * sizeof(curandState));
    setup_kernel<<<NUM_OF_SAMPLES, (DIM_OF_INPUT + 1) * HORIZON>>>(deviceRandomSeed, rand());
    cudaDeviceSynchronize();

    /* sort用入力格納構造体の宣言と初期化 */
    InputVector *deviceInputSeq, *hostInputSeq, *deviceEliteInputSeq;
    hostInputSeq = (InputVector*)malloc(sizeof(InputVector) * NUM_OF_ELITES);
    cudaMalloc(&deviceEliteInputSeq, sizeof(InputVector) * NUM_OF_ELITES);
    cudaMalloc(&deviceInputSeq, sizeof(InputVector) * NUM_OF_SAMPLES);
    // init_Input_vector<<<NUM_OF_SAMPLES, 1>>>(deviceInputSeq, 0.0f);

    /* 2次超平面のパラメータ行列/ベクトル　（←最適値計算用に準備） */
    float *Hessian, *HessElements, *transGmatrix, *Hvector, *invGmHessSsymm /*, *Grad*/;
    cudaMalloc(&Hessian, sizeof(float) * dimHessian );
    cudaMalloc(&transGmatrix, sizeof(float) * dimHessian); /* Ans = -2 * G^T * Hessian * Hvector の　G^T  */
    cudaMalloc(&Hvector, sizeof(float) * HORIZON ); /* Ans = -2 * G^T * Hessian * Hvector の　Hvector */
    cudaMalloc(&invGmHessSsymm, sizeof(float) * dimHessian);
    // cudaMalloc(&HessElements, sizeof(float) * numUnknownParamHessian);
    cudaMalloc(&HessElements, sizeof(float) * numUnknownParamQHP );
    // cudaMalloc(&Grad, sizeof(float) * HORIZON);
    /* 最小二乗法で2次超平面を求める際に使用 */
    float *Gmatrix, *invGmatrix, *Rvector, *ansRvector;
    CHECK(cudaMalloc(&Rvector, sizeof(float) * numUnknownParamQHP));
    CHECK(cudaMalloc(&ansRvector, sizeof(float) * numUnknownParamQHP));
    CHECK(cudaMalloc(&Gmatrix, sizeof(float) * numUnknownParamQHP * numUnknownParamQHP));
    CHECK(cudaMalloc(&invGmatrix, sizeof(float) * numUnknownParamQHP * numUnknownParamQHP) ); //elementsSize_QuadHyperPlaneMatrix = paramsSize_QuadHyperPlane * paramsSize_QuadHyperPlane
    //assert(cudaSuccess == cudaStat2);
    QuadHyperPlane *deviceQuadHyPl;
    cudaMalloc(&deviceQuadHyPl, sizeof(QuadHyperPlane) * paramsSizeQuadHyperPlane); //当面はブロック数分リサンプル　( HORIZON < Blocks < GPUコア数 で設計)
    unsigned int qhpBlocks;
    // qhpBlocks = countBlocks(numUnknownParamQHP, THREAD_PER_BLOCKS);
    qhpBlocks = countBlocks(paramsSizeQuadHyperPlane, THREAD_PER_BLOCKS);
    printf("#qhpblocks = %d\n", qhpBlocks);
    // float *KVALUE_MATRIX, *HESSIAN_MATRIX;
    // KVALUE_MATRIX = (float *)malloc(sizeof(float)*numUnknownParamQHP * numUnknownParamQHP);
    // HESSIAN_MATRIX = (float *)malloc(sizeof(float)*dimHessian);
    //KVALUE_MATRIX = (float *)malloc(sizeof(float)*dimHessian);
    // 行列演算ライブラリ使用用に定義
    const int m_RMatrix = numUnknownParamQHP;
    printf("#NumBlocks = %d\n", m_RMatrix);
    // const int lda_RMatrix = m_RMatrix;
    int work_size, work_size_season2;
    float *work_space, *work_space_season2;
    int *devInfo;
    int *devInfo_season2;
    cudaMalloc ((void**)&devInfo_season2, sizeof(int));
    cublasHandle_t handle_cublas = 0;
    cublasCreate(&handle_cublas);
    float alpha;
    float beta;
    alpha = 1.0f;
    beta = 0.0f;
    cudaMalloc ((void**)&devInfo, sizeof(int));


    /* thrust使用のためのホスト/デバイスベクトル */
    thrust::host_vector<int> indices_host_vec( NUM_OF_SAMPLES );
    thrust::device_vector<int> indices_device_vec = indices_host_vec;
    thrust::host_vector<float> sort_key_host_vec( NUM_OF_SAMPLES );
    thrust::device_vector<float> sort_key_device_vec = sort_key_host_vec; 
    
    /* 推定入力のプロット・データ転送用 */
    float *hostData, *deviceData;
    hostData = (float *)malloc(sizeof(float) * HORIZON);
    cudaMalloc(&deviceData, sizeof(float) * HORIZON);
    for(int i = 0; i < HORIZON; i++){
        hostData[i] = 0.0f;
    }
    cudaMemcpy(deviceData, hostData, sizeof(float) * HORIZON, cudaMemcpyHostToDevice);
    
    // float variance;
    /* 制御ループ */
    float est_input = 0.0f;
    float MCMPC_U, Proposed_U;
    float costFromMCMPC, costFromQHPMethod;
    costFromMCMPC = 0.0f;
    costFromQHPMethod = 0.0f;
    float vars;

    int counter;
    float process_gpu_time, procedure_all_time;
    clock_t start_t, stop_t;
    cudaEvent_t start, stop;

    for(int t = 0; t < TIME; t++){
        shift_Input_vec( hostData );
        cudaMemcpy(deviceData, hostData, sizeof(float) * HORIZON, cudaMemcpyHostToDevice);
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start, 0);
        start_t = clock();
        for(int repeat = 0; repeat < ITERATIONS; repeat++){
            /* サンプルベースニュートンメソッドの初期値を決定するMCMPC */
            vars = powf(0.95,repeat) * variance; 
            MCMPC_Crat_and_SinglePole<<<numBlocks, THREAD_PER_BLOCKS>>>(deviceState, deviceRandomSeed, deviceData, deviceInputSeq, vars, deviceParams, deviceConstraint, deviceWeightMatrix,
                thrust::raw_pointer_cast( sort_key_device_vec.data() ));
            /*MCMPC_Simple_NonLinear_Example<<<numBlocks, THREAD_PER_BLOCKS>>>(deviceState, deviceRandomSeed, deviceData, deviceInputSeq, variance, deviceParams, deviceConstraint, deviceWeightMatrix,
                thrust::raw_pointer_cast( sort_key_device_vec.data() ));*/
            cudaDeviceSynchronize();
            thrust::sequence(indices_device_vec.begin(), indices_device_vec.end());
            thrust::sort_by_key(sort_key_device_vec.begin(), sort_key_device_vec.end(), indices_device_vec.begin());
            
            /* エリートサンプル分の入力・コスト値をコールバックする関数 */ 
            callback_elite_sample<<<NUM_OF_ELITES, 1>>>(deviceEliteInputSeq, deviceInputSeq, thrust::raw_pointer_cast(indices_device_vec.data()));
            cudaDeviceSynchronize();
            cudaMemcpy(hostInputSeq, deviceEliteInputSeq, sizeof(InputVector) * NUM_OF_ELITES, cudaMemcpyDeviceToHost);
            weighted_mean(hostInputSeq, NUM_OF_ELITES, hostData);
            MCMPC_U = hostData[0];
            
            CHECK(cudaMemcpy(deviceData, hostData, sizeof(float) * HORIZON, cudaMemcpyHostToDevice));
            
            costFromMCMPC = calc_Cost_Cart_and_SinglePole(hostData, hostState, hostParams, hostConstraint, hostWeightMatrix);
            printf("%dth MCMPC estimation ended\n", t*repeat);
            

            if(repeat == ITERATIONS - 1){
                /* 推定値近傍をサンプル・評価する関数 */
                MCMPC_Crat_and_SinglePole<<<numBlocks, THREAD_PER_BLOCKS>>>(deviceState, deviceRandomSeed, deviceData, deviceInputSeq, neighborVar, deviceParams, deviceConstraint, deviceWeightMatrix,
                    thrust::raw_pointer_cast( sort_key_device_vec.data() ));
                /*MCMPC_Simple_NonLinear_Example<<<numBlocks, THREAD_PER_BLOCKS>>>(deviceState, deviceRandomSeed, deviceData, deviceInputSeq, neighborVar, deviceParams, deviceConstraint, deviceWeightMatrix,
                    thrust::raw_pointer_cast( sort_key_device_vec.data() ));*/
                cudaDeviceSynchronize();
                thrust::sequence(indices_device_vec.begin(), indices_device_vec.end());
                thrust::sort_by_key(sort_key_device_vec.begin(), sort_key_device_vec.end(), indices_device_vec.begin());
                /*device_QuadHyPlに，最小二乗法の左辺(column)と右辺の行列(テンソル積)計算用のベクトル(tensor)を格納する*/
                printf("hoge here l 185\n");
                LSM_QHP_make_tensor_vector<<<qhpBlocks, THREAD_PER_BLOCKS>>>(deviceQuadHyPl, deviceInputSeq, thrust::raw_pointer_cast( indices_device_vec.data() ));
                printf("hoge here l 187\n");
                cudaDeviceSynchronize();
            /* Gmatrix に正規行列（最小二乗法で使用する逆行列の逆行列）*/ 
                if(numUnknownParamQHP > 1024){
                    LSM_QHP_make_regular_matrix_over_ThreadPerBlockLimit<<<grid,block>>>(Gmatrix, deviceQuadHyPl, paramsSizeQuadHyperPlane, numUnknownParamQHP);
                }else{
                    LSM_QHP_make_regular_matrix<<<numUnknownParamQHP,numUnknownParamQHP>>>(Gmatrix, deviceQuadHyPl, paramsSizeQuadHyperPlane);
                }
                cudaDeviceSynchronize();
                printf("hoge here l 193\n");

                // 最小二乗法の結果（ヘシアンの要素＋勾配＋定数）
                LSM_QHP_make_regular_vector<<<numUnknownParamQHP,1>>>(Rvector, deviceQuadHyPl, paramsSizeQuadHyperPlane);
                cudaDeviceSynchronize();

            /* Gmatrixの逆行列を計算 */
                CHECK_CUSOLVER( cusolverDnSpotrf_bufferSize(cusolverH, uplo, m_RMatrix, Gmatrix, m_RMatrix, &work_size), "Failed to get bufferSize");
                CHECK( cudaMalloc((void**)&work_space, sizeof(float)*work_size));
	            //cudaGetErrorString(cudaStat1);
                //assert(cudaSuccess == cudaStat1);
                CHECK_CUSOLVER(cusolverDnSpotrf(cusolverH, uplo, m_RMatrix, Gmatrix, m_RMatrix , work_space, work_size, devInfo), "Failed to inverse operation for G");
                
                // 逆行列を取得するための単位行列の生成
                if(numUnknownParamQHP > 1024){
                    setup_Identity_Matrix_overMaxThread<<<grid, block>>>(invGmatrix, numUnknownParamQHP); 
                }else{
                    setup_Identity_Matrix<<<numUnknownParamQHP, numUnknownParamQHP>>>(invGmatrix); // invGmatrixを単位行列に変換
                }
                cudaDeviceSynchronize();
                CHECK_CUSOLVER( cusolverDnSpotrs(cusolverH, uplo, m_RMatrix, m_RMatrix , Gmatrix, m_RMatrix, invGmatrix, m_RMatrix, devInfo), "Failed to get inverse Matrix G");

                //LSM_QHP_get_reslt_all_elements<<<numUnknownParamQHP,1>>>(HessElements, invGmatrix, Rvector);
                /* 最小二乗法の行列演算　ansRvector = invGmatrix * Rvector を計算 */ 
                CHECK_CUBLAS( cublasSgemv(handle_cublas, CUBLAS_OP_N, m_RMatrix, m_RMatrix, &alpha, invGmatrix, m_RMatrix, Rvector, 1, &beta, ansRvector , 1), "Failed to get Estimate Input Sequences");

                //assert(  cublas_status == CUBLAS_STATUS_SUCCESS );
                LSM_QHP_get_reslt_all_elements<<<numUnknownParamHessian,1>>>(HessElements, ansRvector); //numUnknownParamHessian これが大きすぎる?
                cudaDeviceSynchronize();
                LSM_QHP_get_Hessian_Result<<<HORIZON, HORIZON>>>( Hessian, HessElements);
                cudaDeviceSynchronize();
                LSM_QHP_transpose<<<HORIZON, HORIZON>>>(transGmatrix, Hessian);
                cudaDeviceSynchronize();
                LSM_QHP_make_symmetric<<<HORIZON, HORIZON>>>(transGmatrix, Hessian);
                // cudaMemcpy(HESSIAN_MATRIX, transGmatrix, sizeof(float) * dimHessian, cudaMemcpyDeviceToHost);
                // printMatrix(HORIZON,HORIZON,HESSIAN_MATRIX, HORIZON, "HESSIAN");
                // ヘッシアンの計算まで終了
                //LSM_Hessian_To_Positive_Symmetric<<<HORIZON, HORIZON>>>(transGmatrix);

                /* -2*Hessian * b^T の b^Tベクトルを作成 (Hvector　←　b^T) */
                LSM_QHP_make_bVector<<<HORIZON, 1>>>(Hvector, ansRvector, numUnknownParamHessian);

                multiply_matrix<<<HORIZON, HORIZON>>>(Hessian, 2.0f, transGmatrix);
                CHECK_CUSOLVER(cusolverDnSpotrf_bufferSize(cusolverH, uplo, HORIZON, Hessian, HORIZON, &work_size_season2),"Failed to get bufferSize of Hessian");
                CHECK( cudaMalloc((void**)&work_space_season2, sizeof(float)*work_size_season2) );
  
                CHECK_CUSOLVER(cusolverDnSpotrf(cusolverH, uplo, HORIZON, Hessian, HORIZON, work_space_season2, work_size_season2, devInfo_season2), "Failed to inverse operation");
            
                setup_Identity_Matrix<<<HORIZON, HORIZON>>>(invGmHessSsymm);
                cudaDeviceSynchronize();
                CHECK_CUSOLVER(cusolverDnSpotrs(cusolverH, uplo, HORIZON, HORIZON, Hessian, HORIZON, invGmHessSsymm, HORIZON, devInfo_season2), "Failed to get inverse Matrix of H");
                // cudaMemcpy(HESSIAN_MATRIX, invGmHessSsymm, sizeof(float) * dimHessian, cudaMemcpyDeviceToHost);
                multiply_matrix<<<HORIZON, HORIZON>>>(transGmatrix, -1.0f, invGmHessSsymm);

                copy_inputSequences<<<numBlocks, THREAD_PER_BLOCKS>>>(deviceInputSeq, deviceData);
                CHECK_CUBLAS(cublasSgemv(handle_cublas, CUBLAS_OP_N, HORIZON, HORIZON, &alpha, transGmatrix, HORIZON, Hvector, 1, &beta,  deviceData, 1),"Failed to get Result");
                //cublas_status = cublasSgemv(handle_cublas, CUBLAS_OP_N, HORIZON, HORIZON, &alpha, invGmHessSsymm, HORIZON, Hvector, 1, &beta,  deviceData, 1);
                cudaMemcpy(hostData, deviceData, sizeof(float) * HORIZON, cudaMemcpyDeviceToHost);
                //costFromQHPMethod = calc_Cost_Simple_NonLinear_Example(hostData, hostState,  hostParams, hostWeightMatrix);
                costFromQHPMethod = calc_Cost_Cart_and_SinglePole(hostData, hostState, hostParams, hostConstraint, hostWeightMatrix);
                Proposed_U = hostData[0];
            }

        }
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&process_gpu_time, start, stop);
        stop_t = clock();
        procedure_all_time = stop_t - start_t;
        // 推定入力列の先頭をコピー
        if(costFromMCMPC < costFromQHPMethod || isnan(costFromQHPMethod)){
	        est_input = MCMPC_U;
            counter = 0;
	    }else{
            est_input = Proposed_U;
            counter = 1;
        }
        Runge_kutta_45_for_Secondary_system(hostState, est_input, hostParams, interval);
        /*float hostDiffState[DIM_OF_STATES] = { };
        calc_nonLinear_example(hostState, est_input, hostParams, hostDiffState);
        for(int k = 0; k < DIM_OF_STATES; k++){
            hostState[k] = hostState[k] + (interval * hostDiffState[k]);
        }*/
        cudaMemcpy(deviceState, hostState, sizeof(float) * DIM_OF_STATES, cudaMemcpyHostToDevice);
        fprintf(fp,"%f %f %f %f %f %f %f %f %f %f %f %f %f %d\n", interval * t, est_input, MCMPC_U, Proposed_U, hostState[0], hostState[1], hostState[2], hostState[3], costFromMCMPC, costFromQHPMethod, costFromMCMPC - costFromQHPMethod, process_gpu_time/1000,procedure_all_time / CLOCKS_PER_SEC, counter);
        printf("u == %f MCMPC == %f  Proposed == %f  MCMPC - Proposed == %f\n", est_input,  costFromMCMPC, costFromQHPMethod, costFromMCMPC - costFromQHPMethod);
    }

    if(cusolverH) cusolverDnDestroy(cusolverH);
    if(handle_cublas) cublasDestroy(handle_cublas);
    fclose(fp);
    cudaDeviceReset();
    printf("%s\n", cudaGetErrorString(cudaGetLastError()));
    return 0;
}
