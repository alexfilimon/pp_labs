#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include <omp.h>
#include <cmath>
#include <random>

using namespace std;

// #define N 32

#define EPS 0.5;
#define CRITICAL_COUNT 50;

// ----------------
// |   HELPERS    |
// ----------------

double dRand(double fMin, double fMax)
{
    double f = (double)rand() / RAND_MAX;
    return fMin + f * (fMax - fMin);
}

void displayMatrix(int matrixSizeDepth, int matrixSizeWidth, float* matrix)
{
    for (int i = 0; i < matrixSizeDepth; i++)
    {
        for (int j = 0; j < matrixSizeWidth; j++)
            cout << matrix[i * matrixSizeDepth + j] << " ";
        cout << endl;
    }
}

void displayVector(int vectorSize, float* vector)
{
    for (int i = 0; i < vectorSize; i++)
        cout << vector[i] << " ";
    cout << endl;
}

// -----------------
// | Initializaion |
// -----------------

void initRandom(int vectorSize, float* vector)
{
    minstd_rand rand(clock());
    uniform_int_distribution<> distribution(0.0, 360.0);
    for (int i = 0; i < vectorSize; i++)
        // vector[i] = 1 + rand() % 10 * 1.0;
        vector[i] = distribution(rand) * 0.0174533;
}

void initRandom(int matrixSizeDepth, int matrixSizeWidth, float* matrix)
{
    minstd_rand rand(clock());
    uniform_int_distribution<> distribution(0.0, 360.0);

    for (int i = 0; i < matrixSizeDepth; i++)
        for (int j = 0; j < matrixSizeWidth; j++)
        {
            // matrix[i * matrixSizeDepth + j] = 1 + rand() % 10 * 1.0;
            matrix[i * matrixSizeDepth + j] = distribution(rand) * 0.0174533;
            // if (i == j) matrix[i * matrixSizeDepth + j] *= matrixSizeDepth;
        }

    for (int i = 0; i < matrixSizeDepth; i++)
    {
        double sum = 0;
        for (int j = 0; j < matrixSizeWidth; j++)
        {
            sum += matrix[i * matrixSizeDepth + j];
        }

        matrix[i * matrixSizeDepth + i] = sum * dRand(1.1, 3.0);
    }

}

void initNull(int vectorSize, float* vector)
{
    for (int i = 0; i < vectorSize; i++)
        vector[i] = 0.0;
}

void initNull(int matrixSizeDepth, int matrixSizeWidth, float* matrix)
{
    for (int i = 0; i < matrixSizeDepth; i++)
        for (int j = 0; j < matrixSizeWidth; j++)
            matrix[i * matrixSizeDepth + j] = 0.0;
}



void initAlpha(int matrixSize, float* matrix, float* result)
{
    initNull(matrixSize, matrixSize, result);
    for (int i = 0; i < matrixSize; i++)
        for (int j = 0; j < matrixSize; j++)
            if (i == j)
                result[i * matrixSize + i] = 0.0;
            else
                result[i * matrixSize + j] = -(double)matrix[i * matrixSize + j] / (double)matrix[i * matrixSize + i];
}

void initBeta(int vectorSize, float* matrix, float* b, float* result)
{
    initNull(vectorSize, result);
    for (int i = 0; i < vectorSize; i++)
        result[i] = b[i] / matrix[i * vectorSize + i];
}

void sequentialMultiMatrixVector(int vectorSize, float* matrix, float* vector, float* result)
{
    initNull(vectorSize, result);
    for (int i = 0; i < vectorSize; i++)
        for (int j = 0; j < vectorSize; j++)
            result[i] += matrix[i * vectorSize + j] * vector[j];
}

void init(int matrixSize, float* matrix, float* prev, float* curr, float* alpha, float* beta)
{
    // initialize X
    initRandom(matrixSize, curr);

    // cout << "X: \n";
    // displayVector(matrixSize, curr);

    // initialize A (matrix)
    initRandom(matrixSize, matrixSize, matrix);

    // cout << "A: \n";
    // displayMatrix(matrixSize, matrixSize, matrix);

    float* b = new float[matrixSize];
    sequentialMultiMatrixVector(matrixSize, matrix, curr, b);

    // cout << "B: \n";
    // displayVector(matrixSize, b);

    initAlpha(matrixSize, matrix, alpha);

    // cout << "alpha: \n";
    // displayMatrix(matrixSize, matrixSize, alpha);

    initBeta(matrixSize, matrix, b, beta);

    // cout << "beta: \n";
    // displayVector(matrixSize, beta);

    initNull(matrixSize, prev);
    for (int i = 0; i < matrixSize; i++)
    {
        prev[i] = beta[i];
    }
    initNull(matrixSize, curr);

    // cout << "prev: \n";
    // displayVector(matrixSize, prev);

    // cout << "curr: \n";
    // displayVector(matrixSize, curr);

}

// -----------------
// |   Sequental    |
// -----------------

bool sequentialIsComplete(int vectorSize, float* prev, float* curr)
{
    bool flag = false;
    float error = 0.0, eps = EPS;
    for (int i = 0; i < vectorSize; i++)
        error += abs(curr[i] - prev[i]);
    if (error < eps)
        flag = true;
    // cout << "\nerror: " << error;
    return flag;
}

void sequentialSubVectors(int vectorSize, float* vectorL, float* vectorR, float* result)
{
    for (int i = 0; i < vectorSize; i++)
        result[i] = vectorL[i] + vectorR[i];
}

void sequentialCopyVectors(int vectorSize, float* vector, float* result)
{
    for (int i = 0; i < vectorSize; i++)
        result[i] = vector[i];
}

// -----------------
// |  PARALLEL OMP |
// -----------------

void parallelOpenMPMultiMatrixVector(int vectorSize, float* matrix, float* vector, float* result)
{
#pragma omp parallel for
    for (int i = 0; i < vectorSize; i++)
    {
        result[i] = 0;
    }

#pragma omp parallel for
    for (int i = 0; i < vectorSize; i++)
    {
#pragma omp parallel for
        for (int j = 0; j < vectorSize; j++)
            result[i] += matrix[i * vectorSize + j] * vector[j];
    }
}

void parallelOpenMPSubVectors(int vectorSize, float* vectorL, float* vectorR, float* result)
{
#pragma omp parallel for
    for (int i = 0; i < vectorSize; i++)
        result[i] = vectorL[i] + vectorR[i];
}

void parallelOpenMPCopyVectors(int vectorSize, float* vector, float* result)
{
#pragma omp parallel for
    for (int i = 0; i < vectorSize; i++)
        result[i] = vector[i];
}

bool parallelOpenMPIsComplete(int vectorSize, float* prev, float* curr)
{
    bool flag = false;
    float error = 0.0, eps = EPS;
#pragma omp parallel for
    for (int i = 0; i < vectorSize; i++)
        error += abs(curr[i] - prev[i]);
    if (error < eps)
    {
        cout << "Finished successfully\n";
        flag = true;
    }
    return flag;
}

// -----------------
// | PARALLEL CUDA |
// -----------------

__global__ void parallelCudaIsComplete(float* prev, float* curr, bool* isGood, int vectorSize)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float error = abs(curr[i] - prev[i]), eps = EPS;
    // for (int i = 0; i < vectorSize; i++)
    //     error += abs(curr[i] - prev[i]);
    if (error > eps)
        *isGood = false;
    // cout << "error: " << error << "\n";
}

__global__ void parallelCudaSum(float* vector, float* sum, int vectorSize)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < vectorSize)
        sum[0] += vector[i];
}

__global__ void parallelCudaSubAbsVector(float* vector1, float* vector2, float* result, int vectorSize)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < vectorSize)
        result[i] = fabs(vector1[i] - vector2[i]);
}

__global__ void parallelCudaMultiMatrixVectorKernel(float* matrix, float* vector, float* result, int vectorSize)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float temp = 0;
    if (i < vectorSize)
        for (int j = 0; j < vectorSize; j++)
            temp += matrix[i * vectorSize + j] * vector[j];
    result[i] = temp;
}

__global__ void parallelCudaSubVectorsKernel(float* vectorL, float* vectorR, float* result, int vectorSize)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < vectorSize)
        result[i] = vectorL[i] + vectorR[i];
}

__global__ void parallelCudaCopyVectorsKernel(float* vector, float* result, int vectorSize)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < vectorSize)
        result[i] = vector[i];
}

// -----------------
// |  CALCULATIONS |
// -----------------

void parallelOpenMPCalculate(int vectorSize, float* alpha, float* beta, float* prev, float* curr)
{
    int count = 0, criticalCount = CRITICAL_COUNT;
    for (count = 0; count < criticalCount; count++)
    {
        parallelOpenMPMultiMatrixVector(vectorSize, alpha, prev, curr);
        parallelOpenMPSubVectors(vectorSize, beta, curr, curr);
        if (parallelOpenMPIsComplete(vectorSize, prev, curr)) break;
        parallelOpenMPCopyVectors(vectorSize, curr, prev);
    }
    cout << "Parallel OpenMP calculation completed with count: " << count;
}

void sequentialCalculate(int vectorSize, float* alpha, float* beta, float* prev, float* curr)
{
    int count = 0, criticalCount = CRITICAL_COUNT;
    for (count = 0; count < criticalCount; count++)
    {
        // cout << "------ iteration: " << count << " -------";

        // cout << "\nalpha: \n";
        // displayMatrix(vectorSize, vectorSize, alpha);

        // cout << "\nprev: \n";
        // displayVector(vectorSize, prev);

        sequentialMultiMatrixVector(vectorSize, alpha, prev, curr);

        // cout << "\ncurr(alpha*prev): \n";
        // displayVector(vectorSize, curr);

        sequentialSubVectors(vectorSize, beta, curr, curr);

        // cout << "\nalpha: \n";
        // displayMatrix(vectorSize, vectorSize, alpha);

        // cout << "\nbeta: \n";
        // displayVector(vectorSize, beta);

        // cout << "\ncurr(alpha*prev+beta): \n";
        // displayVector(vectorSize, curr);

        if (sequentialIsComplete(vectorSize, prev, curr)) break;
        sequentialCopyVectors(vectorSize, curr, prev);

        // cout << "\ncurr(old): \n";
        // displayVector(vectorSize, curr);

        // cout << "\nprev(should be equal curr): \n";
        // displayVector(vectorSize, prev);

        // cout << "\n----------------\n\n";

        // initNull(vectorSize, curr);
    }
    cout << "\ncompleted with count: " << count;
}

void parallelCudaCalculate(int vectorSize, float* alpha, float* beta, float* prev, float* curr)
{
    float* dev_alpha;
    float* dev_beta;
    float* dev_prev;
    float* dev_curr;
    float* error;
    // bool* isGood;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaMalloc((void**)&dev_alpha, (vectorSize * vectorSize) * sizeof(float));
    cudaMalloc((void**)&dev_beta, vectorSize * sizeof(float));
    cudaMalloc((void**)&dev_prev, vectorSize * sizeof(float));
    cudaMalloc((void**)&dev_curr, vectorSize * sizeof(float));
    cudaMalloc((void**)&error, vectorSize * sizeof(float));

    // Copy input matrixes from host memory to GPU buffers.
    cudaMemcpy(dev_alpha, alpha, (vectorSize * vectorSize) * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_beta, beta, vectorSize * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_prev, prev, vectorSize * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_curr, curr, vectorSize * sizeof(float), cudaMemcpyHostToDevice);

    // Launch a kernel on the GPU with one thread for each element.
    int blockSize, gridSize;
    // Number of threads in each thread block
    blockSize = 1024;
    // Number of thread blocks in grid
    gridSize = (int)ceil((float)vectorSize / blockSize);

    cout << "gridSize: " << gridSize << endl;

    // Global memory
    int count = 0, criticalCount = CRITICAL_COUNT;
    // bool *isBad = false;
    for (count = 0; count < criticalCount; count++)
    {
        // *isGood = true;
        parallelCudaMultiMatrixVectorKernel << <gridSize, blockSize >> > (dev_alpha, dev_prev, dev_curr, vectorSize);
        cudaDeviceSynchronize();
        parallelCudaSubVectorsKernel << <gridSize, blockSize >> > (dev_beta, dev_curr, dev_curr, vectorSize);
        cudaDeviceSynchronize();
        // parallelCudaIsComplete << <gridSize, blockSize >> > (dev_prev, dev_curr, isGood, vectorSize);
        // cudaDeviceSynchronize();

        // calc diff vectors
        // parallelCudaSubAbsVector << < gridSize, blockSize >> > (dev_prev, dev_curr, error, vectorSize);
        // cudaDeviceSynchronize();

        // float* sum = 0;
        // parallelCudaSum << < gridSize, blockSize >> > (error, sum, vectorSize);
        // cudaDeviceSynchronize();

        float eps = EPS;
        if (fabs(dev_curr - dev_prev) < eps) break;

        // if (isGood) break;
        parallelCudaCopyVectorsKernel << <gridSize, blockSize >> > (dev_curr, dev_prev, vectorSize);
        // cudaMemcpy(dev_prev, dev_curr, vectorSize * sizeof(float), cudaMemcpyDeviceToDevice);
        cudaDeviceSynchronize();
    }

    cout << "CUDA calculation completed with count: " << count;

        // Copy output array from GPU buffer to host memory.
        cudaMemcpy(alpha, dev_alpha, (vectorSize * vectorSize) * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(beta, dev_beta, vectorSize * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(curr, dev_curr, vectorSize * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(prev, dev_prev, vectorSize * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(dev_alpha);
    cudaFree(dev_beta);
    cudaFree(dev_prev);
    cudaFree(dev_curr);
}

// -----------------
// |     MAIN      |
// -----------------

int main()
{
    srand(time(NULL));
    cudaEvent_t start, stop;
    float KernelTime;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cout << "Enter size of matrix: ";
    int matrixSize = 0;
    cin >> matrixSize;

    float* matrix = new float[matrixSize * matrixSize];
    float* curr = new float[matrixSize];
    float* prev = new float[matrixSize];

    float* alpha = new float[matrixSize * matrixSize];
    float* beta = new float[matrixSize];

    // ------- SEQ START ---------
    init(matrixSize, matrix, prev, curr, alpha, beta);

    // cudaEventRecord(start, 0);

    cout << "------------------------\n";
    cout << "|      SEQUENTAL       |\n";
    cout << "------------------------\n";
    double startSEQ = omp_get_wtime();
    sequentialCalculate(matrixSize, alpha, beta, prev, curr);
    double endSEQ = omp_get_wtime(), timeSEQ = (endSEQ - startSEQ) * 1000;

    // cudaThreadSynchronize();
    // cudaEventRecord(stop, 0);
    // cudaEventSynchronize(stop);
    // cudaEventElapsedTime(&KernelTime, start, stop);

    cout << " with time: " << (timeSEQ) << " ms\n";
    // ------- SEQ END ---------

    // ------- PARALLEL OMP START ---------
    initNull(matrixSize, prev);
    initNull(matrixSize, curr);

    // cudaEventRecord(start, 0);

    cout << "------------------------\n";
    cout << "|   PARALLEL OPENMP    |\n";
    cout << "------------------------\n";
    double startOMP = omp_get_wtime();
    parallelOpenMPCalculate(matrixSize, alpha, beta, prev, curr);

    // cudaThreadSynchronize();
    // cudaEventRecord(stop, 0);
    // cudaEventSynchronize(stop);
    // cudaEventElapsedTime(&KernelTime, start, stop);
    double endOMP = omp_get_wtime(), timeOMP = (endOMP - startOMP) * 1000;

    cout << " with time: " << (timeOMP) << " ms\n";
    // ------- PARALLEL OMP END ---------

    // ------- PARALLEL CUDA START ---------
    initNull(matrixSize, prev);
    initNull(matrixSize, curr);

    cudaEventRecord(start, 0);

    cout << "------------------------\n";
    cout << "|    PARALLEL CUDA     |\n";
    cout << "------------------------\n";
    parallelCudaCalculate(matrixSize, alpha, beta, prev, curr);

    cudaThreadSynchronize();
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&KernelTime, start, stop);

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess)
    {
        // print the CUDA error message and exit
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        exit(1);
    }

    cout << " with time: " << (KernelTime) << " ms\n";
    // ------- PARALLEL CUDA END ---------

    return 0;
}