#include <cstdlib>
#include <cmath>
#include <assert.h>
#include <iostream>
#include <ctime>
#include "kMeansCuda.h"

#define ITER_TIMES 10	//指定迭代次数

//http://blog.csdn.net/lavorange/article/details/41942323

//创建数据
float* createDataColMajored(int sz1, int sz2) {
	float* arr;
	CHECK_CUDA(
			cudaMallocHost(&arr, sz1*sz2*sizeof(float), cudaHostAllocDefault)); //页锁定内存，传输速度更高

	for (int i = 0; i < sz1; ++i) {
		for (int j = 0; j < sz2; ++j) {
			arr[sz1 * j + i] = i * 100 + j;
		}
	}
	return arr;
}


float* callkMeans1(float* hostData, int nObjs, int nDim, int nClusters,
		int*& membership) {
	float* devData, *devClusters, *hostClusters;
	//申请内存并赋值
	CHECK_CUDA(cudaMalloc(&devData, nObjs * nDim * sizeof(float)));
	CHECK_CUDA(
			cudaMemcpy(devData, hostData, nObjs * nDim * sizeof(float),
					cudaMemcpyHostToDevice));
	//申请用于存中心的内存
	CHECK_CUDA(cudaMalloc(&devClusters, nClusters * nDim * sizeof(float)));
	//fprintf(stdout, "memory initialization in kmeans1 called\n");

	if (membership)
		membership = new int[nObjs];

	cuda::kMeans(devData, nDim, nObjs, nClusters, 0, 500, membership,
			devClusters);
	hostClusters = new float[nClusters * nDim * sizeof(float)];

	//等待设备
	CHECK_CUDA(cudaDeviceSynchronize());
	CHECK_CUDA(cudaGetLastError());
	//取得中心
	CHECK_CUDA(
			cudaMemcpy(hostClusters, devClusters,
					nClusters * nDim * sizeof(float), cudaMemcpyDeviceToHost));
	CHECK_CUDA(cudaFree(devData));
	CHECK_CUDA(cudaFree(devClusters));

	return hostClusters;
}

//将结果写到文件中
void writeResult(int* membership, float* clusters, int numClusters,
		int numCoords, int numObjs) {
	FILE *fptr;
	char outFileName[1024];

	//output:the coordinates of the cluster centres
	sprintf(outFileName, "%s.cluster_centres", "col_result");
	printf("writingcoordinates of K=%d cluster centers to file \"%s\"\n",
			numClusters, outFileName);
	fptr = fopen(outFileName, "w");
	for (int i = 0; i < numClusters; i++) {
		fprintf(fptr, "%d ", i);
		for (int j = 0; j < numCoords; j++)
			fprintf(fptr, "%f ", *(clusters + i * numCoords + j));
		fprintf(fptr, "\n");
	}
	fclose(fptr);

	//output:the closest cluster centre to each of the data points
	sprintf(outFileName, "%s.membership", "col_result");
	printf("writing membership of N=%d data objects to file \"%s\" \n", numObjs,
			outFileName);
	fptr = fopen(outFileName, "w");
	for (int i = 0; i < numObjs; i++) {
		fprintf(fptr, "%d %d\n", i, membership[i]);
	}
	fclose(fptr);

}

void startkMeans() {
	fprintf(stdout, "benchmark begins...\n");
	const int sz1 = 1024, sz2 = 1024, nClusters = 10;
	//申请Device内存并初始化
	float* dataCm = createDataColMajored(sz1, sz2);
	//用于记录分类结果
	//一维数组，每一位对应一条数据，其中存放其对应的中心索引
	int* membership1;
	//用于存中心的坐标
	float *clusters1;

	//迭代次数
	const int TIMES = ITER_TIMES;

	fprintf(stdout, "kMeans1 begins %d iterations \n", TIMES);

	{
		clock_t begin = clock();
		for (int i = 0; i < TIMES; ++i) {
			clusters1 = callkMeans1(dataCm, sz1, sz2, nClusters, membership1);
			fprintf(stdout, "%d / %d iter finished\n", i+1, TIMES);
			double elapsed_secs = double(clock() - begin) / CLOCKS_PER_SEC;
			fprintf(stdout, "Total time til now: %lf secs \n", elapsed_secs);
		}

	}

	writeResult(membership1, clusters1, nClusters, sz2, sz1);
	delete[] membership1;
	delete[] clusters1;
	CHECK_CUDA(cudaFreeHost(dataCm));
}

int main(int argc, char **argv) {
	fprintf(stdout, "----kmeans----\n");
	startkMeans();
	return 0;
}

