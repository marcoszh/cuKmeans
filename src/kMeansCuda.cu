#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include "kMeansCuda.h"

namespace cuda {

//返回大于n的最小的2的整数次幂，用于配置dim3的结构
static inline int nextPowerOfTwo(int n) {
	n--;

	n = n >> 1 | n;
	n = n >> 2 | n;
	n = n >> 4 | n;
	n = n >> 8 | n;
	n = n >> 16 | n;
	return ++n;
}

void get_kernel_config_given_ratios(int sz1, int sz2, dim3& szGrid,
		dim3& szBlock, int& rowPerThread, int& colPerThread, int nThreadXRatio,
		int nThreadYRatio) {
	szBlock.x = std::min(sz1, nThreadXRatio);
	szBlock.y = std::min(sz2, nThreadYRatio);
	szBlock.z = 1;
	szGrid.x = szGrid.y = szGrid.z = 1;
	colPerThread = rowPerThread = 1;

	if (sz1 > nThreadXRatio || sz2 > nThreadYRatio) {
		int ratio = sz1 / nThreadXRatio, k;
		for (k = 1; (1 << k) <= ratio; ++k) {
			rowPerThread = (2 << (k / 2));
		}

		szGrid.x = (sz1 + szBlock.x * rowPerThread - 1)
				/ (szBlock.x * rowPerThread);

		ratio = sz2 / nThreadYRatio;
		for (k = 1; (1 << k) <= ratio; ++k) {
			colPerThread = (2 << (k / 2));
		}

		szGrid.y = (sz2 + szBlock.y * colPerThread - 1)
				/ (szBlock.y * colPerThread);
	}
	assert(szGrid.x * szBlock.x * rowPerThread >= sz1);
	assert(szGrid.y * szBlock.y * colPerThread >= sz2);
}

//计算线程配置
void get_kernel_config(int sz1, int sz2, dim3& szGrid, dim3& szBlock,
		int& rowPerThread, int& colPerThread) {

	int nThreadX, nThreadY;
	if (sz1 / sz2 >= 2) {
		nThreadX = 64;
		nThreadY = 16;
	} else if (sz2 / sz1 >= 2) {
		nThreadX = 16;
		nThreadY = 64;
	} else {
		nThreadX = nThreadY = 32;
	}
	get_kernel_config_given_ratios(sz1, sz2, szGrid, szBlock, rowPerThread,
			colPerThread, nThreadX, nThreadY);
}

/**
 * 这段代码实际上就是并行的计算向量objects[objectId]和clusters[clusterId]之间的距离，即第objectId个数据点到第clusterId个中心点的距离。
 */__host__ __device__ inline static
float euclid_dist_2(int numCoords, int numObjs, int numClusters, float *objects, // [numCoords][numObjs]
		float *clusters,    // [numCoords][numClusters]
		int objectId, int clusterId) {
	int i;
	float ans = 0.0;

	for (i = 0; i < numCoords; i++) {
		ans += (objects[numObjs * i + objectId]
				- clusters[numClusters * i + clusterId])
				* (objects[numObjs * i + objectId]
						- clusters[numClusters * i + clusterId]);
	}

	return (ans);
}

/**
 * 这个函数计算的就是第objectId个数据点到numClusters个中心点的距离，然后根据情况比较更新membership。
 */__global__ static
void find_nearest_cluster(int numCoords, int numObjs, int numClusters,
		float *objects,           //  [numCoords][numObjs]
		float *deviceClusters,    //  [numCoords][numClusters]
		int *membership,          //  [numObjs]
		int *intermediates) {
	extern __shared__ char sharedMemory[];

	unsigned char *membershipChanged = (unsigned char *) sharedMemory;
#if BLOCK_SHARED_MEM_OPTIMIZATION
	float *clusters = (float *) (sharedMemory + blockDim.x);
#else
	float *clusters = deviceClusters;
#endif

	membershipChanged[threadIdx.x] = 0;

#if BLOCK_SHARED_MEM_OPTIMIZATION
	//若有太多中心或中心维度过大，共享内存会不够使用，若发生呢，请关闭共享内存使用宏
	for (int i = threadIdx.x; i < numClusters; i += blockDim.x) {
		for (int j = 0; j < numCoords; j++) {
			clusters[numClusters * j + i] = deviceClusters[numClusters * j + i];
		}
	}
	__syncthreads();
#endif

	int objectId = blockDim.x * blockIdx.x + threadIdx.x;

	if (objectId < numObjs) {
		int index, i;
		float dist, min_dist;

		//找到距当前数据最近的中心
		index = 0;
		min_dist = euclid_dist_2(numCoords, numObjs, numClusters, objects,
				clusters, objectId, 0);

		for (i = 1; i < numClusters; i++) {
			dist = euclid_dist_2(numCoords, numObjs, numClusters, objects,
					clusters, objectId, i);

			if (dist < min_dist) {
				//找到更小的，存下索引
				min_dist = dist;
				index = i;
			}
		}

		if (membership[objectId] != index) {
			membershipChanged[threadIdx.x] = 1;
		}

		//为数据指定membership
		membership[objectId] = index;

		__syncthreads();

		for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
			if (threadIdx.x < s) {
				membershipChanged[threadIdx.x] += membershipChanged[threadIdx.x
						+ s];
			}
			__syncthreads();
		}

		if (threadIdx.x == 0) {
			intermediates[blockIdx.x] = membershipChanged[0];
		}
	}
}

/**
 * 这段代码的意义就是将一个线程块中每个线程的对应的intermediates的数据求和最后放到deviceIntermediates[0]中去然后拷贝回主存块中去。
 * 这个问题的更好的解释在这里，实际上就是一个数组求和的问题，应用在这里求得的是有改变的membership中所有数据的和，即改变了簇的点的个数。
 */__global__ static
void compute_delta(int *deviceIntermediates, int numIntermediates, //中间向量的实际数量
		int numIntermediates2) {
	//元素数量需要numIntermediates2是启动的线程数
	extern __shared__ unsigned int intermediates[];

	//从全局拷贝到共享内存
	intermediates[threadIdx.x] =
			(threadIdx.x < numIntermediates) ?
					deviceIntermediates[threadIdx.x] : 0;

	__syncthreads();

	for (unsigned int s = numIntermediates2 / 2; s > 0; s >>= 1) {
		if (threadIdx.x < s) {
			intermediates[threadIdx.x] += intermediates[threadIdx.x + s];
		}
		__syncthreads();
	}

	if (threadIdx.x == 0) {
		deviceIntermediates[0] = intermediates[0];
	}
}

 //更新中心点的坐标
__global__ static
void update_cluster(const float* objects, const int* membership,
		float* clusters, const int nCoords, const int nObjs,
		const int nClusters, const int rowPerThread, const int colPerThread) {
	for (int cIdx = 0; cIdx < colPerThread; ++cIdx) {
		int c = cIdx * gridDim.y * blockDim.y + blockIdx.y * blockDim.y
				+ threadIdx.y;
		if (c >= nClusters)
			break;

		for (int rIdx = 0; rIdx < rowPerThread; ++rIdx) {
			int r = rIdx * gridDim.x * blockDim.x + blockIdx.x * blockDim.x
					+ threadIdx.x;
			if (r >= nCoords)
				break;

			float sumVal(0);
			int clusterCount(0);
			for (int i = 0; i < nObjs; ++i) {
				if (membership[i] == c) {
					sumVal += objects[r * nObjs + i];
					clusterCount++;
				}
			}
			if (clusterCount > 0)
				clusters[nClusters * r + c] = sumVal / clusterCount;
		}
	}
}


__global__ static
void copy_rows(const float* src, const int sz1, const int sz2,
		const int copiedRows, float* dest, const int rowPerThread,
		const int colPerThread) {
	for (int rIdx = 0; rIdx < rowPerThread; ++rIdx) {
		int r = rIdx * gridDim.x * blockDim.x + blockIdx.x * blockDim.x
				+ threadIdx.x;
		if (r >= copiedRows)
			break;

		for (int cIdx = 0; cIdx < colPerThread; ++cIdx) {
			int c = cIdx * gridDim.y * blockDim.y + blockIdx.y * blockDim.y
					+ threadIdx.y;
			if (c >= sz2)
				break;
			dest[c * copiedRows + r] = src[c * sz1 + r];
		}
	}
}

int kMeans(float *deviceObjects, /* host端输入数据 */
int numCoords, /* 每个数据的维度 */
int numObjs, /* 数据条数*/
int numClusters, /* 中心数 */
float threshold, /* % 变换membership的阈值 */
int maxLoop, /* 最大循环次数 */
int *membership, /* 输出的membership向量 */
float *deviceClusters /*device端的Cluster内存*/) {
	int loop = 0;
	float delta; /*变换类别的数据条目占比*/
	int *deviceMembership;
	int *deviceIntermediates;

	CHECK_PARAM(deviceClusters, "deviceClusters cannot be NULL");

	//为支持规约，每个block的线程数必须为2的指数
	const unsigned int numThreadsPerClusterBlock = 128;
	//向上取整，得到blocks数
	const unsigned int numClusterBlocks = (numObjs + numThreadsPerClusterBlock
			- 1) / numThreadsPerClusterBlock;
#if BLOCK_SHARED_MEM_OPTIMIZATION
	//printf("kmeans called");
	const unsigned int clusterBlockSharedDataSize = numThreadsPerClusterBlock
			* sizeof(unsigned char) + numClusters * numCoords * sizeof(float);

	//获取设备描述信息
	cudaDeviceProp deviceProp;
	int deviceNum;
	cudaGetDevice(&deviceNum);
	cudaGetDeviceProperties(&deviceProp, deviceNum);
	//确保共享内存足够使用
	if (clusterBlockSharedDataSize > deviceProp.sharedMemPerBlock) {
		printf(
				"WARNING: Your CUDA hardware has insufficient block shared memory. "
						"You need to recompile with BLOCK_SHARED_MEM_OPTIMIZATION=0. \n");
	}
#else
	const unsigned int clusterBlockSharedDataSize =
	numThreadsPerClusterBlock * sizeof(unsigned char);
#endif

	//求规约线程数
	const unsigned int numReductionThreads = nextPowerOfTwo(numClusterBlocks);
	const unsigned int reductionBlockSharedDataSize = numReductionThreads
			* sizeof(unsigned int);

	CHECK_CUDA(cudaMalloc(&deviceMembership, numObjs * sizeof(int)));
	CHECK_CUDA(
			cudaMalloc(&deviceIntermediates,
					numReductionThreads * sizeof(unsigned int)));

	// 初始化membership向量
	if (membership) {
		for (int i = 0; i < numObjs; i++)
			membership[i] = -1;
		CHECK_CUDA(
				cudaMemcpy(deviceMembership, membership, numObjs * sizeof(int),
						cudaMemcpyHostToDevice));
	} else {
		int* hostMembership = (int*) malloc(numObjs * sizeof(int));
		CHECK_PARAM(hostMembership, "memory allocation failed");
		for (int i = 0; i < numObjs; i++)
			hostMembership[i] = -1;
		CHECK_CUDA(
				cudaMemcpy(deviceMembership, hostMembership,
						numObjs * sizeof(int), cudaMemcpyHostToDevice));
		free(hostMembership);
	}

	//线程结构
	dim3 szGrid, szBlock;
	int rowPerThread, colPerThread;

	//初始化中心点
	get_kernel_config(numClusters, numCoords, szGrid, szBlock, rowPerThread,
			colPerThread);
	copy_rows<<<szGrid, szBlock>>>(deviceObjects, numObjs, numCoords,
			numClusters, deviceClusters, rowPerThread, colPerThread);

	do {
		find_nearest_cluster<<<numClusterBlocks, numThreadsPerClusterBlock,
				clusterBlockSharedDataSize>>>(numCoords, numObjs, numClusters,
				deviceObjects, deviceClusters, deviceMembership,
				deviceIntermediates);


		compute_delta<<<1, numReductionThreads, reductionBlockSharedDataSize>>>(
				deviceIntermediates, numClusterBlocks, numReductionThreads);


		get_kernel_config(numCoords, numClusters, szGrid, szBlock, rowPerThread,
				colPerThread);

		update_cluster<<<szGrid, szBlock>>>(deviceObjects, deviceMembership,
				deviceClusters, numCoords, numObjs, numClusters, rowPerThread,
				colPerThread);

		cudaDeviceSynchronize();
		CHECK_CUDA(cudaGetLastError());

		int d;
		CHECK_CUDA(
				cudaMemcpy(&d, deviceIntermediates, sizeof(int),
						cudaMemcpyDeviceToHost));
		delta = (float) d / numObjs;
	} while (delta > threshold && loop++ < maxLoop);

	if (membership) {
		CHECK_CUDA(
				cudaMemcpy(membership, deviceMembership, numObjs * sizeof(int),
						cudaMemcpyDeviceToHost));
	}
	CHECK_CUDA(cudaFree(deviceMembership));
	CHECK_CUDA(cudaFree(deviceIntermediates));

	return (loop + 1);
}

}
