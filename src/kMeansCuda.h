#ifndef KMEANSCUDA_H
#define	KMEANSCUDA_H

#include <sstream>
#include <stdexcept>
#include <cuda_runtime.h>

//若共性内存足够大，则试用其来提高性能
#define BLOCK_SHARED_MEM_OPTIMIZATION 1
#define COLUM_MAJORED 1;

namespace cuda {
//根据返回值确定是否调用成功，若失败则抛出异常
inline void checkCudaError(cudaError_t err, char const * file,
		unsigned int line) {
	if (err != cudaSuccess) {
		std::stringstream ss;
		ss << "CUDA error " << err << " at " << file << ":" << line;
		throw std::runtime_error(ss.str());
	}
}

//根据传入内存是否为空来判断内存是否申请成功
inline void check(bool bTrue, const char* msg, char const * file,
		unsigned int line) {
	if (!bTrue) {
		std::stringstream ss;
		ss << "Error: \"" << msg << "\" at " << file << ":" << line;
		throw std::runtime_error(ss.str());
	}
}

#define CHECK_PARAM(x, msg)   cuda::check((x), (msg), __FILE__, __LINE__)
#define CHECK_CUDA(cudaError) cuda::checkCudaError((cudaError), __FILE__, __LINE__)


//申请二维内存的宏
#define malloc2D(name, xDim, yDim, type) do {               \
   name = (type **)malloc(xDim * sizeof(type *));          \
   assert(name != NULL);                                   \
   name[0] = (type *)malloc(xDim * yDim * sizeof(type));   \
   assert(name[0] != NULL);                                \
   for (size_t i = 1; i < xDim; i++)                       \
       name[i] = name[i-1] + yDim;                         \
} while (0）

int kMeans(float *deviceObjects, int numCoords, int numObjs, int numClusters,
		float threshold, int maxLoop, int *membership, float *deviceClusters);
}

#endif	/* KMEANSCUDA_H */

