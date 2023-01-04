#include "surf_structures.h"
#include "cuda_utils.h"
// #include "cuda_image.h"



namespace surf
{
#define MAX_SCALE	8	// Max scales
#define MAX_OCTAVE	8	// Max octaves
#define NBIN		72	// The number of histogram bins to quantify orientation estimation
#define window		1.0471975511965976f		// (M_PI / 3)
#define sep_angle	0.08726646259971647f	// (2 * M_PI / NBIN)
#define hwn			6
#define oradius		9
#define oradiussq	81.5f


	/* Set the paramters for hessian response computation */
	void setHessianParams(const int* params);

	/* Set the normalization factor for hessian response computation */
	void setHessianNorms(const float* norms);

	/* Set the border for finding extrema points */
	void setBorders(const int* borders);

	/* Set the maximum size border for finding extrema points */
	void setMaximumBorders(const int* borders);

	/* Set the max number of keypoints allowed */
	void setMaxNumPoints(const int num);

	/* Set the value of iradius */
	void setIradius(int* v);
	void getIradius(int* v);

	/* Set the paramters for feature interpolation */
	void setSurfParam(const SurfParam& its);

	/* Set the size of image */
	void setWhps(const int3* _whps, const int sz = MAX_OCTAVE);

	/* Set the paramters for feature interpolation */
	//void setInterpParams(const int* params);

	/* Get the address of point counter */
	void getPointCounter(void** addr);

	/* Set look up table */
	void setLut1(const float* lut1);
	void setLut2(const float* lut2);

	/* Set angle bins */
	void setBins(const float* hbins);




	/* Calculate integral image */
	__global__ void integralRow(unsigned char* src, int* dst, int srcw, int srch, int srcp, int dstp);
	__global__ void integralCol(int* data, int width, int height, int pitch);
	void cuIntegral(unsigned char* src, int* dst, int3 whp0, int3 whp1);

	/* Calculate integral image with double size */
	__global__ void integralDoubleRow0U2(unsigned char* src, int* dst, int srcw, int srch, int srcp, int dstp);
	__global__ void integralRow1U4(int* data, int width, int height, int pitch);
	__global__ void integralRow2U4(int* data, int width, int height, int pitch);
	__global__ void integralCol0U4(int* data, int width, int height, int pitch);
	__global__ void integralCol1U4(int* data, int width, int height, int pitch);
	__global__ void integralCol2U4(int* data, int width, int height, int pitch);
	void cuIntegralDoubleU4(unsigned char* src, int* dst, int3 whp0, int3 whp1);

	/* Resize image to half size */
	__global__ void halfImage(float* src, float* dst, int srcp, int dstw, int dsth, int dstp);
	void cuHalfImage(float* src, float* dst, int3 whp0, int3 whp1);

	/* Calculate hessian response for a point */
	inline __device__ int getSum(int* data, int x1, int y1, int x2, int y2, int p);
	__device__ float getHessian(int* data, int* x, int p);

	/* Calculate trace feature for a point */
	__device__ int getTrace(int* data, int* x, int p);

	/* Solve linear equation */
	__device__ void solveLinearSystem(float* solution, float sq[3][3], int size);

	/* Fit quadrat */
	//__device__ float fitQuadrat(float* src, float* _offsets, const int s, const  int r, const int c, const int o);
	__device__ float fitQuadrat(float* src, float* _offsets, float* _g, float _H[3][3], const int s, const  int r, const int c, const int osize);

	/* Get a point */
	__device__ void makePoint(int* iimage, SurfPoint* point, float x, float y, float scale, float strength);

	/* Calculate Hessian response for one image */
	__global__ void calcHessian(int* src, float* dst, int3 vas, float norm, int detcols, int detrows, int srcp, int dstp, 
		int border1, int border2, int mask_size, int delta);
	void cuCalcHessian(int* src, float* dst, int3 iwhp, int3 swhp, int border1, int border2, int mask_size, int delta);

	/* Calculate Hessian response for multiple images */
	__global__ void calcHessianMulti(int* src, float* dst, int3 iwhp, int3 swhp, int init_scale, int nscale, int* params, float* norms);
	__global__ void calcHessianMultiU4(int* src, float* dst, int3 iwhp, int3 swhp, int init_scale, int nscale, int* params, float* norms);
	__global__ void calcHessianMultiConst(int* src, float* dst, int init_scale, int nscale);
	void cuCalcHessianMulti(int* src, float* dst, int3 iwhp, int3 swhp, int init_scale, int max_scale, int& init_mask_size, 
		int init_border, int* borders, int octave, int sampling);

	/* Find Maximum */
	__global__ void findMaximum(float* src, SurfPoint* points, int o, int max_scale, float thresh);
	__global__ void findMaximumWithInterp(int* iimage, float* src, SurfPoint* points, int o, int octave, int moves_remain);
	void cuFindMaximum(float* src, SurfData& result, int* borders, int3 whp, int o, int max_scale, float thresh);
	void cuFindMaximumWithInterp(int* iimage, float* src, SurfData& result, int* borders, int3 whp, int o, int octave, int moves_remain, SurfParam& its);

	///* Interpolate features */
	//__global__ void interpFeature(int* iimage, float* src, SurfPoint* points, bool* flags, int moves_remain, int nkpts);
	//__global__ void reassign(SurfPoint* points, bool* indexes, int num);
	//void cuInterpFeature(int* iimage, float* src, SurfData& result, int3* swhps, int* params, int moves_remain, SurfParam& its, int tot_osize);

	/* Extract SURF descriptors */
	inline __device__ int getWavelet1(int* iimage, int x, int y, int size, int p);
	inline __device__ int getWavelet2(int* iimage, int x, int y, int size, int p);
	__device__ void placeInIndex(float* descriptor, float mag1, int ori1, float mag2, int ori2, float rx, float cx);
	__global__ void assignOrientationApprox(int* iimage, float* descriptors, SurfPoint* points);
	__global__ void describeApprox(int* iimage, float* descriptors, SurfPoint* points);
	__global__ void describeApproxWithOrient(int* iimage, float* descriptors, SurfPoint* points);
	__global__ void describeApproxWithoutNormalization(int* iimage, float* descriptors, SurfPoint* points);
	__global__ void describeUR(int* iimage, float* descriptors, SurfPoint* points);
	__global__ void describeURMulti(int* iimage, float* descriptors, SurfPoint* points);
	__global__ void describeURMulti(cudaTextureObject_t iimage, float* descriptors, SurfPoint* points);
	__global__ void describeURWithoutNormalization(int* iimage, float* descriptors, SurfPoint* points);
	__global__ void normalize(float* descriptors);
	__global__ void normalizeAtomic(float* descriptors);
	void cuDescribe(int* iimage, float** desc_addr, SurfData& result, SurfParam& its);
	void cuDescribe(cudaTextureObject_t iimage, float** desc_addr, SurfData& result, SurfParam& its);

	/* Matching */

	/* 分别将图像1和图像2的特征划分为4个一组，计算图像1和图像2的每组之间的相关性，选择最大相关性的一对作为匹配结果 */
	__global__ void findMaxCorr(SurfPoint* surf1, SurfPoint* surf2, float* feat1, float* feat2, int num_pts1, int num_pts2);
	void cuFindMaxCorr(SurfPoint* surf1, SurfPoint* surf2, float* feat1, float* feat2, int num_pts1, int num_pts2, int nfeatures);


}

