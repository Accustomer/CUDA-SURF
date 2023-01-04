#pragma once
#include "surf_structures.h"
#include "cuda_utils.h"



namespace surf
{
	/* Initialze surf data */
	void initSurfData(SurfData& data, const int max_pts, const bool host, const bool dev);

	/* Free surf data */
	void freeSurfData(SurfData& data);


	/***** Class for SURF computation *****/
	class Surfor
	{
	public:
		/* Contructor */
		Surfor();

		/* Destructor */
		~Surfor();

		/* Initialization */
		void init(const int _noctaves, const float _thresh = 0.2f, const bool _doubled = false, const int _init_mask_size = 9,
			const int _sampling_step = 2, const bool _upright = false, const bool _extend = false, const int _desc_wsz = 4, 
			const int _width = -1, const int _height = -1);

		/* Detect keypoints and compute features. */
		//void detectAndCompute(CudaImage& image, SurfData& result, const bool desc = true);
		//void detectAndCompute(CudaImage& image, SurfDataUR& result, const bool desc = true);

		/* Detect keypoints and compute features. Uchar will not cause to floating-point error. */
		void detectAndCompute(unsigned char* image, SurfData& result, int3 whp0, float** desc_addr, const bool desc = true);
		//void detectAndCompute(uchar* image, SurfDataUR& result, int3 whp0, const bool desc = true);

		/* Match surf data */
		void match(SurfData& data1, SurfData& data2, float* features1, float* features2);

	private:
		SurfParam its;
		// the size of image
		int3 whp{0};
		// the memory for reusing
		int* imem = NULL;	// Integral image memory
		float* omem = NULL;	// Octave hessian response images memory
		size_t total_osize = 0;
		size_t total_isize = 0;
		// Look up table for feature extraction
		float table1[83];
		float table2[40];

		/* Intialize loop up table */
		void initLut();

		/* Allocate memory */
		int allocMemory(void** iaddr, void** oaddr, const int w, const int h, int3& iwhp, int3* swhps, int* osizes, const bool reused);


	};
}

