#include "surf.h"
#include "surfd.h"
// #include "cuda_image.h"
#include <opencv2/core.hpp>


namespace surf
{

	void initSurfData(SurfData& data, const int max_pts, const bool host, const bool dev)
	{
		data.num_pts = 0;
		data.max_pts = max_pts;
		const size_t size = sizeof(SurfPoint) * max_pts;
		data.h_data = host ? (SurfPoint*)malloc(size) : NULL;
		data.d_data = NULL;
		if (dev)
		{
			CHECK(cudaMalloc((void**)&data.d_data, size));
		}
	}


	void freeSurfData(SurfData& data)
	{
		if (data.d_data != NULL)
		{
			CHECK(cudaFree(data.d_data));
		}
		if (data.h_data != NULL)
		{
			free(data.h_data);
		}
		data.num_pts = 0;
		data.max_pts = 0;
	}





	Surfor::Surfor()
	{
	}


	Surfor::~Surfor()
	{
		if (imem)
		{
			CHECK(cudaFree(imem));
		}
		if (omem)
		{
			CHECK(cudaFree(omem));
		}
	}


	void Surfor::init(const int _noctaves, const float _thresh, const bool _doubled, const int _init_mask_size,
		const int _sampling_step, const bool _upright , const bool _extend, const int _desc_wsz, const int _width, const int _height)
	{
		whp.x = _width;
		whp.y = _height;
		whp.z = iAlignUp(_width, 128);

		its.doubled = _doubled;
		its.noctaves = _noctaves;
		its.divisor = its.doubled ? 0.5f : 1.f;
		its.init_lobe = _init_mask_size / 3;
		its.max_scale = its.init_lobe + 2;
		its.sampling = _sampling_step + (its.doubled ? _sampling_step : 0);;
		its.thresh = _thresh;
		its.upright = _upright;
		its.extend = _extend;
		its.desc_wsz = _desc_wsz;
		its.mag_factor = 12 / _desc_wsz;
		its.orient_size = 4 + (_extend ? 4 : 0);
		its.nfeatures = _desc_wsz * _desc_wsz * its.orient_size;

		this->initLut();
		setSurfParam(its);
		if (!_upright)
		{
			float hbins[NBIN];
			hbins[0] = -CV_PI;
			for (int i = 1; i < NBIN; i++)
				hbins[i] = hbins[i - 1] + sep_angle;
			setBins(hbins);
		}
	}

	/*
	void Surfor::detectAndCompute(CudaImage& image, SurfData& result, const bool desc)
	{
		// Compute size and allocate memory
		int3 iwhp;
		int3* swhps = new int3[noctaves];
		int* osizes = new int[noctaves];
		int tot_osize = 0;
		float* tmem = NULL;
		const bool reused = image.width == whp.x && image.height == whp.y;
		if (reused)
		{
			tot_osize = this->allocMemory((void**)&smem, whp.x, whp.y, iwhp, swhps, osizes, reused);
			tmem = smem;
		}
		else
		{
			tot_osize = this->allocMemory((void**)&tmem, image.width, image.height, iwhp, swhps, osizes, reused);
		}

		// Compute integral image
		CudaImage iimage;
		iimage.allocate(iwhp.x, iwhp.y, iwhp.z, false, tmem + tot_osize);
		this->integral(image, iimage);
		//cv::Mat show;
		//iimage.readBack();
		//show = cv::Mat(iimage.height, iimage.width, CV_32FC1, iimage.h_data);
		
		// Intensity values of the integral image and the determinant
		int mask_size = init_lobe - 2;

		// Indices
		int octave = 1;
		int s = 0, k, l;

		// border for non-maximum suppression
		int offset = 0;
		int* borders = new int[max_scales];
		for (int o = 0; o < noctaves; o++)
		{
			// distance to border
			int border1 = 0;
			if (o > 0)
			{
				// divide the last and the third last image in order to save the blob response computation for those two levels				
				const int dof0 = offset;
				const int dof1 = offset + osizes[o];
				const int sof0 = offset - 3 * osizes[o - 1];
				const int sof1 = offset - 1 * osizes[o - 1];
				cuHalfImage(tmem + sof0, tmem + dof0, swhps[o - 1].z, swhps[o].x, swhps[o].y, swhps[o].z);
				cuHalfImage(tmem + sof1, tmem + dof1, swhps[o - 1].z, swhps[o].x, swhps[o].y, swhps[o].z);
				//CudaImage im1, im2, im3, im4;
				//im1.allocate(swhps[o].x, swhps[o].y, swhps[o].z, true, tmem + dof0);
				//im2.allocate(swhps[o].x, swhps[o].y, swhps[o].z, true, tmem + dof1);
				//im3.allocate(swhps[o - 1].x, swhps[o - 1].y, swhps[o - 1].z, true, tmem + sof0);
				//im4.allocate(swhps[o - 1].x, swhps[o - 1].y, swhps[o - 1].z, true, tmem + sof1);
				//cv::Mat show1, show2, show3, show4;
				//im1.readBack(); show1 = cv::Mat(im1.height, im1.width, CV_32FC1, im1.h_data);
				//im2.readBack(); show2 = cv::Mat(im2.height, im2.width, CV_32FC1, im2.h_data);
				//im3.readBack(); show3 = cv::Mat(im3.height, im3.width, CV_32FC1, im3.h_data);
				//im4.readBack(); show4 = cv::Mat(im4.height, im4.width, CV_32FC1, im4.h_data);

				// Calculate border and store in vector
				border1 = ((3 * (mask_size + 4 * octave)) / 2) / (sampling * octave) + 1;
				borders[0] = border1;
				borders[1] = border1;
				s = 2;
			}
			else
			{
				// Save borders for non-maximum suppression
				border1 = ((3 * (mask_size + 6 * octave)) / 2) / (sampling * octave) + 1;
			}

			// Calculate blob response map for all scales in the octave
			GpuTimer timer(0);
			while (s < max_scales)
			{
				borders[s] = border1;
				mask_size += 2 * octave;
				if (s > 2)
					border1 = ((3 * (mask_size)) / 2) / (sampling * octave) + 1;

				// Calculate border size 				
				const int border2 = sampling * octave * border1;
				const int delt = sampling * octave;
				//CudaImage heres;
				//heres.allocate(swhps[o].x, swhps[o].y, swhps[o].z, false, tmem + offset + s * osizes[o]);
				//this->calcHessian(iimage, heres, border1, border2, mask_size, delt);
				this->calcHessian(tmem + tot_osize, tmem + offset + s * osizes[o], iwhp, swhps[o], border1, border2, mask_size, delt);
				s++;
			}

			float t0 = timer.read();
			printf("Time of hessian computation: %f\n", t0);
			offset += max_scales * osizes[o];
		}

		if (reused)
		{
			CHECK(cudaMemset(smem, 0, total_size));
		}
		else
		{
			CHECK(cudaFree(tmem));
		}
		delete[] borders;
		delete[] swhps;
		delete[] osizes;
	}
	*/

	void Surfor::detectAndCompute(unsigned char* image, SurfData& result, int3 whp0, float** desc_addr, const bool desc)
	{
		// Get address of point counter
		unsigned int* d_point_counter_addr;
		getPointCounter((void**)&d_point_counter_addr);
		CHECK(cudaMemset(d_point_counter_addr, 0, MAX_OCTAVE * sizeof(unsigned int)));
		int init_iradius = 0;
		setIradius(&init_iradius);
		setMaxNumPoints(result.max_pts);

		// Compute size and allocate memory
		int3 iwhp;
		int3 swhps[MAX_OCTAVE];
		int osizes[MAX_OCTAVE];
		int tot_osize = 0;
		int* iimage = NULL;
		float* tmem = NULL;
		const bool reused = whp0.x == whp.x && whp0.y == whp.y;
		if (reused)
		{
			tot_osize = this->allocMemory((void**)&imem, (void**)&omem, whp.x, whp.y, iwhp, swhps, osizes, reused);
			iimage = imem; tmem = omem;
		}
		else
		{
			tot_osize = this->allocMemory((void**)&iimage, (void**)&tmem, whp0.x, whp0.y, iwhp, swhps, osizes, reused);
		}

		// Compute integral image
		if (its.doubled)
			cuIntegralDoubleU4(image, iimage, whp0, iwhp);
		else
			cuIntegral(image, iimage, whp0, iwhp);

		// Intensity values of the integral image and the determinant
		int mask_size = its.init_lobe - 2;
		// Indices
		int s = 0, octave = 1;
		// border for non-maximum suppression
		int border1 = 0; 
		int borders[MAX_OCTAVE];
		// memory offset
		int offset = 0;
		for (int o = 0; o < its.noctaves; o++)
		{
			if (o > 0)
			{
				// divide the last and the third last image in order to save the blob response computation for those two levels				
				const int dof0 = offset;
				const int dof1 = offset + osizes[o];
				const int sof0 = offset - 3 * osizes[o - 1];
				const int sof1 = offset - 1 * osizes[o - 1];
				cuHalfImage(tmem + sof0, tmem + dof0, swhps[o - 1], swhps[o]);
				cuHalfImage(tmem + sof1, tmem + dof1, swhps[o - 1], swhps[o]);

				// Calculate border and store in vector
				border1 = ((3 * (mask_size + 4 * octave)) / 2) / (its.sampling * octave) + 1;
				borders[0] = border1;
				borders[1] = border1;
				s = 2;
			}
			else
			{
				// Save borders for non-maximum suppression
				border1 = ((3 * (mask_size + 6 * octave)) / 2) / (its.sampling * octave) + 1;
			}

			// Calculate blob response map for all scales in the octave
			//while (s < max_scales)
			//{
			//	borders[s] = border1;
			//	mask_size += 2 * octave;
			//	if (s > 2)
			//		border1 = ((3 * (mask_size)) / 2) / (sampling * octave) + 1;
			//	// Calculate border size 				
			//	const int border2 = sampling * octave * border1;
			//	const int delt = sampling * octave;
			//	cuCalcHessian(iimage, tmem + offset + s * osizes[o], iwhp, swhps[o], border1, border2, mask_size, delt);
			//	s++;
			//}
			//GpuTimer timer(0);
			cuCalcHessianMulti(iimage, tmem + offset, iwhp, swhps[o], s, its.max_scale, mask_size, border1, borders, octave, its.sampling);
			//cuFindMaximum(tmem + offset, result, borders, swhps[o], o, its.max_scale, 0.8f * its.thresh);
			cuFindMaximumWithInterp(iimage, tmem + offset, result, borders, swhps[o], o, octave, 5, its);
			//float t0 = timer.read();
			//printf("Time of hessian computation: %f\n", t0);

			offset += its.max_scale * osizes[o];
			octave += octave;
		}

		// Interpolate keypoints
		//GpuTimer timer(0);		
		//CHECK(cudaMemcpy(&result.num_pts, &d_point_counter_addr[noctaves - 1], sizeof(unsigned int), cudaMemcpyDeviceToHost));
		//cuInterpFeature(iimage, tmem, result, swhps, iparams, 5, its, tot_osize);
		//float t0 = timer.read();
		//printf("Time of feature intepolation: %f\n", t0);
		CHECK(cudaMemcpy(&result.num_pts, &d_point_counter_addr[its.noctaves - 1], sizeof(unsigned int), cudaMemcpyDeviceToHost));
		result.num_pts = std::min<int>(result.num_pts, result.max_pts);

		//// Band intergral image to texture memory
		//cudaResourceDesc res_desc;
		//memset(&res_desc, 0, sizeof(cudaResourceDesc));
		//res_desc.resType = cudaResourceTypePitch2D;
		//res_desc.res.pitch2D.devPtr = iimage;
		//res_desc.res.pitch2D.width = iwhp.x;
		//res_desc.res.pitch2D.height = iwhp.y;
		//res_desc.res.pitch2D.pitchInBytes = iwhp.z * sizeof(int);
		//res_desc.res.pitch2D.desc = cudaCreateChannelDesc<int>();
		//cudaTextureDesc tex_desc;
		//memset(&tex_desc, 0, sizeof(tex_desc));
		//tex_desc.addressMode[0] = cudaAddressModeClamp;
		//tex_desc.addressMode[1] = cudaAddressModeClamp;
		//tex_desc.filterMode = cudaFilterModePoint;
		//tex_desc.readMode = cudaReadModeElementType;
		//tex_desc.normalizedCoords = 0;
		//cudaTextureObject_t tex_iimage = 0;
		//cudaCreateTextureObject(&tex_iimage, &res_desc, &tex_desc, NULL);

		// Extract descriptors
		if (desc)
		{
			//GpuTimer timer(0);
			cuDescribe(iimage, desc_addr, result, its);
			//cuDescribe(tex_iimage, descriptors, result, its);
			//float t0 = timer.read();
			//printf("Time of feature extraction: %f\n", t0);
		}

		// Copy point data to host
		if (result.h_data != NULL)
		{
			float* h_ptr = &result.h_data[0].x;
			float* d_ptr = &result.d_data[0].x;
			CHECK(cudaMemcpy2D(h_ptr, sizeof(SurfPoint), d_ptr, sizeof(SurfPoint), ((desc && !its.upright) ? 7: 6) * sizeof(float), 
				result.num_pts, cudaMemcpyDeviceToHost));
			//CHECK(cudaMemcpy(result.h_data, result.d_data, sizeof(SurfPoint) * result.num_pts, cudaMemcpyDeviceToHost));
		}

		// Postprocessing
		if (reused)
		{
			CHECK(cudaMemset(imem, 0, total_isize));
			CHECK(cudaMemset(omem, 0, total_osize));
		}
		else
		{
			CHECK(cudaFree(tmem));
			CHECK(cudaFree(iimage));
		}
	}


	void Surfor::initLut()
	{
		for (int n = 0; n < 83; n++)
		{
			table1[n] = expf(-(n + 0.5f) / 12.5f);
		}
		setLut1(table1);

		for (int n = 0; n < 40; n++)
		{
			table2[n] = expf(-(n + 0.5f) / 8.f);
		}
		setLut2(table2);
	}


	int Surfor::allocMemory(void** iaddr, void** oaddr, const int w, const int h, int3& iwhp, int3* swhps, int* osizes, const bool reused)
	{
		// Compute sizes
		iwhp.x = its.doubled ? w + w - 1 : w + 1;
		iwhp.y = its.doubled ? h + h - 1 : h + 1;
		iwhp.z = iAlignUp(iwhp.x, 128);
		swhps[0].x = (iwhp.x - 1) / its.sampling;
		swhps[0].y = (iwhp.y - 1) / its.sampling;
		swhps[0].z = iAlignUp(swhps[0].x, 128);
		osizes[0] = swhps[0].y * swhps[0].z;
		int tot_osize = osizes[0] * its.max_scale;
		for (int i = 0, j = 1; j < its.noctaves; i++, j++)
		{
			swhps[j].x = (swhps[i].x >> 1); 
			swhps[j].y = (swhps[i].y >> 1);
			swhps[j].z = iAlignUp(swhps[j].x, 128);
			osizes[j] = swhps[j].y * swhps[j].z;
			tot_osize += osizes[j] * its.max_scale;
		}

		// Allocate integral image
		if ((reused && !imem) || !reused)
		{
			size_t temp_pitch = 0;
			CHECK(cudaMallocPitch(iaddr, &temp_pitch, sizeof(int) * iwhp.x, iwhp.y));
		}

		// Allocate memory for images in octave
		if ((reused && !omem) || !reused)
		{
			CHECK(cudaMalloc(oaddr, tot_osize * sizeof(float)));
			//CHECK(cudaMallocPitch(oaddr, &temp_pitch, 4096, (tot_osize + 4095) / 4096 * sizeof(float)));
		}

		if (reused)
		{
			total_isize = iwhp.z * iwhp.y * sizeof(int);
			total_osize = tot_osize * sizeof(float);
			//total_osize = (((tot_osize + 4095) >> 12) << 12) * sizeof(float);
		}
		return tot_osize;
	}


	void Surfor::match(SurfData& data1, SurfData& data2, float* features1, float* features2)
	{
		cuFindMaxCorr(data1.d_data, data2.d_data, features1, features2, data1.num_pts, data2.num_pts, its.nfeatures);
		if (data1.h_data != NULL && data1.d_data != NULL)
		{
			float* h_ptr = &data1.h_data[0].score;
			float* d_ptr = &data1.d_data[0].score;
			CHECK(cudaMemcpy2D(h_ptr, sizeof(SurfPoint), d_ptr, sizeof(SurfPoint), 5 * sizeof(float), data1.num_pts, cudaMemcpyDeviceToHost));
			//CHECK(cudaMemcpy(data1.h_data, data1.d_data, sizeof(SurfPoint) * 1, cudaMemcpyDeviceToHost));
		}
	}
}