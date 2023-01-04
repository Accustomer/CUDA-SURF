#include "surfd.h"
#include <device_launch_parameters.h>


namespace surf
{

#define X1	256
#define X2	32
#define Y2	32


	__constant__ int hessian_params[7 * MAX_SCALE];
	__constant__ float hessian_norms[MAX_SCALE];
	__constant__ int d_borders[MAX_SCALE];
	__constant__ int maximum_borders[(MAX_SCALE - 2) / 2];
	__constant__ int d_max_num_points;
	__device__ unsigned int d_point_counter[MAX_OCTAVE];
	__device__ int d_iradius;
	__constant__ int3 whps[2];
	__constant__ SurfParam d_interp_struct;
	__constant__ float lookup1[83];
	__constant__ float lookup2[40];
	__constant__ float bins[NBIN];


	void setHessianParams(const int* params)
	{
		CHECK(cudaMemcpyToSymbol(hessian_params, params, 7 * MAX_SCALE * sizeof(int), 0, cudaMemcpyHostToDevice));
	}


	void setHessianNorms(const float* norms)
	{
		CHECK(cudaMemcpyToSymbol(hessian_norms, norms, MAX_SCALE * sizeof(float), 0, cudaMemcpyHostToDevice));
	}


	void setBorders(const int* borders)
	{
		CHECK(cudaMemcpyToSymbol(d_borders, borders, MAX_SCALE * sizeof(int), 0, cudaMemcpyHostToDevice));
	}


	void setMaximumBorders(const int* borders)
	{
		CHECK(cudaMemcpyToSymbol(maximum_borders, borders, (MAX_SCALE - 2) / 2 * sizeof(int), 0, cudaMemcpyHostToDevice));
	}


	void setMaxNumPoints(const int num)
	{
		CHECK(cudaMemcpyToSymbol(d_max_num_points, &num, sizeof(int), 0, cudaMemcpyHostToDevice));
	}


	void setIradius(int* v)
	{
		CHECK(cudaMemcpyToSymbol(d_iradius, v, sizeof(int), 0, cudaMemcpyHostToDevice));
	}


	void getIradius(int* v)
	{
		CHECK(cudaMemcpyFromSymbol(v, d_iradius, sizeof(int), 0, cudaMemcpyDeviceToHost));
	}


	void setSurfParam(const SurfParam& its)
	{
		CHECK(cudaMemcpyToSymbol(d_interp_struct, &its, sizeof(SurfParam), 0, cudaMemcpyHostToDevice));
	}


	void getPointCounter(void** addr)
	{
		CHECK(cudaGetSymbolAddress(addr, d_point_counter));
	}


	void setWhps(const int3* _whps, const int sz)
	{
		CHECK(cudaMemcpyToSymbol(whps, _whps, sz * sizeof(int3), 0, cudaMemcpyHostToDevice));
	}


	void setLut1(const float* lut1)
	{
		CHECK(cudaMemcpyToSymbol(lookup1, lut1, 83 * sizeof(float), 0, cudaMemcpyHostToDevice));
	}


	void setLut2(const float* lut2)
	{
		CHECK(cudaMemcpyToSymbol(lookup2, lut2, 40 * sizeof(float), 0, cudaMemcpyHostToDevice));
	}


	void setBins(const float* hbins)
	{
		CHECK(cudaMemcpyToSymbol(bins, hbins, NBIN * sizeof(float), 0, cudaMemcpyHostToDevice));
	}



	/*void setInterpParams(const int* params)
	{
		CHECK(cudaMemcpyToSymbol(interp_params, params, MAX_OCTAVE * (MAX_SCALE + 3) * sizeof(int), 0, cudaMemcpyHostToDevice));
	}*/




	inline __device__ float dFastAtan2(float y, float x)
	{
		const float absx = fabs(x);
		const float absy = fabs(y);
		const float a = __fdiv_rn(min(absx, absy), max(absx, absy));
		const float s = a * a;
		float r = __fmaf_rn(__fmaf_rn(__fmaf_rn(-0.0464964749f, s, 0.15931422f), s, -0.327622764f), s * a, a);
		//float r = ((-0.0464964749f * s + 0.15931422f) * s - 0.327622764f) * s * a + a;
		r = (absy > absx ? H_PI - r : r);
		r = (x < 0 ? M_PI - r : r);
		r = (y < 0 ? -r : r);
		return r;
	}


	__global__ void integralRow(unsigned char* src, int* dst, int srcw, int srch, int srcp, int dstp)
	{
		unsigned int iy = blockIdx.x * blockDim.x + threadIdx.x;
		if (iy < srch)
		{
			unsigned int sidx = iy * srcp;
			unsigned int last_didx = (iy + 1) * dstp + 1;
			unsigned int curr_didx = last_didx + 1;

			dst[last_didx] = __uchar2int(src[sidx]);
			sidx++;
			for (int i = 1; i < srcw; i++)
			{
				dst[curr_didx] = dst[last_didx] + __uchar2int(src[sidx]);
				last_didx = curr_didx;
				curr_didx++;
				sidx++;
			}
		}
	}


	__global__ void integralCol(int* data, int width, int height, int pitch)
	{
		unsigned int ix = blockIdx.x * blockDim.x + threadIdx.x + 1;
		if (ix < width)
		{
			unsigned int ilast = ix + pitch;
			unsigned int icurr = ilast + pitch;
			for (int i = 2; i < height; i++)
			{
				data[icurr] += data[ilast];
				ilast = icurr;
				icurr += pitch;
			}
		}
	}


	__global__ void integralDoubleRow0U2(unsigned char* src, int* dst, int srcw, int srch, int srcp, int dstp)
	{
		unsigned int ix = blockIdx.x * blockDim.x + threadIdx.x;
		unsigned int iy = blockIdx.y * blockDim.y + threadIdx.y;
		unsigned int bix = ix * 2;
		if (bix >= srcw || iy >= srch)
			return;

		// 0
		uint4 sidx, didx;
		sidx.w = iy * srcp + bix;
		sidx.x = sidx.w + 1;
		sidx.y = sidx.w + srcp;
		sidx.z = sidx.y + 1;
		didx.w = (2 * iy + 1) * dstp + 2 * bix + 1;
		didx.x = didx.w + 1;
		didx.y = didx.w + dstp;
		didx.z = didx.y + 1;
		dst[didx.w] = __uchar2int(src[sidx.w]);
		dst[didx.y] = __float2int_rn((__uchar2int(src[sidx.w]) + __uchar2int(src[sidx.y])) * 0.5f);
		if (bix + 1 >= srcw)
			return;
		dst[didx.x] = dst[didx.w] + __float2int_rn((__uchar2int(src[sidx.w]) + __uchar2int(src[sidx.x])) * 0.5f); // (src[sidx.w] + src[sidx.x]) * 0.5f;
		dst[didx.z] = dst[didx.y] + __float2int_rn((__uchar2int(src[sidx.w]) + __uchar2int(src[sidx.x]) +
			__uchar2int(src[sidx.y]) + __uchar2int(src[sidx.z])) * 0.25f);

		// 1
		sidx.w = sidx.x; sidx.x++;
		sidx.y = sidx.z; sidx.z++;
		didx.w += 2; didx.x += 2;
		didx.y += 2; didx.z += 2;
		dst[didx.w] = dst[didx.w - 1] + __uchar2int(src[sidx.w]);
		dst[didx.y] = dst[didx.y - 1] + __float2int_rn((__uchar2int(src[sidx.w]) + __uchar2int(src[sidx.y])) * 0.5f);
		if (bix + 2 >= srcw)
			return;
		dst[didx.x] = dst[didx.w] + __float2int_rn((__uchar2int(src[sidx.w]) + __uchar2int(src[sidx.x])) * 0.5f);
		dst[didx.z] = dst[didx.y] + __float2int_rn((__uchar2int(src[sidx.w]) + __uchar2int(src[sidx.x]) +
			__uchar2int(src[sidx.y]) + __uchar2int(src[sidx.z])) * 0.25f);
	}


	__global__ void integralRow1U4(int* data, int width, int height, int pitch)
	{
		unsigned int iy = blockIdx.x * blockDim.x + threadIdx.x + 1;
		if (iy < height)
		{
			unsigned int idx = iy * pitch + 1;
			unsigned int i = idx + 3;
			unsigned int j = i + 4;
			unsigned int k = 8;
			while (k < width)
			{
				data[j] += data[i];
				i = j;
				j += 4;
				k += 4;
			}
		}
	}


	__global__ void integralRow2U4(int* data, int width, int height, int pitch)
	{
		unsigned int ix = blockIdx.x * blockDim.x + threadIdx.x;
		unsigned int iy = blockIdx.y * blockDim.y + threadIdx.y + 1;
		unsigned int bix = (ix + 1) * 4 + 1;
		if (bix >= width || iy >= height)
			return;
		// 0
		unsigned int idx = iy * pitch + bix - 1;
		unsigned int j = idx + 1;
		data[j] += data[idx];
		// 1
		if (bix + 1 >= width)
			return;
		j++;
		data[j] += data[idx];
		// 2
		if (bix + 2 >= width)
			return;
		j++;
		data[j] += data[idx];
	}


	__global__ void integralCol0U4(int* data, int width, int height, int pitch)
	{
		unsigned int ix = blockIdx.x * blockDim.x + threadIdx.x + 1;
		unsigned int iy = blockIdx.y * blockDim.y + threadIdx.y;
		unsigned int biy = iy * 4 + 1;
		if (ix >= width || biy + 1 >= height)
			return;
		// 1
		unsigned int idx = biy * pitch + ix;
		unsigned int idx1 = idx + pitch;
		data[idx1] += data[idx];
		// 2
		if (biy + 2 >= height)
			return;
		idx = idx1; idx1 += pitch;
		data[idx1] += data[idx];
		// 3
		if (biy + 3 >= height)
			return;
		idx = idx1; idx1 += pitch;
		data[idx1] += data[idx];
	}


	__global__ void integralCol1U4(int* data, int width, int height, int pitch)
	{
		unsigned int ix = blockIdx.x * blockDim.x + threadIdx.x + 1;
		if (ix < width)
		{
			unsigned int step = 4 * pitch;
			unsigned int i = ix + 4 * pitch;
			unsigned int j = i + step;
			unsigned int k = 8;
			while (k < height)
			{
				data[j] += data[i];
				i = j;
				j += step;
				k += 4;
			}
		}
	}


	__global__ void integralCol2U4(int* data, int width, int height, int pitch)
	{
		unsigned int ix = blockIdx.x * blockDim.x + threadIdx.x + 1;
		unsigned int iy = blockIdx.y * blockDim.y + threadIdx.y;
		unsigned int biy = (iy + 1) * 4 + 1;
		if (ix >= width || biy >= height)
			return;
		// 0
		unsigned int idx = (biy - 1) * pitch + ix;
		unsigned int idx1 = idx + pitch;
		data[idx1] += data[idx];
		// 1
		if (biy + 1 >= height)
			return;
		idx1 += pitch;
		data[idx1] += data[idx];
		// 2
		if (biy + 2 >= height)
			return;
		idx1 += pitch;
		data[idx1] += data[idx];
	}


	__global__ void halfImage(float* src, float* dst, int srcp, int dstw, int dsth, int dstp)
	{
		unsigned int ix = blockIdx.x * blockDim.x + threadIdx.x;
		unsigned int iy = blockIdx.y * blockDim.y + threadIdx.y;
		if (ix < dstw && iy < dsth)
		{
			unsigned int sidx = 2 * iy * srcp + 2 * ix;
			unsigned int didx = iy * dstp + ix;
			dst[didx] = src[sidx];
		}
	}


	inline __device__ int getSum(int* data, int x1, int y1, int x2, int y2, int p)
	{
		int yp1 = y1 * p + p;
		int yp2 = y2 * p;
		int idx1 = yp1 + x1 + 1;
		int idx2 = yp2 + x2;
		int idx3 = yp2 + x1 + 1;
		int idx4 = yp1 + x2;
		return data[idx1] + data[idx2] - data[idx3] - data[idx4];
	}


	inline __device__ int getSum(cudaTextureObject_t tex_obj, int x1, int y1, int x2, int y2)
	{
		return tex2D<int>(tex_obj, x1 + 1, y1 + 1) + tex2D<int>(tex_obj, x2, y2) - 
			tex2D<int>(tex_obj, x1 + 1, y2) - tex2D<int>(tex_obj, x2, y1 + 1);
	}


	__device__ float getHessian(int* data, int* x, int p)
	{
		// Get second order derivatives
		const float r = 0.003921568627f;
		const float dxx = __int2float_rn(getSum(data, x[5] + x[2], x[1] + x[3], x[6] - x[2], x[1] - x[3], p) -
			3 * getSum(data, x[0] + x[2], x[1] + x[3], x[0] - x[2], x[1] - x[3], p));
		const float dyy = __int2float_rn(getSum(data, x[0] + x[3], x[7] + x[2], x[0] - x[3], x[8] - x[2], p) -
			3 * getSum(data, x[0] + x[3], x[1] + x[2], x[0] - x[3], x[1] - x[2], p));
		const float dxy = 0.6f * (getSum(data, x[0] + x[4], x[1], x[0], x[1] - x[4], p) +
			getSum(data, x[0], x[1] + x[4], x[0] - x[4], x[1], p) -
			getSum(data, x[0] + x[4], x[1] + x[4], x[0], x[1], p) -
			getSum(data, x[0], x[1], x[0] - x[4], x[1] - x[4], p));
		return r * r * (dxx * dyy - dxy * dxy);
	}


	__device__ int getTrace(int* data, int* x, int p)
	{
		// Get second order derivatives
		const int lxx = getSum(data, x[5] + x[2], x[1] + x[3], x[6] - x[2], x[1] - x[3], p) -
			3 * getSum(data, x[0] + x[2], x[1] + x[3], x[0] - x[2], x[1] - x[3], p);
		const int lyy = getSum(data, x[0] + x[3], x[7] + x[2], x[0] - x[3], x[8] - x[2], p) -
			3 * getSum(data, x[0] + x[3], x[1] + x[2], x[0] - x[3], x[1] - x[2], p);
		return (lxx + lyy > 0 ? 1 : -1);
	}


	__global__ void calcHessian(int* src, float* dst, int3 vas, float norm, int detcols, int detrows, int srcp, int dstp, 
		int border1, int border2, int mask_size, int delta)
	{
		unsigned int ix = blockIdx.x * blockDim.x + threadIdx.x;
		unsigned int iy = blockIdx.y * blockDim.y + threadIdx.y;
		unsigned int idx = ix + border1;
		unsigned int idy = iy + border1;
		if (idx < detcols && idy < detrows)
		{
			int x[9];
			x[2] = vas.x;
			x[3] = vas.y;
			x[4] = vas.z;

			x[1] = border2 + delta * iy;
			x[7] = x[1] + mask_size;
			x[8] = x[1] - mask_size;

			x[0] = border2 + delta * ix;
			x[5] = x[0] + mask_size;
			x[6] = x[0] - mask_size;
			dst[idy * dstp + idx] = getHessian(src, x, srcp) * norm;
		}
	}


	__global__ void calcHessianMulti(int* src, float* dst, int3 iwhp, int3 swhp, int init_scale, int nscale, int* params, float* norms)
	{
		unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
		unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
		unsigned int z = blockIdx.z;
		if (z >= nscale)
			return;

		int* mask_sizes = params;
		int* borders1 = mask_sizes + nscale;
		int* borders2 = borders1 + nscale;
		int* deltas = borders2 + nscale;
		int* x2s = deltas + nscale;
		int* x3s = x2s + nscale;
		int* x4s = x3s + nscale;

		unsigned int ix = x + borders1[z];
		unsigned int iy = y + borders1[z];
		if (ix >= swhp.x - borders1[z] || iy >= swhp.y - borders1[z])
			return;

		int vas[9];
		vas[2] = x2s[z];
		vas[3] = x3s[z];
		vas[4] = x4s[z];

		vas[1] = borders2[z] + deltas[z] * y;
		vas[7] = vas[1] + mask_sizes[z];
		vas[8] = vas[1] - mask_sizes[z];

		vas[0] = borders2[z] + deltas[z] * x;
		vas[5] = vas[0] + mask_sizes[z];
		vas[6] = vas[0] - mask_sizes[z];

		unsigned int id = (z + init_scale) * swhp.y * swhp.z + iy * swhp.z + ix;
		dst[id] = getHessian(src, vas, iwhp.z) * norms[z];
	}


	__global__ void calcHessianMultiConst(int* src, float* dst, int init_scale, int nscale)
	{
		unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
		unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
		unsigned int z = blockIdx.z;
		if (z >= nscale)
			return;

		int* mask_sizes = hessian_params;
		int* borders1 = mask_sizes + MAX_SCALE;
		int* borders2 = borders1 + MAX_SCALE;
		int* deltas = borders2 + MAX_SCALE;
		int* x2s = deltas + MAX_SCALE;
		int* x3s = x2s + MAX_SCALE;
		int* x4s = x3s + MAX_SCALE;

		unsigned int ix = x + borders1[z];
		unsigned int iy = y + borders1[z];
		if (ix >= whps[0].x - borders1[z] || iy >= whps[0].y - borders1[z])
			return;

		int vas[9];
		vas[2] = x2s[z];
		vas[3] = x3s[z];
		vas[4] = x4s[z];

		vas[1] = borders2[z] + deltas[z] * y;
		vas[7] = vas[1] + mask_sizes[z];
		vas[8] = vas[1] - mask_sizes[z];

		vas[0] = borders2[z] + deltas[z] * x;
		vas[5] = vas[0] + mask_sizes[z];
		vas[6] = vas[0] - mask_sizes[z];

		unsigned int id = (z + init_scale) * whps[0].y * whps[0].z + iy * whps[0].z + ix;
		dst[id] = getHessian(src, vas, whps[1].z) * hessian_norms[z];
	}


	__global__ void calcHessianMultiU4(int* src, float* dst, int3 iwhp, int3 swhp, int init_scale, int nscale, int* params, float* norms)
	{
		unsigned int x = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
		unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
		unsigned int z = blockIdx.z;
		if (z >= nscale)
			return;

		int* mask_sizes = params;
		int* borders1 = mask_sizes + nscale;
		int* borders2 = borders1 + nscale;
		int* deltas = borders2 + nscale;
		int* x2s = deltas + nscale;
		int* x3s = x2s + nscale;
		int* x4s = x3s + nscale;

		int vas[9];
		vas[2] = x2s[z];
		vas[3] = x3s[z];
		vas[4] = x4s[z];

		// 0
		unsigned int ix = x + borders1[z];
		unsigned int iy = y + borders1[z];
		if (ix >= swhp.x - borders1[z] || iy >= swhp.y - borders1[z])
			return;
		vas[1] = borders2[z] + deltas[z] * y;
		vas[7] = vas[1] + mask_sizes[z];
		vas[8] = vas[1] - mask_sizes[z];
		vas[0] = borders2[z] + deltas[z] * x;
		vas[5] = vas[0] + mask_sizes[z];
		vas[6] = vas[0] - mask_sizes[z];
		unsigned int id = (z + init_scale) * swhp.y * swhp.z + iy * swhp.z + ix;
		dst[id] = getHessian(src, vas, iwhp.z) * norms[z];

		// 1
		x++; ix++; id++;
		vas[0] += deltas[z];
		vas[5] += deltas[z];
		vas[6] += deltas[z];
		dst[id] = getHessian(src, vas, iwhp.z) * norms[z];

		// 2
		x++; ix++; id++;
		vas[0] += deltas[z];
		vas[5] += deltas[z];
		vas[6] += deltas[z];
		dst[id] = getHessian(src, vas, iwhp.z) * norms[z];

		// 3
		x++; ix++; id++;
		vas[0] += deltas[z];
		vas[5] += deltas[z];
		vas[6] += deltas[z];
		dst[id] = getHessian(src, vas, iwhp.z) * norms[z];
	}


	__global__ void findMaximum(float* src, SurfPoint* points, int o, int max_scale, float thresh)
	{
		unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
		unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
		unsigned int z = blockIdx.z;
		unsigned int k = 2 * z + 1;
		if (k >= max_scale - 1)
			return;

		unsigned int i = maximum_borders[z] + y * 2;
		unsigned int j = maximum_borders[z] + x * 2;
		if (i >= whps[0].y - maximum_borders[z] || j >= whps[0].x - maximum_borders[z])
			return;

		int4 idx;		
		idx.w = i * whps[0].z + j;
		idx.x = idx.w + 1;
		idx.y = idx.w + whps[0].z;
		idx.z = idx.y + 1;
		
		const int osize = whps[0].y * whps[0].z;
		float* curr_scale = src + k * osize;
		int cas = 0;
		float best = curr_scale[idx.w];		
		if (curr_scale[idx.x] > best)
		{
			best = curr_scale[idx.x];
			cas = 1;
		}
		if (curr_scale[idx.y] > best)
		{
			best = curr_scale[idx.y];
			cas = 2;
		}
		if (curr_scale[idx.z] > best)
		{
			best = curr_scale[idx.z];
			cas = 3;
		}

		curr_scale += osize;
		if (curr_scale[idx.w] > best)
		{
			best = curr_scale[idx.w];
			cas = 4;
		}
		if (curr_scale[idx.x] > best)
		{
			best = curr_scale[idx.x];
			cas = 5;
		}
		if (curr_scale[idx.y] > best)
		{
			best = curr_scale[idx.y];
			cas = 6;
		}
		if (curr_scale[idx.z] > best)
		{
			best = curr_scale[idx.z];
			cas = 7;
		}
		if (best < thresh || (k + 1 == max_scale - 1 && cas > 3))
			return;
		
		int s = k, r = i, c = j;
		int ds = -1, dr = -1, dc = -1;
		if (cas != 0)
		{
			if (cas == 1) { c = j + 1; dc = 1; }
			else if (cas == 2) { r = i + 1; dr = 1; }
			else if (cas == 3) { c = j + 1; r = i + 1; dc = 1; dr = 1; }
			else
			{
				s++;
				ds = 1;
				if (cas == 5) { c = j + 1; dc = 1; }
				else if (cas == 6) { r = i + 1; dr = 1; }
				else if (cas == 7) { c = j + 1; r = i + 1; dc = 1; dr = 1; }
			}
		}

		int ss = s + ds;
		curr_scale = src + ss * osize;
		idx.y = (r - dr) * whps[0].z + c;
		idx.x = idx.y - 1;
		idx.z = idx.y + 1;
		if (best < curr_scale[idx.x]) return;
		if (best < curr_scale[idx.y]) return;
		if (best < curr_scale[idx.z]) return;
		idx.y += dr * whps[0].z;
		idx.x = idx.y - 1;
		idx.z = idx.y + 1;
		if (best < curr_scale[idx.x]) return;
		if (best < curr_scale[idx.y]) return;
		if (best < curr_scale[idx.z]) return;
		idx.y += dr * whps[0].z;
		idx.x = idx.y - 1;
		idx.z = idx.y + 1;
		if (best < curr_scale[idx.x]) return;
		if (best < curr_scale[idx.y]) return;
		if (best < curr_scale[idx.z]) return;
		curr_scale = src + s * osize;
		if (best < curr_scale[idx.x]) return;
		if (best < curr_scale[idx.y]) return;
		if (best < curr_scale[idx.z]) return;
		idx.w = r * whps[0].z + c + dc;
		if (best < curr_scale[idx.w]) return;
		idx.w -= dr * whps[0].z;
		if (best < curr_scale[idx.w]) return;
		ss = s - ds;
		curr_scale = src + ss * osize;
		if (best < curr_scale[idx.x]) return;
		if (best < curr_scale[idx.y]) return;
		if (best < curr_scale[idx.z]) return;
		if (best < curr_scale[idx.w]) return;
		idx.w += dr * whps[0].z;
		if (best < curr_scale[idx.w]) return;

		if (o > 0) 
			atomicMax(&d_point_counter[o], d_point_counter[o - 1]);
		if (d_point_counter[o] < d_max_num_points)
		{
			unsigned int pi = atomicInc(&d_point_counter[o], 0x7fffffff);
			if (pi < d_max_num_points)
			{
				points[pi].x = c;
				points[pi].y = r;
				points[pi].scale = s;
				points[pi].o = o;
			}
		}
	}


	__global__ void findMaximumWithInterp(int* iimage, float* src, SurfPoint* points, int o, int octave, int moves_remain)
	{
		unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
		unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
		unsigned int z = blockIdx.z;
		unsigned int k = 2 * z + 1;
		if (k >= d_interp_struct.max_scale - 1)
			return;

		unsigned int i = maximum_borders[z] + y * 2;
		unsigned int j = maximum_borders[z] + x * 2;
		if (i >= whps[0].y - maximum_borders[z] || j >= whps[0].x - maximum_borders[z])
			return;

		int4 idx;
		idx.w = i * whps[0].z + j;
		idx.x = idx.w + 1;
		idx.y = idx.w + whps[0].z;
		idx.z = idx.y + 1;

		const int osize = whps[0].y * whps[0].z;
		float* curr_scale = src + k * osize;
		int cas = 0;
		float best = curr_scale[idx.w];
		if (curr_scale[idx.x] > best)
		{
			best = curr_scale[idx.x];
			cas = 1;
		}
		if (curr_scale[idx.y] > best)
		{
			best = curr_scale[idx.y];
			cas = 2;
		}
		if (curr_scale[idx.z] > best)
		{
			best = curr_scale[idx.z];
			cas = 3;
		}

		curr_scale += osize;
		if (curr_scale[idx.w] > best)
		{
			best = curr_scale[idx.w];
			cas = 4;
		}
		if (curr_scale[idx.x] > best)
		{
			best = curr_scale[idx.x];
			cas = 5;
		}
		if (curr_scale[idx.y] > best)
		{
			best = curr_scale[idx.y];
			cas = 6;
		}
		if (curr_scale[idx.z] > best)
		{
			best = curr_scale[idx.z];
			cas = 7;
		}
		if (best < d_interp_struct.thresh * 0.8f || (k + 1 == d_interp_struct.max_scale - 1 && cas > 3))
			return;

		int s = k, r = i, c = j;
		int ds = -1, dr = -1, dc = -1;
		if (cas != 0)
		{
			if (cas == 1) { c = j + 1; dc = 1; }
			else if (cas == 2) { r = i + 1; dr = 1; }
			else if (cas == 3) { c = j + 1; r = i + 1; dc = 1; dr = 1; }
			else
			{
				s++;
				ds = 1;
				if (cas == 5) { c = j + 1; dc = 1; }
				else if (cas == 6) { r = i + 1; dr = 1; }
				else if (cas == 7) { c = j + 1; r = i + 1; dc = 1; dr = 1; }
			}
		}

		int ss = s + ds;
		curr_scale = src + ss * osize;
		idx.y = (r - dr) * whps[0].z + c;
		idx.x = idx.y - 1;
		idx.z = idx.y + 1;
		if (best < curr_scale[idx.x]) return;
		if (best < curr_scale[idx.y]) return;
		if (best < curr_scale[idx.z]) return;
		idx.y += dr * whps[0].z;
		idx.x = idx.y - 1;
		idx.z = idx.y + 1;
		if (best < curr_scale[idx.x]) return;
		if (best < curr_scale[idx.y]) return;
		if (best < curr_scale[idx.z]) return;
		idx.y += dr * whps[0].z;
		idx.x = idx.y - 1;
		idx.z = idx.y + 1;
		if (best < curr_scale[idx.x]) return;
		if (best < curr_scale[idx.y]) return;
		if (best < curr_scale[idx.z]) return;
		curr_scale = src + s * osize;
		if (best < curr_scale[idx.x]) return;
		if (best < curr_scale[idx.y]) return;
		if (best < curr_scale[idx.z]) return;
		idx.w = r * whps[0].z + c + dc;
		if (best < curr_scale[idx.w]) return;
		idx.w -= dr * whps[0].z;
		if (best < curr_scale[idx.w]) return;
		ss = s - ds;
		curr_scale = src + ss * osize;
		if (best < curr_scale[idx.x]) return;
		if (best < curr_scale[idx.y]) return;
		if (best < curr_scale[idx.z]) return;
		if (best < curr_scale[idx.w]) return;
		idx.w += dr * whps[0].z;
		if (best < curr_scale[idx.w]) return;

		// Interpolate the detected maximum in order to get a more accurate location
		float _offsets[3] = { 0 };
		float _g[3] = { 0 };
		float _H[3][3] = { {0} };
		float strength = 0;
		int newr = r, newc = c;
		for (int mv = 0; mv < moves_remain; mv++)
		{
			r = newr; c = newc;
			strength = fitQuadrat(src, _offsets, _g, _H, s, r, c, osize);
			if (_offsets[1] > 0.6f && r < whps[0].y - d_borders[s]) newr++;
			if (_offsets[1] < -0.6f && r > d_borders[s]) newr--;
			if (_offsets[2] > 0.6f && c < whps[0].x - d_borders[s]) newc++;
			if (_offsets[2] < -0.6f && c > d_borders[s]) newc--;
			if ((newr == r && newc == c)) break;
		}

		// Do not create a keypoint if interpolation still remains far
		// outside expected limits, or if magnitude of peak value is below
		// threshold (i.e., contrast is too low)		
		if (__isnanf(_offsets[0]) || __isnanf(_offsets[1]) || __isnanf(_offsets[2]) ||
			fabs(_offsets[0]) > 1.5f || fabs(_offsets[1]) > 1.5f || fabs(_offsets[2]) > 1.5f ||
			strength < d_interp_struct.thresh)
		{
			return;
		}

		// Check that no ipoint has been created at this location (to avoid duplicates). Otherwise, mark this map location
		float ns = (d_interp_struct.init_lobe + (octave - 1) * d_interp_struct.max_scale + (s + _offsets[0]) * 2 * octave) / 3.f;
		float ny = octave * (r + _offsets[1]);
		float nx = octave * (c + _offsets[2]);
		if (o > 0)
			atomicMax(&d_point_counter[o], d_point_counter[o - 1]);
		if (d_point_counter[o] < d_max_num_points)
		{
			unsigned int pi = atomicInc(&d_point_counter[o], 0x7fffffff);
			makePoint(iimage, &points[pi], nx, ny, ns, strength);
		}
	}


	__device__ void solveLinearSystem(float* solution, float sq[3][3], int size)
	{
		int row, col, c, pivot = 0, i;
		float maxc, coef, temp, mult, val;

		// Triangularize the matrix
		for (col = 0; col < size - 1; col++)
		{
			// Pivot row with largest coefficient to top
			maxc = -1.f;
			for (row = col; row < size; row++)
			{
				coef = sq[row][col];
				coef = (coef < 0.f ? -coef : coef);
				if (coef > maxc)
				{
					maxc = coef;
					pivot = row;
				}
			}
			if (pivot != col)
			{
				// Exchange "pivot" with "col" row (this is no less efficient
				// than having to perform all array accesses indirectly)
				for (i = 0; i < size; i++)
				{
					temp = sq[pivot][i];
					sq[pivot][i] = sq[col][i];
					sq[col][i] = temp;
				}
				temp = solution[pivot];
				solution[pivot] = solution[col];
				solution[col] = temp;
			}
			// Do reduction for this column
			for (row = col + 1; row < size; row++)
			{
				mult = sq[row][col] / sq[col][col];
				for (c = col; c < size; c++) //Could start with c=col+1
					sq[row][c] -= mult * sq[col][c];
				solution[row] -= mult * solution[col];
			}
		}

		// Do back substitution.  Pivoting does not affect solution order
		for (row = size - 1; row >= 0; row--)
		{
			val = solution[row];
			for (col = size - 1; col > row; col--)
				val -= solution[col] * sq[row][col];
			solution[row] = val / sq[row][row];
		}
	}


	/*__device__ float fitQuadrat(float* src, float* _offsets, const int s, const  int r, const int c, const int o)
	{
		int* osizes = interp_params + MAX_OCTAVE;
		int* offsets = osizes + MAX_OCTAVE;
		float* curr_octave = src + offsets[o];
		float* curr_scale = curr_octave + s * osizes[o];
		float* prev_scale = curr_scale - osizes[o];
		float* next_scale = curr_scale + osizes[o];

		// variables to fit quadratic
		float _g[3], _H[3][3];

		// Fill in the values of the gradient from pixel differences
		int idx = r * whps[o].z + c;
		int idx_nr = idx + whps[o].z;
		int idx_pr = idx - whps[o].z;
		int idx_nc = idx + 1;
		int idx_pc = idx - 1;
		_g[0] = (next_scale[idx] - prev_scale[idx]) * 0.5f;
		_g[1] = (curr_scale[idx_nr] - curr_scale[idx_pr]) * 0.5f;
		_g[2] = (curr_scale[idx_nc] - curr_scale[idx_pc]) * 0.5f;

		// Fill in the values of the Hessian from pixel differences
		const float temp = curr_scale[idx] + curr_scale[idx];
		_H[0][0] = prev_scale[idx] + next_scale[idx] - temp;
		_H[1][1] = curr_scale[idx_nr] + curr_scale[idx_pr] - temp;
		_H[2][2] = curr_scale[idx_nc] + curr_scale[idx_pc] - temp;
		_H[0][1] = ((next_scale[idx_nr] - next_scale[idx_pr]) -
			(prev_scale[idx_nr] - prev_scale[idx_pr])) * 0.25f;
		_H[0][2] = ((next_scale[idx_nc] - next_scale[idx_pc]) - 
			(prev_scale[idx_nc] - prev_scale[idx_pc])) * 0.25f;
		_H[1][2] = ((curr_scale[idx_nr + 1] - curr_scale[idx_nr - 1]) -
			(curr_scale[idx_pr + 1] - curr_scale[idx_pr - 1])) * 0.25f;
		_H[1][0] = _H[0][1]; 
		_H[2][0] = _H[0][2];
		_H[2][1] = _H[1][2];
		_offsets[0] = -_g[0];
		_offsets[1] = -_g[1];
		_offsets[2] = -_g[2];
		solveLinearSystem(_offsets, _H, 3);
		//printf("o=%d,s=%d,r=%d,c=%d,strength=%f,_g(%f,%f,%f),_offsets(%f,%f,%f),_H(%f,%f,%f,%f,%f,%f,%f,%f,%f)\n",
		//	o, s, r, c, 0, _g[0], _g[1], _g[2], _offsets[0], _offsets[1], _offsets[2], _H[0][0], _H[0][1],
		//	_H[0][2], _H[1][0], _H[1][1], _H[1][2], _H[2][0], _H[2][1], _H[2][2]);

		// Also return value of the determinant at peak location using initial
		// value plus 0.5 times linear interpolation with gradient to peak
		// position (this is correct for a quadratic approximation).
		float strength = curr_scale[idx] + 0.5f * (_offsets[0] * _g[0] + _offsets[1] * _g[1] + _offsets[2] * _g[2]);
		return strength;
	}*/


	__device__ float fitQuadrat(float* src, float* _offsets, float* _g, float _H[3][3], const int s, const  int r, const int c, const int osize)
	{
		float* curr_scale = src + s * osize;
		float* prev_scale = curr_scale - osize;
		float* next_scale = curr_scale + osize;

		// variables to fit quadratic
		//float _g[3], _H[3][3];

		// Fill in the values of the gradient from pixel differences
		int idx = r * whps[0].z + c;
		int idx_nr = idx + whps[0].z;
		int idx_pr = idx - whps[0].z;
		int idx_nc = idx + 1;
		int idx_pc = idx - 1;
		_g[0] = (next_scale[idx] - prev_scale[idx]) * 0.5f;
		_g[1] = (curr_scale[idx_nr] - curr_scale[idx_pr]) * 0.5f;
		_g[2] = (curr_scale[idx_nc] - curr_scale[idx_pc]) * 0.5f;

		// Fill in the values of the Hessian from pixel differences
		const float temp = curr_scale[idx] + curr_scale[idx];
		_H[0][0] = prev_scale[idx] + next_scale[idx] - temp;
		_H[1][1] = curr_scale[idx_nr] + curr_scale[idx_pr] - temp;
		_H[2][2] = curr_scale[idx_nc] + curr_scale[idx_pc] - temp;
		_H[0][1] = ((next_scale[idx_nr] - next_scale[idx_pr]) -
			(prev_scale[idx_nr] - prev_scale[idx_pr])) * 0.25f;
		_H[0][2] = ((next_scale[idx_nc] - next_scale[idx_pc]) -
			(prev_scale[idx_nc] - prev_scale[idx_pc])) * 0.25f;
		_H[1][2] = ((curr_scale[idx_nr + 1] - curr_scale[idx_nr - 1]) -
			(curr_scale[idx_pr + 1] - curr_scale[idx_pr - 1])) * 0.25f;
		_H[1][0] = _H[0][1];
		_H[2][0] = _H[0][2];
		_H[2][1] = _H[1][2];
		_offsets[0] = -_g[0];
		_offsets[1] = -_g[1];
		_offsets[2] = -_g[2];
		solveLinearSystem(_offsets, _H, 3);
		//printf("o=%d,s=%d,r=%d,c=%d,strength=%f,_g(%f,%f,%f),_offsets(%f,%f,%f),_H(%f,%f,%f,%f,%f,%f,%f,%f,%f)\n",
		//	o, s, r, c, 0, _g[0], _g[1], _g[2], _offsets[0], _offsets[1], _offsets[2], _H[0][0], _H[0][1],
		//	_H[0][2], _H[1][0], _H[1][1], _H[1][2], _H[2][0], _H[2][1], _H[2][2]);

		// Also return value of the determinant at peak location using initial
		// value plus 0.5 times linear interpolation with gradient to peak
		// position (this is correct for a quadratic approximation).
		float strength = curr_scale[idx] + 0.5f * (_offsets[0] * _g[0] + _offsets[1] * _g[1] + _offsets[2] * _g[2]);
		return strength;
	}


	__device__ void updateIradius(float scale)
	{
		scale = (d_interp_struct.doubled ? 3.3f : 1.65f) * scale;
		const int step = max(1, __float2int_rn(scale * 0.5f));
		const float radius = (d_interp_struct.upright ? 1 : 1.4f) * scale * d_interp_struct.mag_factor * (d_interp_struct.desc_wsz + 1) * 0.5f;
		const int iradius = __float2int_rn(radius / step);
		atomicMax(&d_iradius, iradius);
	}


	__device__ void makePoint(int* iimage, SurfPoint* point, float x, float y, float scale, float strength)
	{
		float temp_delta = d_interp_struct.sampling * d_interp_struct.divisor;
		point->x = x * temp_delta;
		point->y = y * temp_delta;
		point->scale = 1.2f * scale * d_interp_struct.divisor;
		point->strength = strength;
		point->ori = 0.f;
		int vas[9];
		int temp = __float2int_rz(__fmaf_rn(3, scale, 0.5f));
		vas[0] = __float2int_rz(__fmaf_rn(x, d_interp_struct.sampling, 0.5f));
		vas[1] = __float2int_rz(__fmaf_rn(y, d_interp_struct.sampling, 0.5f));
		vas[2] = temp / 2;
		vas[3] = vas[2] + vas[2];
		vas[4] = vas[2] + vas[3];
		vas[5] = vas[0] + temp;
		vas[6] = vas[0] - temp;
		vas[7] = vas[1] + temp;
		vas[8] = vas[1] - temp;
		point->laplace = getTrace(iimage, vas, whps[1].z);
		updateIradius(point->scale);
	}


	/*__device__ int dInterpFeature(int* iimage, float* src, SurfPoint* point, int s, int row, int col, int o, int moves_remain, int* borders)
	{
		int newr = row, newc = col;
		float _offsets[3] = { 0 };
		float strength = 0;
		do
		{
			row = newr; col = newc;
			strength = fitQuadrat(src, _offsets, s, row, col, o);
			if (_offsets[1] > 0.6f && row < whps[o].y - borders[s]) newr++;
			if (_offsets[1] < -0.6f && row > borders[s]) newr--;
			if (_offsets[2] > 0.6f && col < whps[o].x - borders[s]) newc++;
			if (_offsets[2] < -0.6f && col > borders[s]) newc--;
		} while (moves_remain-- > 0 && (newr != row || newc != col));

		if (__isnanf(_offsets[0]) || __isnanf(_offsets[1]) || __isnanf(_offsets[2]))
		{
			//printf("NaN\n");
			return 0;
		}
		if (fabs(_offsets[0]) > 1.5f || fabs(_offsets[1]) > 1.5f || fabs(_offsets[2]) > 1.5f)
		{
			//printf("Offset\n");
			return 0;
		}
		if (strength < d_interp_struct.thresh)
		{
			//printf("Strength = %f < %f\n", strength, d_interp_struct.thresh);
			return 0;
		}

		//if (__isnanf(_offsets[0]) || __isnanf(_offsets[1]) || __isnanf(_offsets[2]) ||
		//	fabs(_offsets[0]) > 1.5f || fabs(_offsets[1]) > 1.5f || fabs(_offsets[2]) > 1.5f ||
		//	strength < d_interp_struct.thresh)
		//{
		//	return 0;
		//}

		// Check that no ipoint has been created at this location (to avoid
		// duplicates). Otherwise, mark this map location
		int octave = interp_params[o];
		float ns = (d_interp_struct.init_lobe + (octave - 1) * d_interp_struct.max_scale + (s + _offsets[0]) * 2 * octave) / 3.f;
		float ny = octave * (row + _offsets[1]);
		float nx = octave * (col + _offsets[2]);
		makePoint(iimage, point, nx, ny, ns, strength);
		return 1;
	}*/


	/*__global__ void interpFeature(int* iimage, float* src, SurfPoint* points, bool* flags, int moves_remain, int nkpts)
	{
		unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i >= nkpts)
			return;

		SurfPoint* p = &points[i];
		int o = p->o;
		int s = __float2int_rn(p->scale);
		int row = __float2int_rn(p->y);
		int col = __float2int_rn(p->x);
		int newr = row;
		int newc = col;
		int* borders = interp_params + 3 * MAX_OCTAVE + o * MAX_SCALE;

		// Interpolate the detected maximum in order to get a more accurate location
		float _offsets[3] = { 0 };
		float strength = 0;
		//const float strength = fitQuadrat(src, _offsets, s, row, col, o);
		//if (_offsets[1] > 0.6f && row < whps[o].y - borders[s]) newr++;
		//if (_offsets[1] < -0.6f && row > borders[s]) newr--;
		//if (_offsets[2] > 0.6f && col < whps[o].x - borders[s]) newc++;
		//if (_offsets[2] < -0.6f && col > borders[s]) newc--;
		////printf("o=%d,s=%d,r=%d,c=%d,newr=%d,newc=%d,strength=%f\n", o, s, row, col, newr, newc, strength);
		//if (moves_remain > 0 && (newr != row || newc != col))
		//{
		//	flags[i] = dInterpFeature(iimage, src, p, s, newr, newc, o, moves_remain - 1, borders);
		//	return;
		//}
		for (int mv = 0; mv < moves_remain; mv++)
		{
			row = newr; col = newc;
			strength = fitQuadrat(src, _offsets, s, row, col, o);
			//printf("moves_remain=%d,o=%d,s=%d,r=%d,c=%d,newr=%d,newc=%d,strength=%f\n", mv, o, s, row, col, newr, newc, strength);
			if (_offsets[1] > 0.6f && row < whps[o].y - borders[s]) newr++;
			if (_offsets[1] < -0.6f && row > borders[s]) newr--;
			if (_offsets[2] > 0.6f && col < whps[o].x - borders[s]) newc++;
			if (_offsets[2] < -0.6f && col > borders[s]) newc--;
			if ((newr == row && newc == col)) break;
		}

		// Do not create a keypoint if interpolation still remains far
		// outside expected limits, or if magnitude of peak value is below
		// threshold (i.e., contrast is too low)		
		if (__isnanf(_offsets[0]) || __isnanf(_offsets[1]) || __isnanf(_offsets[2]))
		{
			//printf("NaN\n");
			return;
		}
		if (fabs(_offsets[0]) > 1.5f || fabs(_offsets[1]) > 1.5f || fabs(_offsets[2]) > 1.5f)
		{
			//printf("Offset\n");
			return;
		}
		if (strength < d_interp_struct.thresh)
		{
			//printf("Strength = %f < %f\n", strength, d_interp_struct.thresh);
			return;
		}
		//if (__isnanf(_offsets[0]) || __isnanf(_offsets[1]) || __isnanf(_offsets[2]) ||
		//	fabs(_offsets[0]) > 1.5f || fabs(_offsets[1]) > 1.5f || fabs(_offsets[2]) > 1.5f ||
		//	strength < d_interp_struct.thresh)
		//{
		//	return;
		//}

		// Check that no ipoint has been created at this location (to avoid
		// duplicates). Otherwise, mark this map location
		int octave = interp_params[o];
		float ns = (d_interp_struct.init_lobe + (octave - 1) * d_interp_struct.max_scale + (s + _offsets[0]) * 2 * octave) / 3.f;
		float ny = octave * (row + _offsets[1]);
		float nx = octave * (col + _offsets[2]);
		makePoint(iimage, p, nx, ny, ns, strength);
		flags[i] = true;
	}*/


	/*__global__ void reassign(SurfPoint* points, bool* indexes, int num)
	{
		int s = 0;
		for (int i = 0; i < num; i++)
		{
			if (indexes[i])
			{
				points[s].o = points[i].o;
				points[s].scale = points[i].scale;
				points[s].x = points[i].x;
				points[s].y = points[i].y;
				points[s].laplace = points[i].laplace;
				points[s].strength = points[i].strength;
				s++;
			}
		}
		d_point_counter[MAX_OCTAVE - 1] = s;
	}*/


	inline __device__ int getWavelet1(int* iimage, int x, int y, int size, int p)
	{
		return getSum(iimage, x + size, y, x - size, y - size, p) - 
			getSum(iimage, x + size, y + size, x - size, y, p);
	}


	inline __device__ int getWavelet2(int* iimage, int x, int y, int size, int p)
	{
		return getSum(iimage, x + size, y + size, x, y - size, p) -
			getSum(iimage, x, y + size, x - size, y - size, p);
	}


	inline __device__ int getWavelet1(cudaTextureObject_t iimage, int x, int y, int size, int p)
	{
		return getSum(iimage, x + size, y, x - size, y - size) -
			getSum(iimage, x + size, y + size, x - size, y);
	}


	inline __device__ int getWavelet2(cudaTextureObject_t iimage, int x, int y, int size, int p)
	{
		return getSum(iimage, x + size, y + size, x, y - size) -
			getSum(iimage, x, y + size, x - size, y - size);
	}


	__device__ void placeInIndex(float* descriptor, float mag1, int ori1, float mag2, int ori2, float rx, float cx)
	{
		int ri = __float2int_rz(rx >= 0.f ? rx : rx - 1.f);
		int ci = __float2int_rz(cx >= 0.f ? cx : cx - 1.f);
		float rfrac = rx - ri;
		float cfrac = cx - ci;
		float cfrac1 = 1 - cfrac;
		int r_index, c_index, ostart;
		float rweight1, cweight1, rweight2, cweight2;

		r_index = ri;
		if (r_index >= 0)
		{
			rweight1 = mag1 * (1.f - rfrac);
			rweight2 = mag2 * (1.f - rfrac);

			c_index = ci;
			if (c_index >= 0)
			{
				cweight1 = rweight1 * cfrac1;
				cweight2 = rweight2 * cfrac1;
				ostart = r_index * d_interp_struct.desc_wsz * d_interp_struct.orient_size +
					c_index * d_interp_struct.orient_size;
				atomicAdd(&descriptor[ostart + ori1], cweight1);
				atomicAdd(&descriptor[ostart + ori2], cweight2);
				//descriptor[ostart + ori1] += cweight1;
				//descriptor[ostart + ori2] += cweight2;
			}
			c_index++;
			if (c_index < d_interp_struct.desc_wsz)
			{
				cweight1 = rweight1 * cfrac;
				cweight2 = rweight2 * cfrac;
				ostart = r_index * d_interp_struct.desc_wsz * d_interp_struct.orient_size +
					c_index * d_interp_struct.orient_size;
				atomicAdd(&descriptor[ostart + ori1], cweight1);
				atomicAdd(&descriptor[ostart + ori2], cweight2);
				//descriptor[ostart + ori1] += cweight1;
				//descriptor[ostart + ori2] += cweight2;
			}
		}
		r_index++;
		if (r_index < d_interp_struct.desc_wsz)
		{
			rweight1 = mag1 * rfrac;
			rweight2 = mag2 * rfrac;

			c_index = ci;
			if (c_index >= 0)
			{
				cweight1 = rweight1 * cfrac1;
				cweight2 = rweight2 * cfrac1;
				ostart = r_index * d_interp_struct.desc_wsz * d_interp_struct.orient_size +
					c_index * d_interp_struct.orient_size;
				atomicAdd(&descriptor[ostart + ori1], cweight1);
				atomicAdd(&descriptor[ostart + ori2], cweight2);
				//descriptor[ostart + ori1] += cweight1;
				//descriptor[ostart + ori2] += cweight2;
			}
			c_index++;
			if (c_index < d_interp_struct.desc_wsz)
			{
				cweight1 = rweight1 * cfrac;
				cweight2 = rweight2 * cfrac;
				ostart = r_index * d_interp_struct.desc_wsz * d_interp_struct.orient_size +
					c_index * d_interp_struct.orient_size;
				atomicAdd(&descriptor[ostart + ori1], cweight1);
				atomicAdd(&descriptor[ostart + ori2], cweight2);
				//descriptor[ostart + ori1] += cweight1;
				//descriptor[ostart + ori2] += cweight2;
			}
		}
	}


	struct DesParamUR
	{
		int ix;
		int iy;
		int iradius;
		int scale;
		int step;
		float dx;
		float dy;
		float spacing;
		float wofs;
	};


	__device__ void addUprightSample(int* iimage, float* sdesc, DesParamUR* dpur, int i, int j)
	{
		float rpos = (dpur->step * i - dpur->dy) / dpur->spacing;
		float cpos = (dpur->step * j - dpur->dx) / dpur->spacing;
		//float wofs = d_interp_struct.desc_wsz * 0.5f - 0.5f;
		float rx = rpos + dpur->wofs;
		float cx = cpos + dpur->wofs;
		if (rx > -1.f && rx < d_interp_struct.desc_wsz && cx > -1.f && cx < d_interp_struct.desc_wsz)
		{
			int r = dpur->iy + i * dpur->step;
			int c = dpur->ix + j * dpur->step;
			//int step = __float2int_rz(scale);
			if (r >= 1 + dpur->scale && r < whps[1].y - 1 - dpur->scale && c >= 1 + dpur->scale && c < whps[1].x - 1 - dpur->scale)
			{
				// Compute descriptor
				float weight = lookup2[__float2int_rz(rpos * rpos + cpos * cpos)];
				float dx = weight * getWavelet2(iimage, c, r, dpur->scale, whps[1].z) * 0.003921568627f;
				float dy = weight * getWavelet1(iimage, c, r, dpur->scale, whps[1].z) * 0.003921568627f;
				if (!d_interp_struct.extend)
				{
					placeInIndex(sdesc, dx, (dx < 0 ? 0 : 1), dy, (dy < 0 ? 2 : 3), rx, cx);
				}
				else
				{
					placeInIndex(sdesc, dx, (dy < 0 ? 0 : 1), fabs(dx), (dy < 0 ? 2 : 3), rx, cx);
					placeInIndex(sdesc, dy, (dx < 0 ? 4 : 5), fabs(dy), (dx < 0 ? 6 : 7), rx, cx);
				}
			}
		}
	}


	__device__ void addUprightSample(cudaTextureObject_t iimage, float* sdesc, DesParamUR* dpur, int i, int j)
	{
		float rpos = (dpur->step * i - dpur->dy) / dpur->spacing;
		float cpos = (dpur->step * j - dpur->dx) / dpur->spacing;
		//float wofs = d_interp_struct.desc_wsz * 0.5f - 0.5f;
		float rx = rpos + dpur->wofs;
		float cx = cpos + dpur->wofs;
		if (rx > -1.f && rx < d_interp_struct.desc_wsz && cx > -1.f && cx < d_interp_struct.desc_wsz)
		{
			int r = dpur->iy + i * dpur->step;
			int c = dpur->ix + j * dpur->step;
			//int step = __float2int_rz(scale);
			if (r >= 1 + dpur->scale && r < whps[1].y - 1 - dpur->scale && c >= 1 + dpur->scale && c < whps[1].x - 1 - dpur->scale)
			{
				// Compute descriptor
				float weight = lookup2[__float2int_rz(rpos * rpos + cpos * cpos)];
				float dx = weight * getWavelet2(iimage, c, r, dpur->scale, whps[1].z) * 0.003921568627f;
				float dy = weight * getWavelet1(iimage, c, r, dpur->scale, whps[1].z) * 0.003921568627f;
				if (!d_interp_struct.extend)
				{
					placeInIndex(sdesc, dx, (dx < 0 ? 0 : 1), dy, (dy < 0 ? 2 : 3), rx, cx);
				}
				else
				{
					placeInIndex(sdesc, dx, (dy < 0 ? 0 : 1), fabs(dx), (dy < 0 ? 2 : 3), rx, cx);
					placeInIndex(sdesc, dy, (dx < 0 ? 4 : 5), fabs(dy), (dx < 0 ? 6 : 7), rx, cx);
				}
			}
		}
	}


	__global__ void describeUR(int* iimage, float* descriptors, SurfPoint* points)
	{
		extern __shared__ float sdesc[];
		__shared__ DesParamUR dpur;

		// Get keypoint
		unsigned int pi = blockIdx.x;
		if (pi >= d_point_counter[d_interp_struct.noctaves - 1])
			return;
		SurfPoint* p = &points[pi];

		// Prepare shared parameters for feature computation
		unsigned int bsz = blockDim.x * blockDim.y;
		unsigned int tid = threadIdx.y * blockDim.x + threadIdx.x;
		if (tid == 0)
		{
			float x, y, scale;
			if (d_interp_struct.doubled)
			{
				x = p->x + p->x;
				y = p->y + p->y;
				scale = 3.3f * p->scale;
			}
			else
			{
				x = p->x;
				y = p->y;
				scale = 1.65f * p->scale;
			}
			dpur.step = max(__float2int_rn(scale * 0.5f), 1);
			dpur.ix = __float2int_rn(x);
			dpur.iy = __float2int_rn(y);
			dpur.dx = x - dpur.ix;
			dpur.dy = y - dpur.iy;
			dpur.spacing = scale * d_interp_struct.mag_factor;
			dpur.iradius = __float2int_rn(dpur.spacing * (d_interp_struct.desc_wsz + 1) * 0.5f / dpur.step);
			dpur.scale = __float2int_rz(scale);
			dpur.wofs = d_interp_struct.desc_wsz * 0.5f - 0.5f;
		}
		__syncthreads();

		// Initialization
		for (unsigned int i = tid; i < d_interp_struct.nfeatures; i += bsz)
		{
			sdesc[i] = 0.f;
		}

		// Add samples
		for (int i = threadIdx.y - dpur.iradius; i <= dpur.iradius; i += blockDim.y)
		{
			for (int j = threadIdx.x - dpur.iradius; j <= dpur.iradius; j += blockDim.x)
			{
				addUprightSample(iimage, sdesc, &dpur, i, j);
			}
		}
		__syncthreads();

		// sqaure
		float* sdesc_sq = sdesc + d_interp_struct.nfeatures;
		for (unsigned int i = tid; i < d_interp_struct.nfeatures; i += bsz)
		{
			sdesc_sq[i] = sdesc[i] * sdesc[i];
		}
		__syncthreads();

		// Compute square sum
		for (int stride = d_interp_struct.nfeatures / 2; stride > 32; stride = stride >> 1)
		{
			if (tid < stride)
			{
				sdesc_sq[tid] += sdesc_sq[tid + stride];
			}
			__syncthreads();
		}
		if (tid < 32)
		{
			volatile float* vmem = sdesc_sq;
			vmem[tid] += vmem[tid + 32];
			vmem[tid] += vmem[tid + 16];
			vmem[tid] += vmem[tid + 8];
			vmem[tid] += vmem[tid + 4];
			vmem[tid] += vmem[tid + 2];
			vmem[tid] += vmem[tid + 1];
		}
		__syncthreads();

		// Normalization
		float fac = 1.f / __fsqrt_rn(sdesc_sq[0]);
		float* desc = descriptors + pi * d_interp_struct.nfeatures;
		for (unsigned int i = tid; i < d_interp_struct.nfeatures; i += bsz)
		{
			desc[i] = sdesc[i] * fac;
		}
	}


	__global__ void describeURMulti(int* iimage, float* descriptors, SurfPoint* points)
	{
		extern __shared__ float sdesc[];
		__shared__ DesParamUR dpur;

		// Get keypoint
		unsigned int pi = blockIdx.z;		
		if (pi >= d_point_counter[d_interp_struct.noctaves - 1])
			return;
		SurfPoint* p = &points[pi];

		// Get shared parameters for feature computation
		float x, y, scale;
		if (d_interp_struct.doubled)
		{
			x = p->x + p->x;
			y = p->y + p->y;
			scale = 3.3f * p->scale;
		}
		else
		{
			x = p->x;
			y = p->y;
			scale = 1.65f * p->scale;
		}		
		dpur.step = max(__float2int_rn(scale * 0.5f), 1);
		dpur.ix = __float2int_rn(x);
		dpur.iy = __float2int_rn(y);
		dpur.dx = x - dpur.ix;
		dpur.dy = y - dpur.iy;
		dpur.spacing = scale * d_interp_struct.mag_factor;
		dpur.iradius = __float2int_rn(dpur.spacing * (d_interp_struct.desc_wsz + 1) * 0.5f / dpur.step);
		dpur.scale = __float2int_rz(scale);
		dpur.wofs = d_interp_struct.desc_wsz * 0.5f - 0.5f;
		__syncthreads();

		// Check neighborhoods
		int idx = blockIdx.x * blockDim.x + threadIdx.x;
		int idy = blockIdx.y * blockDim.y + threadIdx.y;
		int i = idy - dpur.iradius;
		int j = idx - dpur.iradius;
		if (i <= dpur.iradius && j <= dpur.iradius)
		{
			addUprightSample(iimage, sdesc, &dpur, i, j);
		}

		// Unroll 4
		//if (idx == 0)
		//{
		//	if (idy <= dpur.iradius)
		//	{
		//		addUprightSample(iimage, sdesc, &dpur, idx, idy, step);
		//		if (idy > 0)
		//		{
		//			addUprightSample(iimage, sdesc, &dpur, idx, -idy, step);
		//		}
		//	}
		//}
		//else if (idx <= dpur.iradius)
		//{
		//	if (idy <= dpur.iradius)
		//	{
		//		addUprightSample(iimage, sdesc, &dpur, idx, idy, step);
		//		addUprightSample(iimage, sdesc, &dpur, -idx, idy, step);
		//		if (idy > 0)
		//		{
		//			addUprightSample(iimage, sdesc, &dpur, idx, -idy, step);
		//			addUprightSample(iimage, sdesc, &dpur, -idx, -idy, step);
		//		}
		//	}
		//}
		__syncthreads();

		if (blockIdx.x > 0 || blockIdx.y > 0)
		{
			return;
		}

		// sqaure
		float* sdesc_sq = sdesc + d_interp_struct.nfeatures;
		unsigned int tid = threadIdx.y * blockDim.x + threadIdx.x;
		if (tid < d_interp_struct.nfeatures)
		{
			sdesc_sq[tid] = sdesc[tid] * sdesc[tid];
		}
		__syncthreads();
		
		// Compute square sum
		for (int stride = d_interp_struct.nfeatures / 2; stride > 32; stride = stride >> 1)
		{
			if (tid < stride)
			{
				sdesc_sq[tid] += sdesc_sq[tid + stride];
			}
			__syncthreads();
		}
		if (tid < 32)
		{
			volatile float* vmem = sdesc_sq;
			vmem[tid] += vmem[tid + 32];
			vmem[tid] += vmem[tid + 16];
			vmem[tid] += vmem[tid + 8];
			vmem[tid] += vmem[tid + 4];
			vmem[tid] += vmem[tid + 2];
			vmem[tid] += vmem[tid + 1];
		}
		__syncthreads();

		// Normalization
		float fac = 1.f / __fsqrt_rn(sdesc_sq[0]);
		float* desc = descriptors + pi * d_interp_struct.nfeatures;
		if (tid < d_interp_struct.nfeatures)
		{
			desc[tid] = sdesc[tid] * fac;
		}
	}


	__global__ void describeURWithoutNormalization(int* iimage, float* descriptors, SurfPoint* points)
	{
		__shared__ DesParamUR dpur;

		// Get keypoint
		unsigned int pi = blockIdx.z;
		if (pi >= d_point_counter[d_interp_struct.noctaves - 1])
			return;
		SurfPoint* p = &points[pi];

		// Get shared parameters for feature computation
		unsigned int tid = threadIdx.y * blockDim.x + threadIdx.x;
		if (tid == 0)
		{
			float x, y, scale;
			if (d_interp_struct.doubled)
			{
				x = p->x + p->x;
				y = p->y + p->y;
				scale = 3.3f * p->scale;
			}
			else
			{
				x = p->x;
				y = p->y;
				scale = 1.65f * p->scale;
			}
			dpur.step = max(__float2int_rn(scale * 0.5f), 1);
			dpur.ix = __float2int_rn(x);
			dpur.iy = __float2int_rn(y);
			dpur.dx = x - dpur.ix;
			dpur.dy = y - dpur.iy;
			dpur.spacing = scale * d_interp_struct.mag_factor;
			dpur.iradius = __float2int_rn(dpur.spacing * (d_interp_struct.desc_wsz + 1) * 0.5f / dpur.step);
			dpur.scale = __float2int_rz(scale);
			dpur.wofs = d_interp_struct.desc_wsz * 0.5f - 0.5f;
		}
		__syncthreads();

		// Check neighborhoods
		unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
		unsigned int idy = blockIdx.y * blockDim.y + threadIdx.y;
		int i = idy - dpur.iradius;
		int j = idx - dpur.iradius;
		if (i <= dpur.iradius && j <= dpur.iradius)
		{
			float* curr_descriptor = descriptors + pi * d_interp_struct.nfeatures;
			addUprightSample(iimage, curr_descriptor, &dpur, i, j);
		}
	}


	__global__ void describeURMulti(cudaTextureObject_t iimage, float* descriptors, SurfPoint* points)
	{
		extern __shared__ float sdesc[];
		__shared__ DesParamUR dpur;

		// Get keypoint
		unsigned int pi = blockIdx.z;
		if (pi >= d_point_counter[d_interp_struct.noctaves - 1])
			return;
		SurfPoint* p = &points[pi];

		// Get shared parameters for feature computation
		float x, y, scale;
		if (d_interp_struct.doubled)
		{
			x = p->x + p->x;
			y = p->y + p->y;
			scale = 3.3f * p->scale;
		}
		else
		{
			x = p->x;
			y = p->y;
			scale = 1.65f * p->scale;
		}
		dpur.step = max(__float2int_rn(scale * 0.5f), 1);
		dpur.ix = __float2int_rn(x);
		dpur.iy = __float2int_rn(y);
		dpur.dx = x - dpur.ix;
		dpur.dy = y - dpur.iy;
		dpur.spacing = scale * d_interp_struct.mag_factor;
		dpur.iradius = __float2int_rn(dpur.spacing * (d_interp_struct.desc_wsz + 1) * 0.5f / dpur.step);
		dpur.scale = __float2int_rz(scale);
		dpur.wofs = d_interp_struct.desc_wsz * 0.5f - 0.5f;
		__syncthreads();

		// Check neighborhoods
		int idx = blockIdx.x * blockDim.x + threadIdx.x;
		int idy = blockIdx.y * blockDim.y + threadIdx.y;
		int i = idy - dpur.iradius;
		int j = idx - dpur.iradius;
		if (i <= dpur.iradius && j <= dpur.iradius)
		{
			addUprightSample(iimage, sdesc, &dpur, i, j);
		}

		__syncthreads();

		if (blockIdx.x > 0 || blockIdx.y > 0)
		{
			return;
		}

		// sqaure
		float* sdesc_sq = sdesc + d_interp_struct.nfeatures;
		unsigned int tid = threadIdx.y * blockDim.x + threadIdx.x;
		if (tid < d_interp_struct.nfeatures)
		{
			sdesc_sq[tid] = sdesc[tid] * sdesc[tid];
		}
		__syncthreads();

		// Compute square sum
		for (int stride = d_interp_struct.nfeatures / 2; stride > 32; stride = stride >> 1)
		{
			if (tid < stride)
			{
				sdesc_sq[tid] += sdesc_sq[tid + stride];
			}
			__syncthreads();
		}
		if (tid < 32)
		{
			volatile float* vmem = sdesc_sq;
			vmem[tid] += vmem[tid + 32];
			vmem[tid] += vmem[tid + 16];
			vmem[tid] += vmem[tid + 8];
			vmem[tid] += vmem[tid + 4];
			vmem[tid] += vmem[tid + 2];
			vmem[tid] += vmem[tid + 1];
		}
		__syncthreads();

		// Normalization
		float fac = 1.f / __fsqrt_rn(sdesc_sq[0]);
		float* desc = descriptors + pi * d_interp_struct.nfeatures;
		if (tid < d_interp_struct.nfeatures)
		{
			desc[tid] = sdesc[tid] * fac;
		}
	}


	__global__ void assignOrientationApprox(int* iimage, float* descriptors, SurfPoint* points)
	{
#define PASZ (NBIN + hwn + hwn)
		__shared__ int oparam[4];
		__shared__ int hist[NBIN];
		__shared__ float avg_angles[NBIN];
		__shared__ float part_sums[NBIN];
		__shared__ float part_angle_sums[PASZ];
		__shared__ float win_sums[NBIN];
		__shared__ float win_angle_sums[NBIN];
		
		// Stage1 -- Approximate major orientation
		// Get keypoint
		unsigned int pi = blockIdx.x; //  blockIdx.z* gridDim.y* gridDim.x + blockIdx.y * gridDim.x + blockIdx.x;
		if (pi >= d_point_counter[d_interp_struct.noctaves - 1])
			return;
		SurfPoint* p = &points[pi];

		// Approximate major orientation
		unsigned int tid = threadIdx.y * blockDim.x + threadIdx.x;
		if (tid == 0)
		{
			float scale, x, y;
			if (d_interp_struct.doubled)
			{
				x = p->x + p->x;
				y = p->y + p->y;
				scale = p->scale + p->scale;
			}
			else
			{
				x = p->x;
				y = p->y;
				scale = p->scale;
			}
			oparam[0] = __float2int_rz(2 * scale + 1.6f);
			oparam[1] = __float2int_rz(scale + 0.8f);
			oparam[2] = __float2int_rn(x);
			oparam[3] = __float2int_rn(y);
		}
		__syncthreads();

		// Initialization
		if (tid < NBIN)
		{
			hist[tid] = 0;
			avg_angles[tid] = 0.f;
			part_sums[tid] = 0.f;
			win_sums[tid] = 0.f;
			win_angle_sums[tid] = 0.f;
		}
		if (tid < PASZ)
		{
			part_angle_sums[tid] = 0.f;
		}
		__syncthreads();
		//if (tid == 0 && pi < 10)
		//{
		//	printf("part_angle_sums:%f,%f,%f,%f,%f,%f,%f,%f\n", part_angle_sums[0], part_angle_sums[1], part_angle_sums[2], 
		//		part_angle_sums[3], part_angle_sums[4], part_angle_sums[5], part_angle_sums[6], part_angle_sums[7]);
		//}

		for (int y1 = threadIdx.y - oradius; y1 <= oradius; y1 += blockDim.y)
		{
			for (int x1 = threadIdx.x - oradius; x1 <= oradius; x1 += blockDim.x)
			{
				int xx = oparam[2] + x1 * oparam[1];
				int yy = oparam[3] + y1 * oparam[1];
				// Do not use last row or column, which are not valid
				if (yy + oparam[0] + 2 < whps[1].y && yy - oparam[0] > -1 &&
					xx + oparam[0] + 2 < whps[1].x && xx - oparam[0] > -1)
				{
					int distsq = y1 * y1 + x1 * x1;
					if (distsq < oradiussq)
					{
						float dx = getWavelet2(iimage, xx, yy, oparam[0], whps[1].z) * 0.003921568627f;;	//  *0.003921568627f;
						float dy = getWavelet1(iimage, xx, yy, oparam[0], whps[1].z) * 0.003921568627f;;
						float magnitude = __fsqrt_rn(dx * dx + dy * dy);
						if (magnitude > 0.f)
						{
							float weight = lookup1[distsq];
							float angle = dFastAtan2(dy, dx); // atan2f(dy, dx);// dFastAtan2(dy, dx);
							int hid = __float2int_rz((angle + M_PI) / sep_angle) % NBIN;
							float psum = weight * magnitude;
							atomicAdd(&hist[hid], 1);
							atomicAdd(&avg_angles[hid], angle);
							atomicAdd(&part_sums[hid], psum);
							atomicAdd(&part_angle_sums[hid + hwn], angle * psum);
							if (hid - hwn < 0)
							{
								atomicAdd(&part_angle_sums[hid + hwn + NBIN], (angle + 2 * M_PI) * psum);
							}
							else if (hid + hwn >= NBIN)
							{
								atomicAdd(&part_angle_sums[hid + hwn - NBIN], (angle - 2 * M_PI) * psum);
							}
							//if (pi == 0)
							//{
							//	printf("x1 = %d, y1 = %d, hid = %d, psum = %f, part_angle_sums = %f\n",
							//		x1, y1, hid, psum, part_angle_sums[hid + hwn]);
							//}
						}
					}
				}
			}
		}
		__syncthreads();

//		if (tid == 0 && pi < 10)
//		{
//			printf("s=%f, y=%f, x=%f\n\hist:%d,%d,%d,%d,%d,%d\n\\angle_sums:%f,%f,%f,%f,%f,%f\n\
//part_sums:%f,%f,%f,%f,%f,%f\n\part_angle_sums:%f,%f,%f,%f,%f,%f\n",
//				scale, y, x, 
//				hist[10], hist[11], hist[12], hist[13], hist[14], hist[15],
//				angle_sums[10], angle_sums[11], angle_sums[12], angle_sums[13], angle_sums[14], angle_sums[15], 
//				part_sums[10], part_sums[11], part_sums[12], part_sums[13], part_sums[14], part_sums[15], 
//				part_angle_sums[10], part_angle_sums[11], part_angle_sums[12], 
//				part_angle_sums[13], part_angle_sums[14], part_angle_sums[15]);
//		}

		// Avgerage angles
		if (tid < NBIN)
		{
			avg_angles[tid] = hist[tid] > 0 ? avg_angles[tid] / hist[tid] : bins[tid];
		}
		__syncthreads();

		//if (tid == 0 && pi == 0)
		//{
		//	printf("s=%f, y=%f, x=%f, \nbins: %f, %f, %f, %f, %f, %f, %f, %f\navg_angles: %f, %f, %f, %f, %f, %f, %f, %f\n", 
		//		scale, y, x, 			
		//		bins[0], bins[1], bins[2], bins[3], bins[4], bins[5], bins[6], bins[7],
		//		avg_angles[0], avg_angles[1], avg_angles[2], avg_angles[3],
		//		avg_angles[4], avg_angles[5], avg_angles[6], avg_angles[7]);
		//}

		// Compute window sum
		int i = tid / (hwn + hwn + 1);
		if (i < NBIN)
		{
			int j = tid % (hwn + hwn + 1) - hwn;
			int k = i + j;
			if (j == -hwn)	// Left
			{
				float residual_angle = 0.f;
				if (k < 0)
				{
					k += NBIN;
					int k1 = (k + 1) % NBIN;
					residual_angle = bins[k1] + (window / 2) - avg_angles[i] - (bins[k1] < 0 ? 0 : 2 * M_PI);
				}
				else
				{
					residual_angle = bins[k + 1] + (window / 2) - avg_angles[i];
				}
				float edge_ratio = residual_angle / sep_angle;
				atomicAdd(&win_sums[i], edge_ratio * part_sums[k]);
				atomicAdd(&win_angle_sums[i], edge_ratio * part_angle_sums[i]);
				//if (pi == 0)
				//{
				//	printf("s = %f, y = %f, x = %f, i = %d, j = %d, k = %d, edge_ratio = %f, ps= %f, pas = %f\n", 
				//		scale, y, x, i, j, k, edge_ratio, part_sums[k], part_angle_sums[i]);
				//}
			}
			else if (j == hwn)	// Right
			{
				float residual_angle = 0.f;
				if (k >= NBIN)
				{
					k -= NBIN;
					residual_angle = avg_angles[i] + (window / 2) - 2 * M_PI - bins[k];
				}
				else
				{
					residual_angle = avg_angles[i] + (window / 2) - bins[k];
				}
				float edge_ratio = residual_angle / sep_angle;
				atomicAdd(&win_sums[i], edge_ratio * part_sums[k]);
				atomicAdd(&win_angle_sums[i], edge_ratio * part_angle_sums[i + hwn + hwn]);
				//if (pi == 0)
				//{
				//	printf("s = %f, y = %f, x = %f, i = %d, j = %d, k = %d, edge_ratio = %f, ps= %f, pas = %f\n", 
				//		scale, y, x, i, j, k, edge_ratio, part_sums[k], part_angle_sums[i + hwn + hwn]);
				//}
			}
			else	// Middle
			{
				atomicAdd(&win_angle_sums[i], part_angle_sums[k + hwn]);
				if (k < 0) k += NBIN;
				else if (k >= NBIN) k -= NBIN;
				atomicAdd(&win_sums[i], part_sums[k]);
				//if (pi == 0)
				//{
				//	printf("s = %f, y = %f, x = %f, i = %d, j = %d, k = %d, ps= %f, pas = %f\n", 
				//		scale, y, x, i, j, k, part_sums[k], part_angle_sums[i + j + hwn]);
				//}
			}
		}
		__syncthreads();

		//if (tid == 0 && pi < 10)
		//{
		//	printf("s=%f, y=%f, x=%f\nwin_sums:%d,%d,%d,%d,%d,%d\nwin_angle_sums:%f,%f,%f,%f,%f,%f\n",
		//		scale, y, x, 
		//		win_sums[10], win_sums[11], win_sums[12], win_sums[13], win_sums[14], win_sums[15], 
		//		win_angle_sums[10], win_angle_sums[11], win_angle_sums[12], 
		//		win_angle_sums[13], win_angle_sums[14], win_angle_sums[15]);
		//}

		// Find the max value of win sum
		int residual_bin = NBIN;
		int n = __float2int_rz(__log2f(residual_bin));
		int zn = __float2int_rn(__powf(2, n));
		int offset = 0, id1 = 0, id2 = 0;
		while (residual_bin > 0)
		{
			for (int stride = zn / 2; stride > 0; stride = stride >> 1)
			{
				id1 = tid + offset;
				id2 = id1 + stride;
				if (tid < stride && win_sums[id1] < win_sums[id2])
				{
					win_sums[id1] = win_sums[id2];
					win_angle_sums[id1] = win_angle_sums[id2];
				}
				__syncthreads();
			}
			if (win_sums[0] < win_sums[offset])
			{
				win_sums[0] = win_sums[offset];
				win_angle_sums[0] = win_angle_sums[offset];
			}
			residual_bin -= zn;
			offset += zn;
			n = __float2int_rz(__log2f(residual_bin));
			zn = __float2int_rn(__powf(2, n));
		}
		__syncthreads();

		// Compute the best angle 
		if (tid == 0)
		{
			p->ori = win_angle_sums[0] / win_sums[0];
		}
		//if (tid == 0 && pi < 10)
		//{
		//	printf("s=%f, y=%f, x=%f, best_angle=%f, sine=%f, cose=%f\n", scale, y, x, p->ori, __sinf(p->ori), __cosf(p->ori));
		//}
		//__syncthreads();
	}



	struct DesParam
	{
		int ix;
		int iy;
		int iradius;
		int scale;
		int step;
		int pixsi;
		int pixsi2;
		float fracx;
		float fracy;
		float fracr;
		float fracc;
		float spacing;
		float wofs;
		float sine;
		float cose;
	};


	__device__ void addSample(int* iimage, float* sdesc, DesParam* desp, int i, int j)
	{
		float rpos = (desp->step * (desp->cose * i + desp->sine * j) - desp->fracr) / desp->spacing;
		float cpos = (desp->step * (-desp->sine * i + desp->cose * j) - desp->fracc) / desp->spacing;
		//float wofs = d_interp_struct.desc_wsz * 0.5f - 0.5f;
		float rx = rpos + desp->wofs;
		float cx = cpos + desp->wofs;
		if (rx > -1.f && rx < d_interp_struct.desc_wsz && cx > -1.f && cx < d_interp_struct.desc_wsz)
		{
			int r = desp->iy + i * desp->step;
			int c = desp->ix + j * desp->step;
			//int step = __float2int_rz(scale);
			if (r >= 1 + desp->scale && r < whps[1].y - 1 - desp->scale && c >= 1 + desp->scale && c < whps[1].x - 1 - desp->scale)
			{
				// Compute descriptor
				float weight = lookup2[__float2int_rz(rpos * rpos + cpos * cpos)];
				float dxx = weight * getWavelet2(iimage, c, r, desp->scale, whps[1].z) * 0.003921568627f;
				float dyy = weight * getWavelet1(iimage, c, r, desp->scale, whps[1].z) * 0.003921568627f;
				float dx = desp->cose * dxx + desp->sine * dyy;
				float dy = desp->sine * dxx - desp->cose * dyy;
				if (!d_interp_struct.extend)
				{
					placeInIndex(sdesc, dx, (dx < 0 ? 0 : 1), dy, (dy < 0 ? 2 : 3), rx, cx);
				}
				else
				{
					placeInIndex(sdesc, dx, (dy < 0 ? 0 : 1), fabs(dx), (dy < 0 ? 2 : 3), rx, cx);
					placeInIndex(sdesc, dy, (dx < 0 ? 4 : 5), fabs(dy), (dx < 0 ? 6 : 7), rx, cx);
				}
			}
		}
	}


	__global__ void describeApprox(int* iimage, float* descriptors, SurfPoint* points)
	{
		extern __shared__ float sdesc[];
		__shared__ DesParam desp;

		// Get keypoint
		unsigned int pi = blockIdx.x;
		if (pi >= d_point_counter[d_interp_struct.noctaves - 1])
			return;
		SurfPoint* p = &points[pi];

		// Get shared parameters for feature computation
		unsigned int bsz = blockDim.y * blockDim.x;
		unsigned int tid = threadIdx.y * blockDim.x + threadIdx.x;
		if (tid == 0)
		{
			float x, y, scale;
			if (d_interp_struct.doubled)
			{
				x = p->x + p->x;
				y = p->y + p->y;
				scale = 3.3f * p->scale;
			}
			else
			{
				x = p->x;
				y = p->y;
				scale = 1.65f * p->scale;
			}
			desp.step = max(__float2int_rn(scale * 0.5f), 1);
			desp.ix = __float2int_rn(x);
			desp.iy = __float2int_rn(y);
			desp.fracx = x - desp.ix;
			desp.fracy = y - desp.iy;
			desp.sine = __sinf(p->ori);
			desp.cose = __cosf(p->ori);
			desp.fracc = -desp.sine * desp.fracy + desp.cose * desp.fracx;
			desp.fracr = desp.cose * desp.fracy + desp.sine * desp.fracx;
			desp.spacing = scale * d_interp_struct.mag_factor;
			desp.iradius = __float2int_rn(1.4f * desp.spacing * (d_interp_struct.desc_wsz + 1) * 0.5f / desp.step);
			desp.scale = __float2int_rz(scale);
			desp.wofs = d_interp_struct.desc_wsz * 0.5f - 0.5f;
		}
		__syncthreads();

		// Initialization
		for (unsigned int i = tid; i < d_interp_struct.nfeatures; i += bsz)
		{
			sdesc[i] = 0.f;
		}
		__syncthreads();

		// Check neighborhoods
		for (int i = threadIdx.y - desp.iradius; i <= desp.iradius; i += blockDim.y)
		{
			for (int j = threadIdx.x - desp.iradius; j <= desp.iradius; j += blockDim.x)
			{
				addSample(iimage, sdesc, &desp, i, j);
			}
		}
		__syncthreads();

		//if (pi == 0 && threadIdx.x == 0 && threadIdx.y == 0)
		//{
		//	printf("s=%f, y=%f, x=%f, feature: ", p->scale, p->y, p->x);
		//	for (int fi = 0; fi < d_interp_struct.nfeatures; fi++)
		//	{
		//		printf("%f ", sdesc[fi]);
		//	}
		//	printf("\n");
		//}

		// sqaure
		float* sdesc_sq = sdesc + d_interp_struct.nfeatures;
		for (unsigned int i = tid; i < d_interp_struct.nfeatures; i += bsz)
		{
			sdesc_sq[i] = sdesc[i] * sdesc[i];
		}
		__syncthreads();

		// Compute square sum
		for (int stride = d_interp_struct.nfeatures / 2; stride > 32; stride = stride >> 1)
		{
			if (tid < stride)
			{
				sdesc_sq[tid] += sdesc_sq[tid + stride];
			}
			__syncthreads();
		}
		if (tid < 32)
		{
			volatile float* vmem = sdesc_sq;
			vmem[tid] += vmem[tid + 32];
			vmem[tid] += vmem[tid + 16];
			vmem[tid] += vmem[tid + 8];
			vmem[tid] += vmem[tid + 4];
			vmem[tid] += vmem[tid + 2];
			vmem[tid] += vmem[tid + 1];
		}
		__syncthreads();

		// Normalization
		float fac = 1.f / __fsqrt_rn(sdesc_sq[0]);
		float* desc = descriptors + pi * d_interp_struct.nfeatures;
		for (unsigned int i = tid; i < d_interp_struct.nfeatures; i += bsz)
		{
			desc[i] = sdesc[i] * fac;
		}
	}


	__global__ void describeApproxWithOrient(int* iimage, float* descriptors, SurfPoint* points)
	{
#define PASZ (NBIN + hwn + hwn)
		extern __shared__ float sdesc[];
		__shared__ DesParam desp;
		__shared__ int hist[NBIN];
		__shared__ float avg_angles[NBIN];
		__shared__ float part_sums[NBIN];
		__shared__ float part_angle_sums[PASZ];
		__shared__ float win_sums[NBIN];
		__shared__ float win_angle_sums[NBIN];

		// Stage1 -- Approximate major orientation
		
		// Get keypoint
		unsigned int pi = blockIdx.x;
		if (pi >= d_point_counter[d_interp_struct.noctaves - 1])
			return;
		SurfPoint* p = &points[pi];

		// Prepare shared parameters
		unsigned int bsz = blockDim.x * blockDim.y;
		unsigned int tid = threadIdx.y * blockDim.x + threadIdx.x;
		if (tid == 0)
		{
			float scale, x, y;
			if (d_interp_struct.doubled)
			{
				x = p->x + p->x;
				y = p->y + p->y;
				scale = p->scale + p->scale;
			}
			else
			{
				x = p->x;
				y = p->y;
				scale = p->scale;
			}
			desp.pixsi = __float2int_rz(2 * scale + 1.6f);
			desp.pixsi2 = __float2int_rz(scale + 0.8f);
			desp.ix = __float2int_rn(x);
			desp.iy = __float2int_rn(y);
			desp.fracx = x - desp.ix;
			desp.fracy = y - desp.iy;
			desp.wofs = d_interp_struct.desc_wsz * 0.5f - 0.5f;
		}
		__syncthreads();

		// Initialization
		if (tid < NBIN)
		{
			hist[tid] = 0;
			avg_angles[tid] = 0.f;
			part_sums[tid] = 0.f;
			win_sums[tid] = 0.f;
			win_angle_sums[tid] = 0.f;
		}
		if (tid < PASZ)
		{
			part_angle_sums[tid] = 0.f;
		}
		for (unsigned int i = tid; i < d_interp_struct.nfeatures; i += bsz)
		{
			sdesc[i] = 0.f;
		}
		__syncthreads();

		// Compute angles in each part
		for (int y1 = threadIdx.y - oradius; y1 <= oradius; y1 += blockDim.y)
		{
			for (int x1 = threadIdx.x - oradius; x1 <= oradius; x1 += blockDim.x)
			{
				int xx = desp.ix + x1 * desp.pixsi2;
				int yy = desp.iy + y1 * desp.pixsi2;
				// Do not use last row or column, which are not valid
				if (yy + desp.pixsi + 2 < whps[1].y && yy - desp.pixsi > -1 &&
					xx + desp.pixsi + 2 < whps[1].x && xx - desp.pixsi > -1)
				{
					int distsq = y1 * y1 + x1 * x1;
					if (distsq < oradiussq)
					{
						float dx = getWavelet2(iimage, xx, yy, desp.pixsi, whps[1].z) * 0.003921568627f;;	//  *0.003921568627f;
						float dy = getWavelet1(iimage, xx, yy, desp.pixsi, whps[1].z) * 0.003921568627f;;
						float magnitude = __fsqrt_rn(dx * dx + dy * dy);
						if (magnitude > 0.f)
						{
							float weight = lookup1[distsq];
							float angle = dFastAtan2(dy, dx); // atan2f(dy, dx);// dFastAtan2(dy, dx);
							int hid = __float2int_rz((angle + M_PI) / sep_angle) % NBIN;
							float psum = weight * magnitude;
							atomicAdd(&hist[hid], 1);
							atomicAdd(&avg_angles[hid], angle);
							atomicAdd(&part_sums[hid], psum);
							atomicAdd(&part_angle_sums[hid + hwn], angle * psum);
							if (hid - hwn < 0)
							{
								atomicAdd(&part_angle_sums[hid + hwn + NBIN], (angle + 2 * M_PI) * psum);
							}
							else if (hid + hwn >= NBIN)
							{
								atomicAdd(&part_angle_sums[hid + hwn - NBIN], (angle - 2 * M_PI) * psum);
							}
						}
					}
				}
			}
		}
		__syncthreads();

		// Avgerage angles
		if (tid < NBIN)
		{
			avg_angles[tid] = hist[tid] > 0 ? avg_angles[tid] / hist[tid] : bins[tid];
		}
		__syncthreads();

		// Compute window sum
		int i = tid / (hwn + hwn + 1);
		if (i < NBIN)
		{
			int j = tid % (hwn + hwn + 1) - hwn;
			int k = i + j;
			if (j == -hwn)	// Left
			{
				float residual_angle = 0.f;
				if (k < 0)
				{
					k += NBIN;
					int k1 = (k + 1) % NBIN;
					residual_angle = bins[k1] + (window / 2) - avg_angles[i] - (bins[k1] < 0 ? 0 : 2 * M_PI);
				}
				else
				{
					residual_angle = bins[k + 1] + (window / 2) - avg_angles[i];
				}
				float edge_ratio = residual_angle / sep_angle;
				atomicAdd(&win_sums[i], edge_ratio * part_sums[k]);
				atomicAdd(&win_angle_sums[i], edge_ratio * part_angle_sums[i]);
			}
			else if (j == hwn)	// Right
			{
				float residual_angle = 0.f;
				if (k >= NBIN)
				{
					k -= NBIN;
					residual_angle = avg_angles[i] + (window / 2) - 2 * M_PI - bins[k];
				}
				else
				{
					residual_angle = avg_angles[i] + (window / 2) - bins[k];
				}
				float edge_ratio = residual_angle / sep_angle;
				atomicAdd(&win_sums[i], edge_ratio * part_sums[k]);
				atomicAdd(&win_angle_sums[i], edge_ratio * part_angle_sums[i + hwn + hwn]);
			}
			else	// Middle
			{
				atomicAdd(&win_angle_sums[i], part_angle_sums[k + hwn]);
				if (k < 0) k += NBIN;
				else if (k >= NBIN) k -= NBIN;
				atomicAdd(&win_sums[i], part_sums[k]);
			}
		}
		__syncthreads();

		// Find the max value of win sum
		int residual_bin = NBIN;
		int n = __float2int_rz(__log2f(residual_bin));
		int zn = __float2int_rn(__powf(2, n));
		int offset = 0, id1 = 0, id2 = 0;
		while (residual_bin > 0)
		{
			for (int stride = zn / 2; stride > 0; stride = stride >> 1)
			{
				id1 = tid + offset;
				id2 = id1 + stride;
				if (tid < stride && win_sums[id1] < win_sums[id2])
				{
					win_sums[id1] = win_sums[id2];
					win_angle_sums[id1] = win_angle_sums[id2];
				}
				__syncthreads();
			}
			if (win_sums[0] < win_sums[offset])
			{
				win_sums[0] = win_sums[offset];
				win_angle_sums[0] = win_angle_sums[offset];
			}
			residual_bin -= zn;
			offset += zn;
			n = __float2int_rz(__log2f(residual_bin));
			zn = __float2int_rn(__powf(2, n));
		}
		__syncthreads();

		// Compute the best angle 
		if (tid == 0)
		{
			float scale = (d_interp_struct.doubled ? 3.3f : 1.65f) * p->scale;
			desp.step = max(__float2int_rn(scale * 0.5f), 1);
			desp.spacing = scale * d_interp_struct.mag_factor;
			desp.iradius = __float2int_rn(1.4f * desp.spacing * (d_interp_struct.desc_wsz + 1) * 0.5f / desp.step);
			desp.scale = __float2int_rz(scale);
			p->ori = win_angle_sums[0] / win_sums[0];
			desp.sine = __sinf(p->ori);
			desp.cose = __cosf(p->ori);
			desp.fracc = -desp.sine * desp.fracy + desp.cose * desp.fracx;
			desp.fracr = desp.cose * desp.fracy + desp.sine * desp.fracx;
		}
		__syncthreads();

		// Satge2: Compute features
		
		// Add samples
		for (int i = threadIdx.y - desp.iradius; i <= desp.iradius; i += blockDim.y)
		{
			for (int j = threadIdx.x - desp.iradius; j <= desp.iradius; j += blockDim.x)
			{
				addSample(iimage, sdesc, &desp, i, j);
			}
		}
		__syncthreads();

		// sqaure
		float* sdesc_sq = sdesc + d_interp_struct.nfeatures;
		for (unsigned int i = tid; i < d_interp_struct.nfeatures; i += bsz)
		{
			sdesc_sq[i] = sdesc[i] * sdesc[i];
		}
		__syncthreads();

		// Compute square sum
		for (int stride = d_interp_struct.nfeatures / 2; stride > 32; stride = stride >> 1)
		{
			if (tid < stride)
			{
				sdesc_sq[tid] += sdesc_sq[tid + stride];
			}
			__syncthreads();
		}
		if (tid < 32)
		{
			volatile float* vmem = sdesc_sq;
			vmem[tid] += vmem[tid + 32];
			vmem[tid] += vmem[tid + 16];
			vmem[tid] += vmem[tid + 8];
			vmem[tid] += vmem[tid + 4];
			vmem[tid] += vmem[tid + 2];
			vmem[tid] += vmem[tid + 1];
		}
		__syncthreads();

		// Normalization
		float fac = 1.f / __fsqrt_rn(sdesc_sq[0]);
		float* desc = descriptors + pi * d_interp_struct.nfeatures;
		for (unsigned int i = tid; i < d_interp_struct.nfeatures; i += bsz)
		{
			desc[i] = sdesc[i] * fac;
		}
	}


	__global__ void describeApproxWithoutNormalization(int* iimage, float* descriptors, SurfPoint* points)
	{
		__shared__ DesParam desp;

		// Get keypoint
		unsigned int pi = blockIdx.z;
		if (pi >= d_point_counter[d_interp_struct.noctaves - 1])
			return;
		SurfPoint* p = &points[pi];

		// Get shared parameters for feature computation
		unsigned int tid = threadIdx.y * blockDim.x + threadIdx.x;
		if (tid == 0)
		{
			float x, y, scale;
			if (d_interp_struct.doubled)
			{
				x = p->x + p->x;
				y = p->y + p->y;
				scale = 3.3f * p->scale;
			}
			else
			{
				x = p->x;
				y = p->y;
				scale = 1.65f * p->scale;
			}
			desp.step = max(__float2int_rn(scale * 0.5f), 1);
			desp.ix = __float2int_rn(x);
			desp.iy = __float2int_rn(y);
			desp.fracx = x - desp.ix;
			desp.fracy = y - desp.iy;
			desp.sine = __sinf(p->ori);
			desp.cose = __cosf(p->ori);
			desp.fracc = -desp.sine * desp.fracy + desp.cose * desp.fracx;
			desp.fracr = desp.cose * desp.fracy + desp.sine * desp.fracx;
			desp.spacing = scale * d_interp_struct.mag_factor;
			desp.iradius = __float2int_rn(1.4f * desp.spacing * (d_interp_struct.desc_wsz + 1) * 0.5f / desp.step);
			desp.scale = __float2int_rz(scale);
			desp.wofs = d_interp_struct.desc_wsz * 0.5f - 0.5f;
		}
		__syncthreads();

		// Check neighborhoods
		unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
		unsigned int idy = blockIdx.y * blockDim.y + threadIdx.y;
		int i = idy - desp.iradius;
		int j = idx - desp.iradius;
		if (i <= desp.iradius && j <= desp.iradius)
		{
			float* curr_descriptor = descriptors + pi * d_interp_struct.nfeatures;
			addSample(iimage, curr_descriptor, &desp, i, j);
		}
	}


	__global__ void normalize(float* descriptors)
	{
		extern __shared__ float sdesc_sq[];

		unsigned int tid = threadIdx.x;
		unsigned int bid = blockIdx.x;
		if (bid >= d_point_counter[d_interp_struct.noctaves - 1] || tid >= d_interp_struct.nfeatures)
		{
			return;
		}
		float* desc = descriptors + bid * d_interp_struct.nfeatures;

		// sqaure
		sdesc_sq[tid] = desc[tid] * desc[tid];
		__syncthreads();

		// Compute square sum
		for (int stride = d_interp_struct.nfeatures / 2; stride > 32; stride = stride >> 1)
		{
			if (tid < stride)
			{
				sdesc_sq[tid] += sdesc_sq[tid + stride];
			}
			__syncthreads();
		}
		if (tid < 32)
		{
			volatile float* vmem = sdesc_sq;
			vmem[tid] += vmem[tid + 32];
			vmem[tid] += vmem[tid + 16];
			vmem[tid] += vmem[tid + 8];
			vmem[tid] += vmem[tid + 4];
			vmem[tid] += vmem[tid + 2];
			vmem[tid] += vmem[tid + 1];
		}
		__syncthreads();

		// Compute factor
		if (tid == 0)
		{
			sdesc_sq[0] = 1.f / __fsqrt_rn(sdesc_sq[0]);
		}
		__syncthreads();

		// Normalization
		desc[tid] *= sdesc_sq[0];
	}


	__global__ void normalizeAtomic(float* descriptors)
	{
		__shared__ float dsq_sum;

		unsigned int tid = threadIdx.x;
		unsigned int bid = blockIdx.x;
		if (bid >= d_point_counter[d_interp_struct.noctaves - 1] || tid >= d_interp_struct.nfeatures)
		{
			return;
		}
		float* desc = descriptors + bid * d_interp_struct.nfeatures;

		if (tid == 0)
		{
			dsq_sum = 0.f;
		}
		__syncthreads();

		// Sqaure and add
		atomicAdd(&dsq_sum, desc[tid] * desc[tid]);
		__syncthreads();

		// Compute factor
		if (tid == 0)
		{
			dsq_sum = 1.f / __fsqrt_rn(dsq_sum);
		}
		__syncthreads();

		// Normalization
		desc[tid] *= dsq_sum;
	}


#define M7W   32
#define M7H   32
#define M7R    4
#define NRX    2
//#define NDIM 128
	__global__ void findMaxCorr(SurfPoint* surf1, SurfPoint* surf2, float* feat1, float* feat2, int num_pts1, int num_pts2)
	{
		//__shared__ float4 buffer1[M7W * NDIM / 4];
		//__shared__ float4 buffer2[M7H * NDIM / 4];
		extern __shared__ float4 buffers[];

		int NDIM4 = d_interp_struct.nfeatures / 4;
		float4* buffer1 = buffers;
		float4* buffer2 = buffer1 + M7W * NDIM4;		

		int tx = threadIdx.x;
		int ty = threadIdx.y;
		int bp1 = M7W * blockIdx.x;
		for (int j = ty; j < M7W; j += M7H / M7R)
		{
			int p1 = min(bp1 + j, num_pts1 - 1);
			for (int d = tx; d < NDIM4; d += M7W)
			{
				buffer1[j * NDIM4 + (d + j) % NDIM4] = ((float4*)(feat1 + p1 * d_interp_struct.nfeatures))[d];
			}
		}

		float max_score[NRX];
		float sec_score[NRX];
		int index[NRX];
		for (int i = 0; i < NRX; i++)
		{
			max_score[i] = 0.0f;
			sec_score[i] = 0.0f;
			index[i] = -1;
		}
		int idx = ty * M7W + tx;
		int ix = idx % (M7W / NRX);
		int iy = idx / (M7W / NRX);
		for (int bp2 = 0; bp2 < num_pts2 - M7H + 1; bp2 += M7H)
		{
			for (int j = ty; j < M7H; j += M7H / M7R)
			{
				int p2 = min(bp2 + j, num_pts2 - 1);
				for (int d = tx; d < NDIM4; d += M7W)
				{
					buffer2[j * NDIM4 + d] = ((float4*)(feat2 + p2 * d_interp_struct.nfeatures))[d];	// &sift2[p2].features
				}
			}
			__syncthreads();

			if (idx < M7W * M7H / M7R / NRX)
			{
				float score[M7R][NRX];
				for (int dy = 0; dy < M7R; dy++)
				{
					for (int i = 0; i < NRX; i++)
					{
						score[dy][i] = 0.0f;
					}
				}
				for (int d = 0; d < NDIM4; d++)
				{
					float4 v1[NRX];
					for (int i = 0; i < NRX; i++)
					{
						v1[i] = buffer1[((M7W / NRX) * i + ix) * NDIM4 + (d + (M7W / NRX) * i + ix) % NDIM4];
					}
					for (int dy = 0; dy < M7R; dy++)
					{
						float4 v2 = buffer2[(M7R * iy + dy) * NDIM4 + d];
						for (int i = 0; i < NRX; i++)
						{
							score[dy][i] += v1[i].x * v2.x;
							score[dy][i] += v1[i].y * v2.y;
							score[dy][i] += v1[i].z * v2.z;
							score[dy][i] += v1[i].w * v2.w;
						}
					}
				}
				for (int dy = 0; dy < M7R; dy++)
				{
					for (int i = 0; i < NRX; i++)
					{
						if (score[dy][i] > max_score[i])
						{
							sec_score[i] = max_score[i];
							max_score[i] = score[dy][i];
							index[i] = min(bp2 + M7R * iy + dy, num_pts2 - 1);
						}
						else if (score[dy][i] > sec_score[i])
						{
							sec_score[i] = score[dy][i];
						}
					}
				}
			}
			__syncthreads();
		}

		float* scores1 = (float*)buffer1;
		float* scores2 = &scores1[M7W * M7H / M7R];
		int* indices = (int*)&scores2[M7W * M7H / M7R];
		if (idx < M7W * M7H / M7R / NRX)
		{
			for (int i = 0; i < NRX; i++)
			{
				scores1[iy * M7W + (M7W / NRX) * i + ix] = max_score[i];
				scores2[iy * M7W + (M7W / NRX) * i + ix] = sec_score[i];
				indices[iy * M7W + (M7W / NRX) * i + ix] = index[i];
			}
		}
		__syncthreads();

		if (ty == 0)
		{
			float max_score = scores1[tx];
			float sec_score = scores2[tx];
			int index = indices[tx];
			for (int y = 0; y < M7H / M7R; y++)
			{
				if (index != indices[y * M7W + tx])
				{
					if (scores1[y * M7W + tx] > max_score)
					{
						sec_score = max(max_score, sec_score);
						max_score = scores1[y * M7W + tx];
						index = indices[y * M7W + tx];
					}
					else if (scores1[y * M7W + tx] > sec_score)
					{
						sec_score = scores1[y * M7W + tx];
					}
				}
			}
			surf1[bp1 + tx].score = max_score;
			surf1[bp1 + tx].match = index;
			surf1[bp1 + tx].match_x = surf2[index].x;
			surf1[bp1 + tx].match_y = surf2[index].y;
			surf1[bp1 + tx].ambiguity = sec_score / (max_score + 1e-6f);
		}
	}



#ifdef  _DEBUG
#define SHOW 1
#endif //  _DEBUG

#if SHOW
#include <opencv2/core.hpp>
#endif //  SHOW

	void cuIntegral(unsigned char* src, int* dst, int3 whp0, int3 whp1)
	{
		dim3 block0(X2);
		dim3 grid0((whp0.y + X2 - 1) / X2);
		integralRow << <grid0, block0 >> > (src, dst, whp0.x, whp0.y, whp0.z, whp1.z);

#if SHOW
		cv::Mat show(whp1.y, whp1.x, CV_32SC1);
		const size_t spitch = sizeof(int) * whp1.z;
		const size_t dpitch = sizeof(int) * whp1.x;
		CHECK(cudaMemcpy2D(show.data, dpitch, dst, spitch, dpitch, whp1.y, cudaMemcpyDeviceToHost));
#endif // SHOW

		dim3 block1(X2);
		dim3 grid1((whp0.x + X2 - 1) / X2);
		integralCol << <grid1, block1 >> > (dst, whp1.x, whp1.y, whp1.z);

#if SHOW
		CHECK(cudaMemcpy2D(show.data, dpitch, dst, spitch, dpitch, whp1.y, cudaMemcpyDeviceToHost));
#endif // SHOW
		CheckMsg("cuIntegralPad0Int execution failed\n");
	}


	void cuIntegralDoubleU4(unsigned char* src, int* dst, int3 whp0, int3 whp1)
	{
		const int w1 = whp0.x, h1 = whp0.y, p1 = whp0.z;
		const int w2 = whp1.x, h2 = whp1.y, p2 = whp1.z;

#if SHOW
		cv::Mat show1(h1, w1, CV_8UC1);
		const size_t spitch1 = sizeof(unsigned char) * p1;
		const size_t dpitch1 = sizeof(unsigned char) * w1;
		CHECK(cudaMemcpy2D(show1.data, dpitch1, src, spitch1, dpitch1, h1, cudaMemcpyDeviceToHost));
#endif // SHOW

		dim3 block0(X2, Y2);
		dim3 grid0((w1 / 2 + X2 - 1) / X2, (h1 + Y2 - 1) / Y2);
		integralDoubleRow0U2 << <grid0, block0 >> > (src, dst, w1, h1, p1, p2);

#if SHOW
		cv::Mat show(h2, w2, CV_32SC1);
		const size_t spitch = sizeof(int) * p2;
		const size_t dpitch = sizeof(int) * w2;
		CHECK(cudaMemcpy2D(show.data, dpitch, dst, spitch, dpitch, h2, cudaMemcpyDeviceToHost));
		CHECK(cudaMemcpy2D(show1.data, dpitch1, src, spitch1, dpitch1, h1, cudaMemcpyDeviceToHost));
#endif // SHOW

		dim3 block1(X1);
		dim3 grid1((h2 + X1 - 2) / X1);
		integralRow1U4 << <grid1, block1 >> > (dst, w2, h2, p2);

#if SHOW
		CHECK(cudaMemcpy2D(show.data, dpitch, dst, spitch, dpitch, h2, cudaMemcpyDeviceToHost));
#endif // SHOW

		dim3 block2(X2, Y2);
		dim3 grid2(((w2 - 1) / 4 + X2 - 1) / X2, (h2 + Y2 - 2) / Y2);
		integralRow2U4 << <grid2, block2 >> > (dst, w2, h2, p2);

#if SHOW
		CHECK(cudaMemcpy2D(show.data, dpitch, dst, spitch, dpitch, h2, cudaMemcpyDeviceToHost));
#endif // SHOW

		dim3 block3(X2, Y2);
		dim3 grid3(((w2 - 1) + X2 - 1) / X2, ((h2 - 1) / 4 + Y2 - 1) / Y2);
		integralCol0U4 << <grid3, block3 >> > (dst, w2, h2, p2);

#if SHOW
		CHECK(cudaMemcpy2D(show.data, dpitch, dst, spitch, dpitch, h2, cudaMemcpyDeviceToHost));
#endif // SHOW

		dim3 block4(X1);
		dim3 grid4((w2 + X1 - 2) / X1);
		integralCol1U4 << <grid4, block4 >> > (dst, w2, h2, p2);

#if SHOW
		CHECK(cudaMemcpy2D(show.data, dpitch, dst, spitch, dpitch, h2, cudaMemcpyDeviceToHost));
#endif // SHOW

		dim3 block5(X2, Y2);
		dim3 grid5((w2 + X2 - 2) / X2, ((h2 - 1) / 4 + Y2 - 1) / Y2);
		integralCol2U4 << <grid5, block5 >> > (dst, w2, h2, p2);

#if SHOW
		CHECK(cudaMemcpy2D(show.data, dpitch, dst, spitch, dpitch, h2, cudaMemcpyDeviceToHost));
#endif // SHOW

		CheckMsg("cuIntegralDoubleUnroll4Pad0 execution failed\n");
	}


	void cuHalfImage(float* src, float* dst, int3 whp0, int3 whp1)
	{
		dim3 block0(X2, Y2);
		dim3 grid0((whp1.x + X2 - 1) / X2, (whp1.y + Y2 - 1) / Y2);
		halfImage << <grid0, block0 >> > (src, dst, whp0.z, whp1.x, whp1.y, whp1.z);

#if SHOW
		cv::Mat show0(whp0.y, whp0.x, CV_32FC1);
		const size_t spitch0 = sizeof(float) * whp0.z;
		const size_t dpitch0 = sizeof(float) * whp0.x;
		CHECK(cudaMemcpy2D(show0.data, dpitch0, src, spitch0, dpitch0, whp0.y, cudaMemcpyDeviceToHost));

		cv::Mat show1(whp1.y, whp1.x, CV_32FC1);
		const size_t spitch1 = sizeof(float) * whp1.z;
		const size_t dpitch1 = sizeof(float) * whp1.x;
		CHECK(cudaMemcpy2D(show1.data, dpitch1, dst, spitch1, dpitch1, whp1.y, cudaMemcpyDeviceToHost));
#endif // SHOW

		CheckMsg("cuHalfImage execution failed\n");
	}


	void cuCalcHessian(int* src, float* dst, int3 iwhp, int3 swhp, int border1, int border2, int mask_size, int delta)
	{
		int3 vas;
		vas.x = mask_size / 2;
		vas.y = 2 * vas.x;
		vas.z = vas.x + vas.y;
		float norm = 9.f / (mask_size * mask_size);
		norm *= norm;

		const int detrows = swhp.y - border1;
		const int detcols = swhp.x - border1;
		dim3 block(X2, Y2);
		dim3 grid((detcols - border1 + X2 - 1) / X2, (detrows - border1 + Y2 - 1) / Y2);
		calcHessian<< <grid, block >> > (src, dst, vas, norm, detcols, detrows, iwhp.z, swhp.z, border1, border2, mask_size, delta);
		//printf("border1=%d, border2=%d, delta=%d, mask_size=%d, x2=%d, x3=%d, x4=%d, norm=%f\n",
		//	border1, border2, delta, mask_size, vas.x, vas.y, vas.z, norm);

#if SHOW
		cv::Mat show0(iwhp.y, iwhp.x, CV_32SC1);
		const size_t spitch0 = sizeof(int) * iwhp.z;
		const size_t dpitch0 = sizeof(int) * iwhp.x;
		CHECK(cudaMemcpy2D(show0.data, dpitch0, src, spitch0, dpitch0, iwhp.y, cudaMemcpyDeviceToHost));

		cv::Mat show(swhp.y, swhp.x, CV_32FC1);
		const size_t spitch = sizeof(float) * swhp.z;
		const size_t dpitch = sizeof(float) * swhp.x;
		CHECK(cudaMemcpy2D(show.data, dpitch, dst, spitch, dpitch, swhp.y, cudaMemcpyDeviceToHost));
#endif
		CheckMsg("cuCalcHessian execution failed\n");
	}


	void cuCalcHessianMulti(int* src, float* dst, int3 iwhp, int3 swhp, int init_scale, int max_scale, int& init_mask_size, 
		int init_border, int* borders, int octave, int sampling)
	{
		// Calculate parameters
		const int nscale = max_scale - init_scale;
		int params[MAX_SCALE * 7] = { 0 };
		int* mask_sizes = params;
		int* borders1 = mask_sizes + MAX_SCALE;
		int* borders2 = borders1 + MAX_SCALE;
		int* deltas = borders2 + MAX_SCALE;
		int* x2s = deltas + MAX_SCALE;
		int* x3s = x2s + MAX_SCALE;
		int* x4s = x3s + MAX_SCALE;
		float norms[MAX_SCALE] = { 0 };

		int border1 = init_border;
		int maxw = 0, maxh = 0;
		for (int i = 0, s = init_scale; s < max_scale; i++, s++)
		{
			borders[s] = border1;
			deltas[i] = sampling * octave;
			mask_sizes[i] = init_mask_size + 2 * octave * (i + 1);
			if (s > 2)
				border1 = 3 * mask_sizes[i] / 2 / deltas[i] + 1;
			borders1[i] = border1;
			borders2[i] = deltas[i] * border1;
			norms[i] = 9.f / (mask_sizes[i] * mask_sizes[i]);
			norms[i] *= norms[i];
			x2s[i] = mask_sizes[i] / 2;
			x3s[i] = x2s[i] + x2s[i];
			x4s[i] = x2s[i] + x3s[i];
			//printf("border1=%d, border2=%d, delta=%d, mask_size=%d, x2=%d, x3=%d, x4=%d, norm=%f\n",
			//	borders1[i], borders2[i], deltas[i], mask_sizes[i], x2s[i], x3s[i], x4s[i], norms[i]);
			maxw = std::max<int>(maxw, swhp.x - border1 * 2);
			maxh = std::max<int>(maxh, swhp.y - border1 * 2);
		}
		init_mask_size = mask_sizes[nscale - 1];

		// Copy parameters to device
		setHessianParams(params);
		setHessianNorms(norms);
		int3 temp_whps[2] = { swhp, iwhp };
		setWhps(temp_whps, 2);

		// Launch kernel function
		dim3 block(X2, Y2);
		dim3 grid((maxw + X2 - 1) / X2, (maxh + Y2 - 1) / Y2, nscale);	// int(exp2(ceil(log2(nscale))))
		//calcHessianMulti<<<grid, block>>>(src, dst, iwhp, swhp, init_scale, max_scale, d_params, d_norms);
		calcHessianMultiConst << <grid, block >> > (src, dst, init_scale, nscale);
		//dim3 grid((maxw/4 + X2 - 1) / X2, (maxh + Y2 - 1) / Y2, nscale);	// int(exp2(ceil(log2(nscale))))
		//calcHessianMultiU4 << <grid, block >> > (src, dst, iwhp, swhp, init_scale, max_scale, d_params, d_norms);
		//CHECK(cudaDeviceSynchronize());

#if SHOW
		cv::Mat show(swhp.y, swhp.x, CV_32FC1);
		size_t spitch = 0, dpitch = 0;
		for (int i = 0; i < max_scale; i++)
		{
			spitch = sizeof(float) * swhp.z;
			dpitch = sizeof(float) * swhp.x;
			CHECK(cudaMemcpy2D(show.data, dpitch, dst + i * swhp.y * swhp.z, spitch, dpitch, swhp.y, cudaMemcpyDeviceToHost));
		}
#endif // SHOW

		CheckMsg("cuCalcHessianMulti execution failed\n");
	}


	void cuFindMaximum(float* src, SurfData& result, int* borders, int3 whp, int o, int max_scale, float thresh)
	{
		int n = 0, b = 0, maxw = 0, maxh = 0;
		int mborders[(MAX_SCALE - 2) / 2] = { 0 };
		for (int k = 1; k < max_scale - 1; k+=2)
		{
			mborders[n] = borders[k + 1] + 1;
			b = mborders[n] + mborders[n];
			maxw = std::max<int>(maxw, whp.x - b);
			maxh = std::max<int>(maxh, whp.y - b);
			n++;
		}
		setMaximumBorders(mborders);
		
		dim3 block(X2, Y2);
		dim3 grid((maxw / 2 + X2 - 1) / X2, (maxh / 2 + Y2 - 1) / Y2, n);
		findMaximum << <grid, block >> > (src, result.d_data, o, max_scale, thresh);

#if SHOW
		//unsigned int* d_point_counter_addr;
		//getPointCounter((void**)&d_point_counter_addr);
		//CHECK(cudaMemcpy(&result.num_pts, &d_point_counter_addr[o], sizeof(unsigned int), cudaMemcpyDeviceToHost));
		//CHECK(cudaMemcpy(result.h_data, result.d_data, sizeof(SurfPoint) * result.num_pts, cudaMemcpyDeviceToHost));
		//static int m = 0;
		//for (int i = m; i < result.num_pts; i++)
		//{
		//	printf("s=%d,r=%d,c=%d\n", cvRound(result.h_data[i].scale), cvRound(result.h_data[i].y), cvRound(result.h_data[i].x));
		//}
		//m = result.num_pts;

		//float* hsrc = (float*)malloc(max_scale * whp.y * whp.z * sizeof(float));
		//CHECK(cudaMemcpy(hsrc, src, max_scale * whp.y * whp.z * sizeof(float), cudaMemcpyDeviceToHost));
		//float best;
		//int r, c, s, ss;
		//int dr, dc, ds;
		//int cas;
		//int pcounter = 0;
		//for (int k = 1, fuck = 0; k < max_scale - 1; k+=2, fuck++)
		//{
		//	int mb = mborders[fuck];
		//	for (int i = mb; i < whp.y - mb; i += 2)
		//	{
		//		for (int j = mb; j < whp.x - mb; j += 2)
		//		{
		//			int4 idx;
		//			idx.w = i * whp.z + j;
		//			idx.x = idx.w + 1;
		//			idx.y = idx.w + whp.z;
		//			idx.z = idx.y + 1;
		//			const int osize = whp.y * whp.z;
		//			float* curr_scale = hsrc + k * osize;
		//			int cas = 0;
		//			float best = curr_scale[idx.w];
		//			if (curr_scale[idx.x] > best)
		//			{
		//				best = curr_scale[idx.x];
		//				cas = 1;
		//			}
		//			if (curr_scale[idx.y] > best)
		//			{
		//				best = curr_scale[idx.y];
		//				cas = 2;
		//			}
		//			if (curr_scale[idx.z] > best)
		//			{
		//				best = curr_scale[idx.z];
		//				cas = 3;
		//			}
		//			curr_scale += osize;
		//			if (curr_scale[idx.w] > best)
		//			{
		//				best = curr_scale[idx.w];
		//				cas = 4;
		//			}
		//			if (curr_scale[idx.x] > best)
		//			{
		//				best = curr_scale[idx.x];
		//				cas = 5;
		//			}
		//			if (curr_scale[idx.y] > best)
		//			{
		//				best = curr_scale[idx.y];
		//				cas = 6;
		//			}
		//			if (curr_scale[idx.z] > best)
		//			{
		//				best = curr_scale[idx.z];
		//				cas = 7;
		//			}
		//			if (best < thresh || (k + 1 == max_scale - 1 && cas > 3))
		//				continue;
		//			int s = k, r = i, c = j;
		//			int ds = -1, dr = -1, dc = -1;
		//			if (cas != 0)
		//			{
		//				if (cas == 1) { c = j + 1; dc = 1; }
		//				else if (cas == 2) { r = i + 1; dr = 1; }
		//				else if (cas == 3) { c = j + 1; r = i + 1; dc = 1; dr = 1; }
		//				else
		//				{
		//					s++;
		//					ds = 1;
		//					if (cas == 5) { c = j + 1; dc = 1; }
		//					else if (cas == 6) { r = i + 1; dr = 1; }
		//					else if (cas == 7) { c = j + 1; r = i + 1; dc = 1; dr = 1; }
		//				}
		//			}
		//			int ss = s + ds;
		//			curr_scale = hsrc + ss * osize;
		//			idx.y = (r - dr) * whp.z + c;
		//			idx.x = idx.y - 1;
		//			idx.z = idx.y + 1;
		//			if (best < curr_scale[idx.x]) continue;
		//			if (best < curr_scale[idx.y]) continue;
		//			if (best < curr_scale[idx.z]) continue;
		//			idx.y += dr * whp.z;
		//			idx.x = idx.y - 1;
		//			idx.z = idx.y + 1;
		//			if (best < curr_scale[idx.x]) continue;
		//			if (best < curr_scale[idx.y]) continue;
		//			if (best < curr_scale[idx.z]) continue;
		//			idx.y += dr * whp.z;
		//			idx.x = idx.y - 1;
		//			idx.z = idx.y + 1;
		//			if (best < curr_scale[idx.x]) continue;
		//			if (best < curr_scale[idx.y]) continue;
		//			if (best < curr_scale[idx.z]) continue;
		//			curr_scale = hsrc + s * osize;
		//			if (best < curr_scale[idx.x]) continue;
		//			if (best < curr_scale[idx.y]) continue;
		//			if (best < curr_scale[idx.z]) continue;
		//			idx.w = r * whp.z + c + dc;
		//			if (best < curr_scale[idx.w]) continue;
		//			idx.w -= dr * whp.z;
		//			if (best < curr_scale[idx.w]) continue;
		//			ss = s - ds;
		//			curr_scale = hsrc + ss * osize;
		//			if (best < curr_scale[idx.x]) continue;
		//			if (best < curr_scale[idx.y]) continue;
		//			if (best < curr_scale[idx.z]) continue;
		//			if (best < curr_scale[idx.w]) continue;
		//			idx.w += dr * whp.z;
		//			if (best < curr_scale[idx.w]) continue;
		//			printf("i=%d,j=%d,k=%d,s=%d,r=%d,c=%d\n", i, j, k, s, r, c);
		//			if (pcounter < d_max_num_points)
		//			{
		//				SurfPoint& p = result.h_data[pcounter];
		//				p.x = j;
		//				p.y = i;
		//				p.scale = k;
		//				p.o = o;
		//			}
		//		}
		//	}
		//}
#endif // SHOW
		
		CheckMsg("cuFindMaximum execution failed\n");
	}


	void cuFindMaximumWithInterp(int* iimage, float* src, SurfData& result, int* borders, int3 whp, int o, int octave, int moves_remain, SurfParam& its)
	{
#define DX 16
#define DY 16
		int n = 0, b = 0, maxw = 0, maxh = 0;
		int mborders[(MAX_SCALE - 2) / 2] = { 0 };
		for (int k = 1; k < its.max_scale - 1; k += 2)
		{
			mborders[n] = borders[k + 1] + 1;
			b = mborders[n] + mborders[n];
			maxw = std::max<int>(maxw, whp.x - b);
			maxh = std::max<int>(maxh, whp.y - b);
			n++;
		}
		setMaximumBorders(mborders);
		setBorders(borders);

		dim3 block(DX, DY);
		dim3 grid((maxw / 2 + DX - 1) / DX, (maxh / 2 + DY - 1) / DY, n);
		findMaximumWithInterp << <grid, block >> > (iimage, src, result.d_data, o, octave, moves_remain);
		CheckMsg("findMaximumWithInterp execution failed\n");
	}


	void hSolveLinearSystem(float* solution, float sq[3][3], int size)
	{
		int row, col, c, pivot = 0, i;
		float maxc, coef, temp, mult, val;

		// Triangularize the matrix
		for (col = 0; col < size - 1; col++)
		{
			// Pivot row with largest coefficient to top
			maxc = -1.f;
			for (row = col; row < size; row++)
			{
				coef = sq[row][col];
				coef = (coef < 0.f ? -coef : coef);
				if (coef > maxc)
				{
					maxc = coef;
					pivot = row;
				}
			}
			if (pivot != col)
			{
				// Exchange "pivot" with "col" row (this is no less efficient
				// than having to perform all array accesses indirectly)
				for (i = 0; i < size; i++)
				{
					temp = sq[pivot][i];
					sq[pivot][i] = sq[col][i];
					sq[col][i] = temp;
				}
				temp = solution[pivot];
				solution[pivot] = solution[col];
				solution[col] = temp;
			}
			// Do reduction for this column
			for (row = col + 1; row < size; row++)
			{
				mult = sq[row][col] / sq[col][col];
				for (c = col; c < size; c++) //Could start with c=col+1
					sq[row][c] -= mult * sq[col][c];
				solution[row] -= mult * solution[col];
			}
		}

		// Do back substitution.  Pivoting does not affect solution order
		for (row = size - 1; row >= 0; row--)
		{
			val = solution[row];
			for (col = size - 1; col > row; col--)
				val -= solution[col] * sq[row][col];
			solution[row] = val / sq[row][row];
		}
	}


	float hFitQuadrat(float* src, float* _offsets, int3* swhps, int* params, const int s, const  int r, const int c, const int o)
	{
		int* osizes = params + MAX_OCTAVE;
		int* offsets = osizes + MAX_OCTAVE;
		float* curr_octave = src + offsets[o];
		float* curr_scale = curr_octave + s * osizes[o];
		float* prev_scale = curr_scale - osizes[o];
		float* next_scale = curr_scale + osizes[o];

		// variables to fit quadratic
		float _g[3], _H[3][3];

		// Fill in the values of the gradient from pixel differences
		int idx = r * swhps[o].z + c;
		int idx_nr = idx + swhps[o].z;
		int idx_pr = idx - swhps[o].z;
		int idx_nc = idx + 1;
		int idx_pc = idx - 1;
		_g[0] = (next_scale[idx] - prev_scale[idx]) * 0.5f;
		_g[1] = (curr_scale[idx_nr] - curr_scale[idx_pr]) * 0.5f;
		_g[2] = (curr_scale[idx_nc] - curr_scale[idx_pc]) * 0.5f;

		// Fill in the values of the Hessian from pixel differences
		const float temp = curr_scale[idx] + curr_scale[idx];
		_H[0][0] = prev_scale[idx] + next_scale[idx] - temp;
		_H[1][1] = curr_scale[idx_nr] + curr_scale[idx_pr] - temp;
		_H[2][2] = curr_scale[idx_nc] + curr_scale[idx_pc] - temp;
		_H[0][1] = ((next_scale[idx_nr] - next_scale[idx_pr]) -
			(prev_scale[idx_nr] - prev_scale[idx_pr])) * 0.25f;
		_H[0][2] = ((next_scale[idx_nc] - next_scale[idx_pc]) -
			(prev_scale[idx_nc] - prev_scale[idx_pc])) * 0.25f;
		_H[1][2] = ((curr_scale[idx_nr + 1] - curr_scale[idx_nr - 1]) -
			(curr_scale[idx_pr + 1] - curr_scale[idx_pr - 1])) * 0.25f;
		_H[1][0] = _H[0][1];
		_H[2][0] = _H[0][2];
		_H[2][1] = _H[1][2];
		_offsets[0] = -_g[0];
		_offsets[1] = -_g[1];
		_offsets[2] = -_g[2];
		hSolveLinearSystem(_offsets, _H, 3);
		//printf("o=%d,s=%d,r=%d,c=%d,strength=%f,_g(%f,%f,%f),_offsets(%f,%f,%f),_H(%f,%f,%f,%f,%f,%f,%f,%f,%f)\n",
		//	o, s, r, c, 0, _g[0], _g[1], _g[2], _offsets[0], _offsets[1], _offsets[2], _H[0][0], _H[0][1],
		//	_H[0][2], _H[1][0], _H[1][1], _H[1][2], _H[2][0], _H[2][1], _H[2][2]);

		// Also return value of the determinant at peak location using initial
		// value plus 0.5 times linear interpolation with gradient to peak
		// position (this is correct for a quadratic approximation).
		float strength = curr_scale[idx] + 0.5f * (_offsets[0] * _g[0] + _offsets[1] * _g[1] + _offsets[2] * _g[2]);
		return strength;
	}


//	void cuInterpFeature(int* iimage, float* src, SurfData& result, int3* swhps, int* params, int moves_remain, SurfParam& its, int tot_osize)
//	{
//#if SHOW
//		//float* hsrc = (float*)malloc(sizeof(float) * tot_osize);
//		//CHECK(cudaMemcpy(hsrc, src, sizeof(float) * tot_osize, cudaMemcpyDeviceToHost));
//		//CHECK(cudaMemcpy(result.h_data, result.d_data, sizeof(SurfPoint) * result.num_pts, cudaMemcpyDeviceToHost));
//		//for (int i = 0; i < result.num_pts; i++)
//		//{
//		//	SurfPoint* p = &result.h_data[i];
//		//	int o = p->o;
//		//	int s = cvRound(p->scale);
//		//	int row = cvRound(p->y);
//		//	int col = cvRound(p->x);
//		//	int newr = row;
//		//	int newc = col;
//		//	int* borders = params + 3 * MAX_OCTAVE + o * MAX_SCALE;
//		//	// Interpolate the detected maximum in order to get a more accurate location
//		//	float _offsets[3] = { 0 };
//		//	float strength = 0;
//		//	for (int mv = 0; mv < moves_remain; mv++)
//		//	{
//		//		row = newr; col = newc;
//		//		strength = hFitQuadrat(hsrc, _offsets, swhps, params, s, row, col, o);
//		//		printf("moves_remain=%d,o=%d,s=%d,r=%d,c=%d,newr=%d,newc=%d,strength=%f\n", mv, o, s, row, col, newr, newc, strength);
//		//		if (_offsets[1] > 0.6f && row < swhps[o].y - borders[s]) newr++;
//		//		if (_offsets[1] < -0.6f && row > borders[s]) newr--;
//		//		if (_offsets[2] > 0.6f && col < swhps[o].x - borders[s]) newc++;
//		//		if (_offsets[2] < -0.6f && col > borders[s]) newc--;
//		//		if ((newr == row && newc == col)) break;
//		//	}
//		//}	
//#endif // SHOW
//
//		// Copy parameters to const
//		setWhps(swhps);
//		setSurfParam(&its);
//		setInterpParams(params);
//
//		// Indexes
//		bool* flags = NULL;
//		CHECK(cudaMalloc((void**)&flags, result.num_pts * sizeof(bool)));
//		CHECK(cudaMemset(flags, 0, result.num_pts * sizeof(bool)));
//		GpuTimer timer(0);
//
//		// Launch device for interpolate
//		dim3 block(X1);
//		dim3 grid((result.num_pts + X1 - 1) / X1);
//		interpFeature << <grid, block >> > (iimage, src, result.d_data, flags, moves_remain, result.num_pts);
//		float t0 = timer.read();
//
//		// Re-assign points
//		reassign << <1, 1 >> > (result.d_data, flags, result.num_pts);
//
//		float t1 = timer.read();
//		printf("Time of feature intepolation: %f\n", t0);
//		printf("Time of Reassign: %f\n", t1 - t0);
//
//		CHECK(cudaFree(flags));
//		CheckMsg("cuInterpFeature execution failed\n");
//	}


	void cuDescribe(int* iimage, float** desc_addr, SurfData& result, SurfParam& its)
	{
//#if SHOW
//		for (int i = 0; i < result.num_pts; i++)
//		{
//			const SurfPoint& p = result.h_data[i];
//			printf("s=%f, x=%f, y=%f\n", p.scale, p.x, p.y);
//		}		
//#endif // SHOW

		// Allocate memory for descriptors
		const size_t oned_bytes = its.nfeatures * sizeof(float);
		const size_t nbytes = result.num_pts * oned_bytes;
		CHECK(cudaMalloc((void**)desc_addr, nbytes));
		float* descriptors = *desc_addr;
		CHECK(cudaMemset(descriptors, 0, nbytes));
		int max_iradius;
		getIradius(&max_iradius);
		const int msz = max_iradius * 2 + 1;

		if (its.upright)
		{
			// Extract upright sample descriptor
			//dim3 block(X2, Y2);
			//dim3 grid(result.num_pts);
			//describeUR << <grid, block, oned_bytes * 2 >> > (iimage, descriptors, result.d_data);

			dim3 block0(X2, Y2);
			dim3 grid0((msz + X2 - 1) / X2, (msz + Y2 - 1) / Y2, result.num_pts);
			describeURWithoutNormalization << <grid0, block0 >> > (iimage, descriptors, result.d_data);

			dim3 block1(its.nfeatures);
			dim3 grid1(result.num_pts);
			normalize << <grid1, block1, oned_bytes >> > (descriptors);
			//normalizeAtomic << <grid1, block1 >> > (descriptors);
		}
		else
		{
			// Extract main orientation rotated descriptor
			dim3 block0(X2, Y2);
			dim3 grid0(result.num_pts);
			assignOrientationApprox << <grid0, block0 >> > (iimage, descriptors, result.d_data);
			//describeApprox << <grid, block, oned_bytes * 2 >> > (iimage, descriptors, result.d_data);
			//describeApproxWithOrient << <grid, block, oned_bytes * 2 >> > (iimage, descriptors, result.d_data);

			dim3 block1(X2, Y2);
			dim3 grid1((msz + X2 - 1) / X2, (msz + Y2 - 1) / Y2, result.num_pts);
			describeApproxWithoutNormalization << <grid1, block1 >> > (iimage, descriptors, result.d_data);

			dim3 block2(its.nfeatures);
			dim3 grid2(result.num_pts);
			normalize << <grid2, block2, oned_bytes >> > (descriptors);
		}
		CHECK(cudaDeviceSynchronize());

#if SHOW
		//float* hdesc = (float*)malloc(nbytes);
		//CHECK(cudaMemcpy(hdesc, descriptors, nbytes, cudaMemcpyDeviceToHost));
		//float* temp = hdesc;
		//for (int i = 0; i < 10; i++)
		//{
		//	const SurfPoint& p = result.h_data[i];
		//	printf("s=%f, y=%f, x=%f, features: ", p.scale, p.y, p.x);
		//	for (int j = 0; j < its.nfeatures; j++)
		//	{
		//		printf("%f ", temp[j]);
		//	}
		//	printf("\n");
		//	temp += its.nfeatures;
		//}
		//free(hdesc);
#endif // SHOW

		CheckMsg("cuDescribe execution failed\n");
	}


	void cuDescribe(cudaTextureObject_t iimage, float** desc_addr, SurfData& result, SurfParam& its)
	{
		// Allocate memory for descriptors
		const size_t oned_bytes = its.nfeatures * sizeof(float);
		const size_t nbytes = result.num_pts * oned_bytes;
		CHECK(cudaMalloc((void**)desc_addr, nbytes));
		float* descriptors = *desc_addr;
		CHECK(cudaMemset(descriptors, 0, nbytes));
		int max_iradius;
		getIradius(&max_iradius);
		const int msz = max_iradius * 2 + 1;
		// assert(X2 * Y2 >= its.nfeatures);

		dim3 block(X2, Y2);
		dim3 grid((msz + X2 - 1) / X2, (msz + Y2 - 1) / Y2, result.num_pts);
		if (its.upright)
		{
			// Extract upright sample descriptor
			describeURMulti << <grid, block, oned_bytes * 2 >> > (iimage, descriptors, result.d_data);
		}
		else
		{
			// Extract main orientation rotated descriptor
		}
		CHECK(cudaDeviceSynchronize());

#if SHOW
		//float* hdesc = (float*)malloc(nbytes);
		//CHECK(cudaMemcpy(hdesc, descriptors, nbytes, cudaMemcpyDeviceToHost));
		//float* temp = hdesc;
		//for (int i = 0; i < 10; i++)
		//{
		//	const SurfPoint& p = result.h_data[i];
		//	printf("s=%f, y=%f, x=%f, features: ", p.scale, p.y, p.x);
		//	for (int j = 0; j < its.nfeatures; j++)
		//	{
		//		printf("%f ", temp[j]);
		//	}
		//	printf("\n");
		//	temp += its.nfeatures;
		//}
		//free(hdesc);
#endif // SHOW

		CheckMsg("cuDescribe execution failed\n");
	}


    /*
	void hFindMaxCorr(SurfPoint* surf1, SurfPoint* surf2, float* feat1, float* feat2, int num_pts1, int num_pts2, int nfeatures)
	{
		// Has error!!!
		int NDIM4 = nfeatures / 4;
		std::vector<std::vector<float4>> buffer1(M7W, std::vector<float4>(NDIM4));
		std::vector<std::vector<float4>> buffer2(M7H, std::vector<float4>(NDIM4));
		cv::Mat coords1(M7W, NDIM4, CV_32SC2);
		cv::Mat coords2(M7H, NDIM4, CV_32SC2);

		for (int bid = 0; bid < iDivUp(num_pts1, M7W); bid++)
		{
			int bp1 = M7W * bid;
			float max_score[NRX];
			float sec_score[NRX];
			int index[NRX];
			for (int i = 0; i < NRX; i++)
			{
				max_score[i] = 0.0f;
				sec_score[i] = 0.0f;
				index[i] = -1;
			}

			for (int bp2 = 0; bp2 < num_pts2 - M7H + 1; bp2 += M7H)
			{
				for (int ty = 0; ty < M7H / M7R; ty++)
				{
					for (int tx = 0; tx < M7W; tx++)
					{
						for (int j = ty; j < M7W; j += M7W / M7R)
						{
							int p1 = MIN(bp1 + j, num_pts1 - 1);
							int p2 = MIN(bp2 + j, num_pts2 - 1);
							for (int d = tx; d < NDIM4; d += M7W)
							{
								float* desc1 = feat1 + p1 * nfeatures;
								float* desc2 = feat2 + p2 * nfeatures;
								int bufx = (d + j) % NDIM4;
								int bufy = j;
								buffer1[bufy][bufx] = ((float4*)desc1)[d];
								buffer2[bufy][d] = ((float4*)desc2)[d];
								coords1.at<cv::Vec2i>(bufy, bufx) = cv::Vec2i(p1, 4 * d);								
								coords2.at<cv::Vec2i>(bufy, d) = cv::Vec2i(p2, 4 * d);
							}
						}
					}
				}

				for (int ty = 0; ty < M7H / M7R; ty++)
				{
					for (int tx = 0; tx < M7W; tx++)
					{
						int idx = ty * M7W + tx;
						int ix = idx % (M7W / NRX);
						int iy = idx / (M7W / NRX);
						if (idx < M7W * M7H / M7R / NRX)
						{
							float score[M7R][NRX];
							for (int dy = 0; dy < M7R; dy++)
							{
								for (int i = 0; i < NRX; i++)
								{
									score[dy][i] = 0.0f;
								}
							}
							for (int d = 0; d < NDIM4; d++)
							{
								float4 v1[NRX];
								cv::Vec2i c1[NRX];
								for (int i = 0; i < NRX; i++)
								{
									//v1[i] = buffer1[((M7W / NRX) * i + ix) * NDIM4 + (d + (M7W / NRX) * i + ix) % NDIM4];
									int bufy = (M7W / NRX) * i + ix;
									int bufx = (d + (M7W / NRX) * i + ix) % NDIM4;
									v1[i] = buffer1[bufy][bufx];
									c1[i] = coords1.at<cv::Vec2i>(bufy, bufx);
								}
								for (int dy = 0; dy < M7R; dy++)
								{
									//float4 v2 = buffer2[(M7R * iy + dy) * NDIM4 + d];
									int bufy = M7R * iy + dy;
									int bufx = d;
									float4 v2 = buffer2[bufy][bufx];									
									cv::Vec2i c2 = coords2.at<cv::Vec2i>(bufy, bufx);
									//if (c2[0] == 0 && c2[1] == 0)
									//{
									//	printf("C1: (%d, %d), C2: (%d, %d)\n", c1[0][0], c1[0][1], c2[0], c2[1]);
									//}
									for (int i = 0; i < NRX; i++)
									{
										score[dy][i] += v1[i].x * v2.x;
										score[dy][i] += v1[i].y * v2.y;
										score[dy][i] += v1[i].z * v2.z;
										score[dy][i] += v1[i].w * v2.w;
										if (c1[0][0] == 0 && c1[0][1] == 0)
										{
											printf("C1: (%d, %d), C2: (%d, %d)\n", c1[0][0], c1[0][1], c2[0], c2[1]);
										}
									}
								}
							}
							for (int dy = 0; dy < M7R; dy++)
							{
								for (int i = 0; i < NRX; i++)
								{
									if (score[dy][i] > max_score[i])
									{
										sec_score[i] = max_score[i];
										max_score[i] = score[dy][i];
										index[i] = MIN(bp2 + M7R * iy + dy, num_pts2 - 1);
									}
									else if (score[dy][i] > sec_score[i])
									{
										sec_score[i] = score[dy][i];
									}
								}
							}
						}
					}
				}
			}

			std::vector<float> scores1(M7W * M7H / M7R);
			std::vector<float> scores2(M7W * M7H / M7R);
			std::vector<int> indices(M7W * M7H / M7R);
			for (int ty = 0; ty < M7H / M7R; ty++)
			{
				for (int tx = 0; tx < M7W; tx++)
				{
					int idx = ty * M7W + tx;
					int ix = idx % (M7W / NRX);
					int iy = idx / (M7W / NRX);
					if (idx < M7W * M7H / M7R / NRX)
					{
						for (int i = 0; i < NRX; i++)
						{
							scores1[iy * M7W + (M7W / NRX) * i + ix] = max_score[i];
							scores2[iy * M7W + (M7W / NRX) * i + ix] = sec_score[i];
							indices[iy * M7W + (M7W / NRX) * i + ix] = index[i];
						}
					}
				}
			}

			for (int tx = 0; tx < M7W; tx++)
			{
				float max_score = scores1[tx];
				float sec_score = scores2[tx];
				int index = indices[tx];
				for (int y = 0; y < M7H / M7R; y++)
				{
					if (index != indices[y * M7W + tx])
					{
						if (scores1[y * M7W + tx] > max_score)
						{
							sec_score = std::max<float>(max_score, sec_score);
							max_score = scores1[y * M7W + tx];
							index = indices[y * M7W + tx];
						}
						else if (scores1[y * M7W + tx] > sec_score)
						{
							sec_score = scores1[y * M7W + tx];
						}
					}
				}

				int pi = bp1 + tx;
				printf("%dth point1 --> %dth point2\n", pi, index);
			}
		}
	}
    */


	void cuFindMaxCorr(SurfPoint* surf1, SurfPoint* surf2, float* feat1, float* feat2, int num_pts1, int num_pts2, int nfeatures)
	{
		//const size_t nbytes1 = num_pts1 * nfeatures * sizeof(float);
		//const size_t nbytes2 = num_pts2 * nfeatures * sizeof(float);
		//float* hdesc1 = (float*)malloc(nbytes1);
		//float* hdesc2 = (float*)malloc(nbytes2);
		//CHECK(cudaMemcpy(hdesc1, feat1, nbytes1, cudaMemcpyDeviceToHost));
		//CHECK(cudaMemcpy(hdesc2, feat2, nbytes2, cudaMemcpyDeviceToHost));
		//hFindMaxCorr(surf1, surf2, hdesc1, hdesc2, num_pts1, num_pts2, nfeatures);
				
		dim3 blocks(iDivUp(num_pts1, M7W));
		dim3 threads(M7W, M7H / M7R);
		size_t sbytes = (M7W + M7H) * nfeatures * sizeof(float);
		findMaxCorr << <blocks, threads, sbytes >> > (surf1, surf2, feat1, feat2, num_pts1, num_pts2);
		CHECK(cudaDeviceSynchronize());
		CheckMsg("cuFindMaxCorr() execution failed\n");
	}

}

