#include "surf.h"
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>


void cudaSurfDemo(int argc, char** argv);
void cudaSurfDemo2(int argc, char** argv);
void drawKeypoints(surf::SurfData& a, cv::Mat& img, cv::Mat& dst);
void drawMatches(surf::SurfData& a, surf::SurfData& b, cv::Mat& img1, cv::Mat& img2, cv::Mat& dst, const bool horizontal = true);


int main(int argc, char** argv)
{
    cudaSurfDemo2(argc, argv);

    CHECK(cudaDeviceReset());
    return 0;
}


void drawKeypoints(surf::SurfData& a, cv::Mat& img, cv::Mat& dst)
{
	surf::SurfPoint* data = a.h_data;
	cv::merge(std::vector<cv::Mat>{ img, img, img }, dst);
	for (int i = 0; i < a.num_pts; i++)
	{
		surf::SurfPoint& p = data[i];
		cv::Point center(cvRound(p.x), cvRound(p.y));
		cv::Scalar color(rand() % 255, rand() % 255, rand() % 255);
		cv::circle(dst, center, MIN(5, MAX(1, cvRound(p.strength))), color);
	}
}


void drawMatches(surf::SurfData& a, surf::SurfData& b, cv::Mat& img1, cv::Mat& img2, cv::Mat& dst, const bool horizontal)
{
	int num_pts = a.num_pts;
	surf::SurfPoint* surf1 = a.h_data;
	surf::SurfPoint* surf2 = b.h_data;
	const int h1 = img1.rows;
	const int h2 = img2.rows;
	const int w1 = img1.cols;
	const int w2 = img2.cols;
	if (horizontal)
	{
		cv::Mat cat_img = cv::Mat::zeros(std::max<int>(h1, h2), w1 + w2, CV_32FC1);
		img1.copyTo(cat_img(cv::Rect(0, 0, w1, h1)));
		img2.copyTo(cat_img(cv::Rect(w1, 0, w2, h2)));
		cv::merge(std::vector<cv::Mat>{cat_img, cat_img, cat_img}, dst);
	}
	else
	{
		cv::Mat cat_img = cv::Mat::zeros(h1 + h2, std::max<int>(w1, w2), CV_32FC1);
		img1.copyTo(cat_img(cv::Rect(0, 0, w1, h1)));
		img2.copyTo(cat_img(cv::Rect(0, h1, w2, h2)));
		cv::merge(std::vector<cv::Mat>{cat_img, cat_img, cat_img}, dst);
	}
	dst.convertTo(dst, CV_8UC3);
	for (int i = 0; i < num_pts; i++)
	{
		int k = surf1[i].match;
		cv::Point p1(cvRound(surf1[i].x), cvRound(surf1[i].y));
		cv::Point p2(cvRound(surf2[k].x), cvRound(surf2[k].y));
		if (horizontal)
			p2.x += w1;
		else
			p2.y += h1;
		cv::Scalar color(rand() % 255, rand() % 255, rand() % 255);
		cv::line(dst, p1, p2, color);
	}
}


void cudaSurfDemo(int argc, char** argv)
{
    int devNum = 0, imgSet = 0;
	if (argc > 1)
		devNum = std::atoi(argv[1]);
	if (argc > 2)
		imgSet = std::atoi(argv[2]);

    initDevice(devNum);

	/* Read image using OpenCV. */
    std::string image_path = imgSet ? "data/img1.png" : "data/left.pgm";
	cv::Mat limg = cv::imread(image_path, cv::IMREAD_GRAYSCALE);

	/* Allocate 1280x960 pixel image with device side pitch of 1280 floats. */
	/* Memory on host side already allocated by OpenCV is reused.           */
	int3 whp;
	whp.x = limg.cols;
	whp.y = limg.rows;
	whp.z = iAlignUp(whp.x, 128);
	size_t size = whp.y * whp.z * sizeof(uchar);
	uchar* img = NULL;
	size_t tmp_pitch = 0;
	CHECK(cudaMallocPitch((void**)&img, &tmp_pitch, sizeof(uchar) * whp.x, whp.y));
	const size_t dpitch = sizeof(uchar) * whp.z;
	const size_t spitch = sizeof(uchar) * whp.x;
	CHECK(cudaMemcpy2D(img, dpitch, limg.data, spitch, spitch, whp.y, cudaMemcpyHostToDevice));
	//CudaImage img;
	//img.allocate(limg.cols, limg.rows, limg.cols, false, NULL, (float*)limg.data);
	/* Download image from host to device */
	//img.download();

	// Initial sampling step (default 2)
	int samplingStep = 2;
	// Number of analysed octaves (default 4)
	int octaves = 4;
	// Blob response treshold
	float thres = 4.f;
	// Set this flag "true" to double the image size
	bool doubleImageSize = false;
	// Initial lobe size, default 3 and 5 (with double image size)
	int initLobe = 3;
	// Upright SURF or rotation invaraiant
	bool upright = true;
	// If the extended flag is turned on, SURF 128 is used
	bool extended = false;
	// Spatial size of the descriptor window (default 4)
	int indexSize = 4;
	// verbose output
	bool bVerbose = true;
	int max_npts = 10000;

	/* Reserve memory space for a whole bunch of SURF features. */	
	surf::SurfData surf_data;
	surf::initSurfData(surf_data, max_npts, true, true);
	float* surf_descriptors = NULL;

	surf::Surfor detector;
	detector.init(octaves, thres, doubleImageSize, initLobe * 3, samplingStep, upright, extended, indexSize, limg.cols, limg.rows);
	const int n = 1000;

	GpuTimer timer(0);
	for (int i = 0; i < n; i++)
	{
		detector.detectAndCompute(img, surf_data, whp, &surf_descriptors, true);
		//printf("%dth, %d SURF points extracted.\n", i, surf_data.num_pts);
		//if (surf_descriptors)
		//{
		//	CHECK(cudaFree(surf_descriptors));
		//	surf_descriptors = NULL;
		//}
	}
	float t0 = timer.read();

	cv::Mat show;
	drawKeypoints(surf_data, limg, show);
	std::cout << "Time of keypoints detection: " << t0 / n << std::endl;
    std::cout << "Number of features detected: " << surf_data.num_pts << std::endl;

	/* Free space allocated from SURF features */
	surf::freeSurfData(surf_data);
	CHECK(cudaFree(img));
	if (surf_descriptors)
	{
		CHECK(cudaFree(surf_descriptors));
	}	
}


void cudaSurfDemo2(int argc, char** argv)
{
	int devNum = 0, imgSet = 0;
	if (argc > 1)
		devNum = std::atoi(argv[1]);
	if (argc > 2)
		imgSet = std::atoi(argv[2]);

	// Read images using OpenCV
	cv::Mat limg, rimg;
	if (imgSet)
	{
		limg = cv::imread("data/left.pgm", cv::IMREAD_GRAYSCALE);
		rimg = cv::imread("data/right.pgm", cv::IMREAD_GRAYSCALE);
	}
	else
	{
		limg = cv::imread("data/img1.png", cv::IMREAD_GRAYSCALE);
		rimg = cv::imread("data/img2.png", cv::IMREAD_GRAYSCALE);
	}
	unsigned int w = limg.cols;
	unsigned int h = limg.rows;
	std::cout << "Image size = (" << w << "," << h << ")" << std::endl;

	// Initial sampling step (default 2)
	int samplingStep = 2;
	// Number of analysed octaves (default 4)
	int octaves = 4;
	// Blob response treshold
	float thres = 4.f;
	// Set this flag "true" to double the image size
	bool doubleImageSize = false;
	// Initial lobe size, default 3 and 5 (with double image size)
	int initLobe = 3;
	// Upright SURF or rotation invaraiant
	bool upright = true;
	// If the extended flag is turned on, SURF 128 is used
	bool extended = false;
	// Spatial size of the descriptor window (default 4)
	int indexSize = 4;
	// Max number of points
	int max_npts = 10000;

	// Initial Cuda images and download images to device
	std::cout << "Initializing data..." << std::endl;
	initDevice(devNum);

	GpuTimer timer(0);
	int3 whp1, whp2;
	whp1.x = limg.cols; whp1.y = limg.rows; whp1.z = iAlignUp(whp1.x, 128);
	whp2.x = rimg.cols; whp2.y = rimg.rows; whp2.z = iAlignUp(whp2.x, 128);
	size_t size1 = whp1.y * whp1.z * sizeof(uchar);
	size_t size2 = whp2.y * whp2.z * sizeof(uchar);
	uchar* img1 = NULL;
	uchar* img2 = NULL;
	size_t tmp_pitch = 0;
	CHECK(cudaMallocPitch((void**)&img1, &tmp_pitch, sizeof(uchar) * whp1.x, whp1.y));
	CHECK(cudaMallocPitch((void**)&img2, &tmp_pitch, sizeof(uchar) * whp2.x, whp2.y));
	const size_t dpitch1 = sizeof(uchar) * whp1.z;
	const size_t spitch1 = sizeof(uchar) * whp1.x;
	const size_t dpitch2 = sizeof(uchar) * whp2.z;
	const size_t spitch2 = sizeof(uchar) * whp2.x;
	CHECK(cudaMemcpy2D(img1, dpitch1, limg.data, spitch1, spitch1, whp1.y, cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy2D(img2, dpitch2, rimg.data, spitch2, spitch2, whp2.y, cudaMemcpyHostToDevice));
	float t0 = timer.read();

	/* Reserve memory space for a whole bunch of SURF features. */
	surf::SurfData surf_data1, surf_data2;
	surf::initSurfData(surf_data1, max_npts, true, true);
	surf::initSurfData(surf_data2, max_npts, true, true);
	float* surf_descriptors1 = NULL;
	float* surf_descriptors2 = NULL;

	std::unique_ptr<surf::Surfor> detector(new surf::Surfor);
	detector->init(octaves, thres, doubleImageSize, initLobe * 3, samplingStep, upright, extended, indexSize, limg.cols, limg.rows);

	int nrepeats = 100;
	float t1 = timer.read();
	for (int i = 0; i < nrepeats; i++)
	{
		detector->detectAndCompute(img1, surf_data1, whp1, &surf_descriptors1, true);
		detector->detectAndCompute(img2, surf_data2, whp2, &surf_descriptors2, true);
	}

	float t2 = timer.read();
	for (int i = 0; i < nrepeats; i++)
	{
		detector->match(surf_data1, surf_data2, surf_descriptors1, surf_descriptors2);
	}

	float t3 = timer.read();
	std::cout << "Number of features1: " << surf_data1.num_pts << std::endl
		<< "Number of features2: " << surf_data2.num_pts << std::endl;
	std::cout << "Time for allocating image memory:  " << t0 << std::endl
		<< "Time for allocating point memory:  " << t1 - t0 << std::endl
		<< "Time of detection and computation: " << (t2 - t1) / nrepeats << std::endl
		<< "Time of matching surf keypoints:   " << (t3 - t2) / nrepeats << std::endl;

	// Show
	cv::Mat show1, show2, show_matched;
	drawKeypoints(surf_data1, limg, show1);
	drawKeypoints(surf_data2, rimg, show2);
	drawMatches(surf_data1, surf_data2, limg, rimg, show_matched, false);
	cv::imwrite("data/surf_show1.jpg", show1);
	cv::imwrite("data/surf_show2.jpg", show2);
	cv::imwrite("data/surf_show_matched.jpg", show_matched);

	// Free Sift data from device
	surf::freeSurfData(surf_data1);
	surf::freeSurfData(surf_data2);
	CHECK(cudaFree(img1));
	CHECK(cudaFree(img2));
	if (surf_descriptors1)
	{
		cudaFree(surf_descriptors1);
	}
	if (surf_descriptors2)
	{
		cudaFree(surf_descriptors2);
	}
}
