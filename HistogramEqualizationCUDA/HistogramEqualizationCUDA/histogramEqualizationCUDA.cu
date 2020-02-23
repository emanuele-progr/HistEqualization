#include <iostream>
#include <stdlib.h>
#include <string>
#include <iostream>
#include <chrono>
#include <fstream>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>


using namespace std;
using namespace cv;

#define CUDA_CHECK_RETURN(value) CheckCudaErrorAux(__FILE__,__LINE__, #value, value)
#define HIST_LENGHT 256

//Check the return value of the CUDA runtime API call and exit the application if the call has failed.

static void CheckCudaErrorAux(const char* file, unsigned line, const char* statement, cudaError_t err)
{
	if (err == cudaSuccess)
		return;
	std::cerr << statement << " returned " << cudaGetErrorString(err) << "(" << err << ") at " << file << ":" << line << std::endl;
	exit(1);
}


inline int cudaDeviceInit()
{
	CUDA_CHECK_RETURN(cudaFree(0));
	int deviceCount;
	CUDA_CHECK_RETURN(cudaGetDeviceCount(&deviceCount));

	if (deviceCount == 0)
	{
		std::cerr << "CUDA error: no devices supporting CUDA." << std::endl;
		exit(EXIT_FAILURE);
	}

	int dev = 0;
	cudaDeviceProp deviceProp;
	CUDA_CHECK_RETURN(cudaGetDeviceProperties(&deviceProp, dev));
	std::cerr << "cudaSetDevice GPU" << dev << " = " << deviceProp.name << std::endl;

	CUDA_CHECK_RETURN(cudaSetDevice(dev));

	return dev;
}

__global__ void calcCumulativeHist(int* cumulative_dist, int* histogram,
	int width, int height) {

	__shared__ int partialScan[HIST_LENGHT];
	int i = threadIdx.x + blockDim.x * blockIdx.x;

	//load phase, load values on shared memory
	if (i <  HIST_LENGHT )
		partialScan[i] = histogram[i];
	__syncthreads();

	//work efficient reduction phase
	for (unsigned int stride = 1; stride <= HIST_LENGHT / 2; stride *= 2) {
		unsigned int index = (threadIdx.x + 1) * stride * 2 - 1;
		if (index <  HIST_LENGHT )
			partialScan[index] += partialScan[index - stride];
		__syncthreads();
	}

	//work efficient post reduction phase
	for (unsigned int stride = HIST_LENGHT / 2; stride > 0; stride /= 2) {
		__syncthreads();
		unsigned int index = (threadIdx.x + 1) * stride * 2 - 1;
		if (index + stride < HIST_LENGHT ) {
			partialScan[index + stride] += partialScan[index];
		}
	}

	//back on global memory
	__syncthreads();
	if (i < HIST_LENGHT) {
		cumulative_dist[i] += partialScan[i];
	}

}

__global__ void convertToYCbCr(unsigned char* image, int width, int height, int* histogram) {

	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	long index;


	__shared__ int hist_priv[HIST_LENGHT];

	for (int bin_idx = threadIdx.x; bin_idx < HIST_LENGHT; bin_idx += blockDim.x) {
		hist_priv[bin_idx] = 0;
	}

	__syncthreads();
	
	for (int i = idx; i < width * height; i += blockDim.x * gridDim.x) {

		index = i * 3;

		int R = image[index];
		int G = image[index + 1];
		int B = image[index + 2];

		int Y = R * .257000 + G * .504000 + B * .098000 + 16;
		int Cb = R * -.148000 + G * -.291000 + B * .439000 + 128;
		int Cr = R * .439000 + G * -.368000 + B * -.071000 + 128;


		atomicAdd(&(hist_priv[Y]), 1);

		image[index] = Y;
		image[index + 1] = Cb;
		image[index + 2] = Cr;
	}

	__syncthreads();

	//The shared histograms are added to the global histogram.
	for (int bin_idx = threadIdx.x; bin_idx < HIST_LENGHT; bin_idx += blockDim.x) {
		atomicAdd(&(histogram[bin_idx]), hist_priv[bin_idx]);
	}




}

__global__ void equalize(int* equalized, int* cumulative_dist, int width, int height) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	for (int k = idx; k < HIST_LENGHT; k += blockDim.x * gridDim.x) {
		equalized[k] = (int)(((float)cumulative_dist[k] - cumulative_dist[0]) / ((float)width * height - 1) * 255);
	}
}

__global__ void  revertToRGB(unsigned char* image, int* equalized, int width, int height) {

	long index;
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	for (int i = idx; i < width * height; i += blockDim.x * gridDim.x) {

		index = i * 3;

		int Y = equalized[image[index]];
		int Cb = image[index + 1];
		int Cr = image[index + 2];

		unsigned char R = (unsigned char)max(0, min(255, (int)((Y - 16) * 1.164 + 1.596 * (Cr - 128))));
		unsigned char G = (unsigned char)max(0, min(255, (int)((Y - 16) * 1.164 - 0.813 * (Cr - 128) - (0.392 * (Cb - 128)))));
		unsigned char B = (unsigned char)max(0, min(255, (int)((Y - 16) * 1.164 + 2.017 * (Cb - 128))));


		image[index] = R;
		image[index + 1] = G;
		image[index + 2] = B;

	}

}




int main() {
	// Load the image
	String folderpath = "img/*.jpg";
	vector<String> filenames;
	double timesAdded = 0;
	int imageCounter = 0;
	cudaDeviceInit();
	glob(folderpath, filenames);

	for (size_t i = 0; i < filenames.size(); i++) {

		Mat im = imread(filenames[i]);
		resize(im, im, Size(800, 600), INTER_NEAREST);
		//imshow("Original Image", im);
		//waitKey();

		int width = im.cols;
		int height = im.rows;


		auto start = chrono::steady_clock::now();

		int host_equalized[HIST_LENGHT];						//cpu equalized histogram
		//int host_cumulative_dist[HIST_LENGHT] = { 0 };

		unsigned char* host_image = im.ptr();		//Mat image to array image

		//int host_histogram[HIST_LENGHT] = { 0 };					//cpu histogram

		unsigned char* device_image;	//gpu image

		int* device_histogram;			//gpu histogram
		int* device_equalized;			//gpu equalized histogram
		int* device_cumulative_dist;	//gpu cumulative dist.

		//allocate gpu global memory.

		CUDA_CHECK_RETURN(cudaMalloc((void**)&device_image, sizeof(char) * (width * height * 3)));
		CUDA_CHECK_RETURN(cudaMalloc((void**)&device_histogram, sizeof(int) * 256));
		CUDA_CHECK_RETURN(cudaMalloc((void**)&device_equalized, sizeof(int) * 256));
		CUDA_CHECK_RETURN(cudaMalloc((void**)&device_cumulative_dist, sizeof(int) * 256));

		//copy the image on global memory.
		
		CUDA_CHECK_RETURN(cudaMemcpy(device_image, host_image, sizeof(char) * (width * height * 3), cudaMemcpyHostToDevice));

		//initialize gpu hist and cumulative hist

		CUDA_CHECK_RETURN(cudaMemset(device_histogram, 0, sizeof(int) * 256));
		CUDA_CHECK_RETURN(cudaMemset(device_cumulative_dist, 0, sizeof(int) * 256));

		int block_size = HIST_LENGHT;
		int grid_size = ((width * height + (block_size - 1)) / block_size) / 2 ;

		//first kernel to build histogram and convert to YCbCr.
		
		convertToYCbCr << <grid_size, block_size >> > (device_image, width, height, device_histogram);

		//CUDA_CHECK_RETURN(cudaMemcpy(host_histogram, device_histogram, sizeof(int) * 256, cudaMemcpyDeviceToHost));
		
		/*
		host_cumulative_dist[0] = host_histogram[0];

		for (int i = 1; i < HIST_LENGHT; i++) {
			host_cumulative_dist[i] = host_histogram[i] + host_cumulative_dist[i - 1];
		}
		
		*/

		//CUDA_CHECK_RETURN(cudaMemcpy(device_equalized, host_equalized, sizeof(int) * 256, cudaMemcpyHostToDevice));
		//CUDA_CHECK_RETURN(cudaMemcpy(device_cumulative_dist, host_cumulative_dist, sizeof(int) * 256, cudaMemcpyHostToDevice));

		calcCumulativeHist << <1, block_size >> > (device_cumulative_dist, device_histogram, width, height);

		//second kernel that do the real equalization.

		equalize << <1, block_size >> > (device_equalized, device_cumulative_dist,  width, height);

		//third and last kernel to go back to RGB.

		revertToRGB << <grid_size, block_size >> > (device_image, device_equalized, width, height);

		//retrieve the equalized image

		CUDA_CHECK_RETURN(cudaMemcpy(host_image, device_image, sizeof(char) * (width * height * 3), cudaMemcpyDeviceToHost));

		//release gpu memory

		CUDA_CHECK_RETURN(cudaFree(device_image));
		CUDA_CHECK_RETURN(cudaFree(device_histogram));
		CUDA_CHECK_RETURN(cudaFree(device_equalized));
		CUDA_CHECK_RETURN(cudaFree(device_cumulative_dist));

		auto end = chrono::steady_clock::now();
		double elapsed_time = chrono::duration_cast<chrono::milliseconds>(end - start).count();


		cout << "elapsed : " << elapsed_time;

		cout << "correctly freed memory \n";

		Mat equalized_image = Mat(Size(width, height), CV_8UC3, host_image);
		//imwrite(filenames[i] + "Equalized.jpg",final_image);
		imshow("Final Image", equalized_image);
		waitKey();

		timesAdded += elapsed_time;
		imageCounter += 1;
	}

	double meanTimes = timesAdded / imageCounter;
	cout << "MEAN ELAPSED TIME : " << meanTimes << " ms" << endl;

}