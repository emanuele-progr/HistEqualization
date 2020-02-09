#include <iostream>
#include <stdlib.h>
#include <string>
#include <iostream>
#include <chrono>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>


using namespace std;
using namespace cv;
#define CUDA_CHECK_RETURN(value) CheckCudaErrorAux(__FILE__,__LINE__, #value, value)

__global__ void convertToYCbCr(unsigned char* image, int width, int height, int* histogram) {

	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	long index;

	//shared memory histogram

	__shared__ int histShared[256];

	for (int j = threadIdx.x; j < 256; j += blockDim.x) {
		histShared[j] = 0;
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


		atomicAdd(&(histShared[Y]), 1);

		image[index] = Y;
		image[index + 1] = Cb;
		image[index + 2] = Cr;
	}

	__syncthreads();

	//reconstruct histogram

	for (int j = threadIdx.x; j < 256; j += blockDim.x) {
		atomicAdd(&(histogram[j]), histShared[j]);
	}
}

__global__ void equalize(int* equalized, int* cumulative_dist, int* histogram, int width, int height) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	for (int k = idx; k < 256; k += blockDim.x * gridDim.x) {
		equalized[k] = (int)(((float)cumulative_dist[k] - histogram[0]) / ((float)width * height - 1) * 255);
	}
}

__global__ void  revertToRGB(unsigned char* image, int* cumulative_dist, int* histogram, int* equalized, int width, int height) {

	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	long index;

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

//Check the return value of the CUDA runtime API call and exit the application if the call has failed.

static void CheckCudaErrorAux(const char* file, unsigned line, const char* statement, cudaError_t err)
{
	if (err == cudaSuccess)
		return;
	std::cerr << statement << " returned " << cudaGetErrorString(err) << "(" << err << ") at " << file << ":" << line << std::endl;
	exit(1);
}

static void InitCUDA() {

	// Init function to exclude start up delay overhead

	CUDA_CHECK_RETURN(cudaFree(0));

	// basically, does nothing.

}

int main() {
	// Load the image
	String folderpath = "C:/Users/Emanuele/source/repos/HistogramEqualizationCUDA/HistogramEqualizationCUDA/img/*.jpg";
	vector<String> filenames;
	double timesAdded = 0;
	int imageCounter = 0;
	InitCUDA();
	glob(folderpath, filenames);

	for (size_t i = 0; i < filenames.size(); i++) {

		Mat im = imread(filenames[i]);
		resize(im, im, Size(800, 600), INTER_NEAREST);
		//imshow("Original Image", im);
		//waitKey();

		int width = im.cols;
		int height = im.rows;

		auto start = chrono::steady_clock::now();

		int host_equalized[256];						//cpu equalized histogram
		int host_cumulative_dist[256];

		unsigned char* host_image = im.ptr();		//Mat image to array image
		int host_histogram[256] = { 0 };					//cpu histogram
		
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
		CUDA_CHECK_RETURN(cudaMemcpy(device_histogram, host_histogram, sizeof(int) * 256, cudaMemcpyHostToDevice));			

		int block_size = 256;
		int grid_size = (width * height + (block_size - 1)) / block_size;

		//first kernel to build histogram and convert to YCbCr.

		convertToYCbCr << <grid_size, block_size >> > (device_image, width, height, device_histogram);		

		//copy to host the histogram computed.

		CUDA_CHECK_RETURN(cudaMemcpy(host_histogram, device_histogram, sizeof(int) * 256, cudaMemcpyDeviceToHost));

		//calculate cumulative distribution hist

		host_cumulative_dist[0] = host_histogram[0];										
																						
		for (int i = 1; i < 256; i++) {														
			host_cumulative_dist[i] = host_histogram[i] + host_cumulative_dist[i - 1];		
		}																					

		//copy to device

		CUDA_CHECK_RETURN(cudaMemcpy(device_cumulative_dist, host_cumulative_dist, sizeof(int) * 256, cudaMemcpyHostToDevice));
		CUDA_CHECK_RETURN(cudaMemcpy(device_equalized, host_equalized, sizeof(int) * 256, cudaMemcpyHostToDevice));

		//second kernel that do the real equalization.

		equalize << <grid_size, block_size >> > (device_equalized, device_cumulative_dist, device_histogram, width, height);					

		//third and last kernel to go back to RGB.

		revertToRGB << <grid_size, block_size >> > (device_image, device_cumulative_dist, device_histogram, device_equalized, width, height);	

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

		Mat final_image = Mat(Size(width, height), CV_8UC3, host_image);
		//imwrite(filenames[i] + "Equalized.jpg",final_image);
		//imshow("Final Image", final_image);
		//waitKey();

		timesAdded += elapsed_time;
		imageCounter += 1;
	}

	double meanTimes = timesAdded / imageCounter;
	cout << "MEAN ELAPSED TIME : " << meanTimes << " ms" << endl;

}