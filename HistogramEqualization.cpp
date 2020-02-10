// HistogramEqualization.cpp : Questo file contiene la funzione 'main', in cui inizia e termina l'esecuzione del programma.
//
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <chrono>
#include <iostream>

using namespace std;
using namespace cv;

void convertToYCbCr(int width, int height, unsigned char * im_ptr, int histogram[]){

    for (int i = 0; i < height * width * 3; i += 3) {

        int R = im_ptr[i + 0];
        int G = im_ptr[i + 1];
        int B = im_ptr[i + 2];

        int Y = R * .257000 + G * .504000 + B * .098000 + 16;
        int Cb = R * -.148000 + G * -.291000 + B * .439000 + 128;
        int Cr = R * .439000 + G * -.368000 + B * -.071000 + 128;

        im_ptr[i + 0] = Y;
        im_ptr[i + 1] = Cb;
        im_ptr[i + 2] = Cr;

        histogram[Y] ++;
    }

 }
    


void equalizeHist(int histogram[], int equalizedHist[], int cols, int rows)
{
    int cumulative_histogram[256];

    cumulative_histogram[0] = histogram[0];

    for (int i = 1; i < 256; i++)
    {
        cumulative_histogram[i] = histogram[i] + cumulative_histogram[i - 1];
        equalizedHist[i] = (int)(((float)cumulative_histogram[i] - histogram[0]) / ((float)cols * rows - 1) * 255);

    }
}

void revertToRGB(unsigned char * im_ptr, int width, int height, int equalizedHist[]) {
    for (int i = 0; i < height * width * 3; i += 3) {

        int value_before = im_ptr[i];
        int value_equalized = equalizedHist[value_before];

        im_ptr[i] = value_equalized;

        int Y = im_ptr[i + 0];
        int Cb = im_ptr[i + 1];
        int Cr = im_ptr[i + 2];

        unsigned char R = (unsigned char)max(0, min(255, (int)((Y - 16) * 1.164 + 1.596 * (Cr - 128))));
        unsigned char G = (unsigned char)max(0, min(255, (int)((Y - 16) * 1.164 - 0.813 * (Cr - 128) - (0.392 * (Cb - 128)))));
        unsigned char B = (unsigned char)max(0, min(255, (int)((Y - 16) * 1.164 + 2.017 * (Cb - 128))));

        im_ptr[i + 0] = R;
        im_ptr[i + 1] = G;
        im_ptr[i + 2] = B;
    }

}



int main()
{
    // Load the image
    String folderpath = "img/*.jpg";
    vector<String> filenames;
    double timesAdded = 0;
    int imageCounter = 0;
    glob(folderpath, filenames);
    for (size_t i = 0; i < filenames.size(); i++)
    {
        Mat im = imread(filenames[i]);
        resize(im, im, Size(2000, 2000), INTER_NEAREST);
        unsigned char* im_ptr = im.ptr();
        //imshow("Original Image", im);
        //waitKey();

        
        int width = im.cols;
        int height = im.rows;

        auto start = chrono::steady_clock::now();

        //int* YCbCrvector = new int[height * width * 3];

        // Generate the histogram and convert RGB -> YCbCr

        int histogram[256];

        for (int i = 0; i < 256; i++) {
            histogram[i] = 0;
        }
      

        convertToYCbCr(width, height, im_ptr, histogram);

        // Generate the equalized histogram
        int equalizedHist[256];
        equalizeHist(histogram, equalizedHist, width, height);

        // Back to RGB
        revertToRGB(im_ptr, width, height, equalizedHist);

        auto end = chrono::steady_clock::now();
        double elapsed_time = chrono::duration_cast<chrono::milliseconds>(end - start).count();

        cout << "elapsed : " << elapsed_time << " ms" << endl;
        //imshow("Equalized Image", im);
        //waitKey();

        timesAdded += elapsed_time;
        imageCounter += 1;


    }

    double meanTimes = timesAdded / imageCounter;
    cout << "MEAN ELAPSED TIME : " << meanTimes << " ms" << endl;

}

