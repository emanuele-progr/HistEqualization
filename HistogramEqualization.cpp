// HistogramEqualization.cpp : Questo file contiene la funzione 'main', in cui inizia e termina l'esecuzione del programma.
//
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <chrono>
#include <iostream>

using namespace std;
using namespace cv;

void convertToYCbCr(Mat im, int histogram[], int YCbCrvector[]){

    for (int i = 0; i < im.rows; i++) {
        for (int j = 0; j < im.cols; j++) {

            Vec3b intensity = im.at<Vec3b>(i, j);

            int R = intensity.val[0];
            int G = intensity.val[1];
            int B = intensity.val[2];

            //conversion to YCbCr space

            int Y = R * .257000 + G * .504000 + B * .098000 + 16;
            int Cb = R * -.148000 + G * -.291000 + B * .439000 + 128;
            int Cr = R * .439000 + G * -.368000 + B * -.071000 + 128;

            histogram[Y]++;

            int index = (j * im.rows + i) * 3;

            YCbCrvector[index] = Y;
            YCbCrvector[index + 1] = Cb;
            YCbCrvector[index + 2] = Cr;

        }
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

void revertToRGB(Mat im, int width, int height, int equalizedHist[], int YCbCrvector[]) {
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {

            int index = (j * height + i) * 3;

            int Y = equalizedHist[YCbCrvector[index]];
            int Cb = YCbCrvector[index + 1];
            int Cr = YCbCrvector[index + 2];

            unsigned char R = (unsigned char)max(0, min(255, (int)((Y - 16) * 1.164 + 1.596 * (Cr - 128))));
            unsigned char G = (unsigned char)max(0, min(255, (int)((Y - 16) * 1.164 - 0.813 * (Cr - 128) - (0.392 * (Cb - 128)))));
            unsigned char B = (unsigned char)max(0, min(255, (int)((Y - 16) * 1.164 + 2.017 * (Cb - 128))));

            Vec3b intensity = im.at<Vec3b>(i, j);

            intensity.val[0] = R;
            intensity.val[1] = G;
            intensity.val[2] = B;

            im.at<Vec3b>(i, j) = intensity;
        }
    }
}



int main()
{
    // Load the image
    String folderpath = "C:/Users/Emanuele/source/repos/HistogramEqualization/HistogramEqualization/img/*.jpg";
    vector<String> filenames;
    double timesAdded = 0;
    int imageCounter = 0;
    glob(folderpath, filenames);
    for (size_t i = 0; i < filenames.size(); i++)
    {
        Mat im = imread(filenames[i]);
        resize(im, im, Size(800, 600), INTER_NEAREST);
        imshow("Original Image", im);
        waitKey();

        int width = im.cols;
        int height = im.rows;

        auto start = chrono::steady_clock::now();

        int* YCbCrvector = new int[height * width * 3];

        // Generate the histogram and convert RGB -> YCbCr
        int histogram[256];
        // initialize all intensity values to 0
        for (int i = 0; i < 256; i++) {
            histogram[i] = 0;
        }

        convertToYCbCr(im, histogram, YCbCrvector);

        // Generate the equalized histogram
        int equalizedHist[256];
        equalizeHist(histogram, equalizedHist, width, height);

        // Back to RGB
        revertToRGB(im, width, height, equalizedHist, YCbCrvector);

        auto end = chrono::steady_clock::now();
        double elapsed_time = chrono::duration_cast<chrono::milliseconds>(end - start).count();

        cout << "elapsed : " << elapsed_time << " ms" << endl;
        imshow("Equalized Image", im);
        waitKey();

        timesAdded += elapsed_time;
        imageCounter += 1;


    }

    double meanTimes = timesAdded / imageCounter;
    cout << "MEAN ELAPSED TIME : " << meanTimes << " ms" << endl;

}

