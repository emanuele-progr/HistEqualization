/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package histogramequalizationjavathread;
import java.awt.image.*;
import java.util.concurrent.Callable;

/**
 *
 * @author Emanuele
 */
public class ConvertToYCbCr implements Callable<int[]>{
    
    final private BufferedImage img;

    ConvertToYCbCr(BufferedImage subImg){
        this.img = subImg;
    }

    public int[] call(){
        return convertToYCbCr(this.img);
        
    }
    
    public int[] convertToYCbCr(BufferedImage subImage) {

    int cols = subImage.getWidth();
    int rows = subImage.getHeight();
    int[] histogram = new int[256];
    int[] im = new int[cols*3];

    for (int y = 0; y < rows; y++) {

        subImage.getRaster().getPixels(0, y, cols, 1, im);

        for (int x = 0; x < cols*3; x+=3) {

            int R = im[x+0];
            int G = im[x+1];
            int B = im[x+2];

            int Y = (int) (R * .257000 + G * .504000 + B * .098000 + 16);
            int Cb = (int) (R * -.148000 + G * -.291000 + B * .439000 + 128);
            int Cr = (int) (R * .439000 + G * -.368000 + B * -.071000 + 128);


            im[x+0] = Y;
            im[x+1] = Cb;
            im[x+2] = Cr;

            histogram[Y]++;

        }
        
        subImage.getRaster().setPixels(0, y, cols, 1, im);
        
    }

    return histogram;
    
    }
}
