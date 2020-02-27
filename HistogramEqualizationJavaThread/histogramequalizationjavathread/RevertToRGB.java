/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package histogramequalizationjavathread;
import java.awt.image.BufferedImage;

/**
 *
 * @author Emanuele
 */
public class RevertToRGB implements Runnable{
    
    int[] histogram;
    BufferedImage img;
    
    RevertToRGB(BufferedImage img, int [] histogram ){
        this.img = img;
        this.histogram = histogram;
    }
    
    public void run(){
        revertToRGB(img);
    }
    
    public void revertToRGB(BufferedImage subImage){

    int width = subImage.getWidth();
    int height = subImage.getHeight();
    int[] im = new int[width*3];

    for (int y = 0; y < height; y++){

        subImage.getRaster().getPixels(0, y, width, 1, im);

        for (int x = 0; x < width*3; x +=3) {

            int valueBefore = im[x];
            int valueEqualized = histogram[valueBefore];

            im[x] = valueEqualized;

            int Y = im[x+0];
            int Cb = im[x+1];
            int Cr = im[x+2];

            int R = Math.max(0, Math.min(255, (int) ((Y - 16) * 1.164 + 1.596 * (Cr - 128))));
            int G = Math.max(0, Math.min(255, (int) ((Y - 16) * 1.164 - 0.813 * (Cr - 128) - (0.392 * (Cb - 128)))));
            int B = Math.max(0, Math.min(255, (int) ((Y - 16) * 1.164 + 2.017 * (Cb - 128))));

            im[x+0] = R;
            im[x+1] = G;
            im[x+2] = B;
        }
        subImage.getRaster().setPixels(0, y, width, 1, im);
    }
   }
}
