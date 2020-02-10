/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package histogramequalizationjava;

/**
 *
 * @author Emanuele
 */

import javax.imageio.ImageIO;
import javax.swing.*;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.awt.image.Raster;
import java.io.File;
import java.io.IOException;

import static java.lang.Math.round;
public class HistogramEqualizationJava {

       public static BufferedImage resize(BufferedImage img, int newW, int newH) {
        Image tmp = img.getScaledInstance(newW, newH, Image.SCALE_SMOOTH);
        BufferedImage dimg = new BufferedImage(newW, newH, BufferedImage.TYPE_INT_RGB);

        Graphics2D g2d = dimg.createGraphics();
        g2d.drawImage(tmp, 0, 0, null);
        g2d.dispose();

        return dimg;
    }


        
    public static void convertToYCbCr(BufferedImage image, int[] histogram){
        
        
        Raster raster = image.getRaster();

        for (int y = 0; y < image.getHeight(); y++) {
            int[] iarray = new int[image.getWidth()*3];

            raster.getPixels(0, y, image.getWidth(), 1, iarray);

            for (int x = 0; x < image.getWidth()*3; x+=3) {

                int R = iarray[x+0];
                int G = iarray[x+1];
                int B = iarray[x+2];

                int Y = (int) (R * .257000 + G * .504000 + B * .098000 + 16);
                int Cb = (int) (R * -.148000 + G * -.291000 + B * .439000 + 128);
                int Cr = (int) (R * .439000 + G * -.368000 + B * -.071000 + 128);


                iarray[x+0] = Y;
                iarray[x+1] = Cb;
                iarray[x+2] = Cr;

                histogram[Y] ++;
            }

            image.getRaster().setPixels(0, y, image.getWidth(), 1, iarray);
        }
    }
    
    public static void equalizeHist(int [] histogram, int [] equalizedHist, int cols, int rows){
        
        int []cumulative_histogram = new int[256];
        cumulative_histogram[0] = histogram[0];
        
        for (int i = 1; i < 256; i++){
            
            cumulative_histogram[i] = histogram[i] + cumulative_histogram[i - 1];
            equalizedHist[i] = (int)(((float)cumulative_histogram[i] - histogram[0]) / ((float)cols * rows - 1) * 255);
        }
        
    }
    
    
    public static void revertToRGB(BufferedImage img, int [] equalizedHist){
        
        Raster raster = img.getRaster();

        for (int y = 0; y < img.getHeight(); y ++) {
            int[] iarray = new int[img.getWidth()*3];

            raster.getPixels(0, y, img.getWidth(), 1, iarray);

            for (int x = 0; x < img.getWidth()*3; x+=3) {

                int valueBefore = iarray[x];
                int valueAfter = equalizedHist[valueBefore];

                iarray[x] = valueAfter;

                int Y = iarray[x+0];
                int Cb = iarray[x+1];
                int Cr = iarray[x+2];

                int R = Math.max(0, Math.min(255, (int) ((Y - 16) * 1.164 + 1.596 * (Cr - 128))));
                int G = Math.max(0, Math.min(255, (int) ((Y - 16) * 1.164 - 0.813 * (Cr - 128) - (0.392 * (Cb - 128)))));
                int B = Math.max(0, Math.min(255, (int) ((Y - 16) * 1.164 + 2.017 * (Cb - 128))));

                iarray[x+0] = R;
                iarray[x+1] = G;
                iarray[x+2] = B;
            }
            img.getRaster().setPixels(0,y, img.getWidth(), 1,  iarray);
        }
    }
    
    public static void main(String[] args) {
        int image_counter = 0;
        long sum_times = 0;
        File folder = new File("img");
        File[] listOfFiles = folder.listFiles();
        for (File file : listOfFiles) {
            if (file.isFile()) {
                BufferedImage img = null;
             
             try{
                 String s = "img/" + file.getName();
                 img = ImageIO.read(new File(s));
                 img = resize( img ,800, 600);
         
         
                 
                 ImageIcon icon = new ImageIcon(img);
                 JLabel label = new JLabel(icon, JLabel.CENTER);
                 JOptionPane.showMessageDialog(null, label, "Original image", -1);

                 int cols = img.getWidth();
                 int rows = img.getHeight();
                 
                 long startTime = System.currentTimeMillis();

                 int[] histogram = new int[256];

                 for (int i = 0; i < 256; i++) {
                    histogram[i] = 0;
                 }

                 convertToYCbCr(img, histogram);

                 int[] equalizedHist = new int[256];
                 equalizeHist(histogram, equalizedHist, cols, rows);

                 revertToRGB(img, equalizedHist);
                 
                 long endTime = System.currentTimeMillis();
                 long duration = (endTime - startTime); 
                 System.out.println(duration + " ms");

                 //icon = new ImageIcon(resize(img, 800, 600));
                 //label = new JLabel(icon, JLabel.CENTER);
                 //JOptionPane.showMessageDialog(null, label, "Image Equalized", -1);
                 image_counter += 1;
                 sum_times += duration;
                 
             }catch(IOException e){
                 
            System.out.println(e);
            
             }
            }

            
        }
        long mean_times = sum_times/image_counter;
        System.out.println("MEAN TIMES : " + mean_times + " ms");
    }
}
            
