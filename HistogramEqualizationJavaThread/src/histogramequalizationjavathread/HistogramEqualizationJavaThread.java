/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package histogramequalizationjavathread;

/**
 *
 * @author Emanuele
 */

import javax.imageio.ImageIO;
import javax.swing.*;
import java.awt.*;
import java.awt.image.*;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.CompletionService;
import java.util.concurrent.ExecutorCompletionService;

import static java.lang.Math.ceil;

public class HistogramEqualizationJavaThread {

       public static int subdivision(double data1, int cores){
           return (int)ceil(data1 / (double)cores);
       }
       
       
       public static BufferedImage resize(BufferedImage img, int newW, int newH) {
        Image tmp = img.getScaledInstance(newW, newH, Image.SCALE_SMOOTH);
        BufferedImage res_img = new BufferedImage(newW, newH, BufferedImage.TYPE_INT_RGB);

        Graphics2D g2d = res_img.createGraphics();
        g2d.drawImage(tmp, 0, 0, null);
        g2d.dispose();

        return res_img;
    }
       

    
    
    public static void main(String[] args) {
        
        int imageCounter = 0;
        long sumTimes = 0;
        
        //load images folder and and the corresponding images
        
        File folder = new File("img");
        File[] listOfFiles = folder.listFiles();
        for (File file : listOfFiles) {
            if (file.isFile()) {
                BufferedImage img = null;
             
             try{
                 
                 String s = "img/" + file.getName();
                 img = ImageIO.read(new File(s));
                 img = resize(img, 7680, 4800);
                   
                 
             }catch(IOException e) {
                 System.out.println(e);
                 return;
             }
         
             try{

                //ImageIcon icon = new ImageIcon(resize(img, 300, 300));
                //JLabel label = new JLabel(icon, JLabel.CENTER);
                //JOptionPane.showMessageDialog(null, label, "Original image", -1);
                int cores = Runtime.getRuntime().availableProcessors();
                
                long startTime = System.currentTimeMillis();
                
                int height = img.getHeight();
                int width = img.getWidth();

                int[] histogram = new int[256];
                int[] equalizedHist = new int[256];
                
                //allocate an array of future to take results of parallel execution through callable function
                
                ArrayList<Future> futures = new ArrayList<>();
                
                //we use a fixed pool of threads equal to the number of cores/threads
                
                ExecutorService exec = Executors.newFixedThreadPool(cores);
                CompletionService<int []> completionService = new ExecutorCompletionService<int []>(exec);
                //divide the image according to the number of threads

                int yStart = subdivision((double)height, cores);
                int subHeight = subdivision((double)height, cores);
                
                for (int i = 0; i < cores; i++) {

                    if(i == cores-1 && height%cores != 0){
                        subHeight = height - i*subdivision((double)height, cores);
                        //System.out.println(subHeight);
                    }
                    BufferedImage subImg = img.getSubimage(0,(i*yStart),(width),(subHeight));

                    //start a thread that run the callable function on the subImg to convert RGB->YCbCr
                    completionService.submit(new ConvertToYCbCr(subImg));
                    //futures.add(exec.submit(new ConvertToYCbCr(subImg)));

                }
                //Wait for the pool to finish and reconstruct the histogram
                
                for(Future<int[]> future : futures){
                    future = completionService.take();
                    int[] localHist = future.get();
                    
                    for(int j = 0; j < 256; j++) {
                        histogram[j] += localHist[j];
                    }
                }
                
                //compute the cumulative histogram
                
                int []cumulativeHist = new int[256];
                cumulativeHist[0] = histogram[0];
                
                for (int i = 1; i < 256; i++){
                    
                    cumulativeHist[i] = histogram[i] + cumulativeHist[i - 1];
                    
                }
                
                //find a fair division of the histogram based on the number of threads
                
                int subHistogram = subdivision((double)256, cores);
                //System.out.println(subHistogram);
                //clear future values and start the equalizing process on the subHistogram
                //each thread runs the equalization on a part of the histogram

                futures.clear();
                

                for(int i = 0; i < cores; i++){
                    
                    if(i == cores -1 && 256 % cores != 0){
                        subHistogram = 256 - i*subdivision((double)256, cores);
                    }
                    
                    futures.add(exec.submit(new ThreadEqualizer(i, subHistogram, cumulativeHist, equalizedHist, width, height)));
                }
                //get histogram_equalized and clear futures

                for(Future future : futures){
                    future.get();
                }
                
                futures.clear();
                
                //same as conversion/first pool
                
                subHeight = subdivision((double)height,cores);

                for (int i = 0; i < cores; i++){

                    if(i == cores-1 && height%cores != 0){
                        subHeight = height - i*subdivision((double)height,cores);
                    }
                    
                    BufferedImage subImg = img.getSubimage(0,(i*yStart),(width),(subHeight));

                    futures.add(exec.submit(new RevertToRGB(subImg, equalizedHist)));
                    
                }

                exec.shutdown();
                exec.awaitTermination(1, TimeUnit.MINUTES);

                long endTime = System.currentTimeMillis();
                long duration = (endTime - startTime); 
                System.out.println(duration + " ms");

                
                //icon = new ImageIcon(resize(img, 300, 300));
                //label = new JLabel(icon, JLabel.CENTER);
                //JOptionPane.showMessageDialog(null, label, "Image Equalized", -1);

                imageCounter += 1;
                sumTimes += duration;
         
            


            }catch(Exception e){
                
            System.out.println(e);
            
            }
            
            }
        }
        long meanTimes = sumTimes/imageCounter;
        System.out.println("Mean times : " + meanTimes + " ms");
    }
}
