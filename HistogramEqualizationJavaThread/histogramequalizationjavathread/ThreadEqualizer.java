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
public class ThreadEqualizer implements Runnable{
    int id;
    int subHistogram;
    int cols;
    int rows;
    int[] cumulativeHist;
    int[] equalizedHist;
    

    ThreadEqualizer(int id, int subHistogram, int[] cumulativeHist, int[] equalizedHist, int cols, int rows){

        this.id = id;
        this.subHistogram = subHistogram;
        this.cols = cols;
        this.rows = rows;

        
        this.cumulativeHist = cumulativeHist;
        this.equalizedHist = equalizedHist;
        
    }
    
        public void run(){

        for(int i = id*subHistogram; i < (id*subHistogram + subHistogram); i++){

            equalizedHist[i] = (int)((((float)(cumulativeHist[i] - cumulativeHist[0]))/((float)(cols*rows - 1))) * 255);

        }
    }
}
