package DPCD;

import java.util.Random;
import java.util.UUID;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Date;
import javax.imageio.stream.IIOByteBuffer;
import javax.print.attribute.standard.DateTimeAtCompleted;
import javax.swing.plaf.basic.BasicInternalFrameTitlePane.SystemMenuBar;



public class test {
	
	public static boolean func(double x, double y) {
		if(x>=70&&x<=110&&y>=30&&y<=150) {
			return true;
		}else {
			return false;
		}
	}
	public static void main(String[] args) throws InterruptedException, IOException {
		// TODO Auto-generated method stub
		FileWriter fw = new FileWriter("dataset/testfile.arff", false);
		FileWriter fwmaj = new FileWriter("dataset/majtestfile.arff", false);
		FileWriter fwmin= new FileWriter("dataset/minestfile.arff", false);
	    fw.write("@relation glass1\r\n" + 
	    		"@attribute RI real [0, 150]\r\n" + 
	    		"@attribute Na real [0, 150]\r\n" +
	    		"@attribute Class {0, 1}\r\n" + 
	    		"@data\r\n");
		double[] tempx = new double[600];
		double[] x = new double[300];
		double[] y = new double[300];
		int[]  z = new int[600];
		for(int i = 0; i < 600; i++) {
			
				tempx[i] = Math.random()*150;
				tempx[i] = tempx[i]+Math.random();
			
			if(Math.random() > 0.5) {
				z[i]=0;
			}else {
				z[i]=1;
			}
			if(i >= 300) {
				y[i-300] = tempx[i];
			}else {
				x[i] = tempx[i];
			}
		}

		
	    
	    
		for(int ii = 0; ii < 300; ++ii ) {
			
			if(ii%6==0) {
				System.out.println();
			}
			if(ii==0) {
				
			}else {
			
			}
			if(func(x[ii], y[ii])) {
					fwmaj.write("{value:["+String.format("%.2f",x[ii])+","+String.format("%.2f",y[ii])+"],color:'#F3A43B'},\r\n");
					fw.write(String.format("%.2f", x[ii])+","+String.format("%.2f", y[ii])+",0\r\n");
		
			}else if((x[ii]>110&&x[ii]<130&&y[ii]>=100&&y[ii]<=130)||(x[ii]>=50&&x[ii]<=80&&y[ii]>=20&&y[ii]<=70)){
				fwmin.write("{value:["+String.format("%.2f",x[ii])+","+String.format("%.2f",y[ii])+"],color:'#60C0DD'},\r\n");
				fw.write(String.format("%.2f", x[ii])+","+String.format("%.2f", y[ii])+",1\r\n");
			}
			
			
		}
		fwmaj.close();
		fwmin.close();
		fw.close();
	}

}
