package smote;

import java.lang.reflect.Array;
import java.util.Arrays;
import java.util.Comparator;

import org.omg.CORBA.PUBLIC_MEMBER;

import weka.core.DistanceFunction;
import weka.core.expressionlanguage.common.IfElseMacro;

public class Density  {
	public int density;
	public double distance;
	public double avgDensity;
	public int index;
	public static double eps = 1e-5;
	public static double area;
	public Density() {
		density = 0;
		distance = 0.0f;
		index = -1;
	}
	
	

	public static void main(String[] args) {
		// TODO Auto-generated method stub
		Density[] densities = new Density[7];
		
		densities[0] = new Density();
		densities[0].density = 0;
		densities[0].distance = 1.2;
		densities[1] = new Density();
		densities[1].density = 0;
		densities[1].distance = 0.8;
		densities[2] = new Density();
		densities[2].density = 3;
		densities[2].distance = 0.7;
		densities[3] = new Density();
		densities[3].density = 5;
		densities[3].distance = 0.7;
		densities[4] = new Density();
		densities[4].density = 0;
		densities[4].distance = 0.3;
		densities[5] = new Density();
		densities[5].density = 4;
		densities[5].distance = 0.1;
		densities[6] = new Density();
		densities[6].density = 10;
		densities[6].distance = 10;
		//先根据密度来排序
		Arrays.sort(densities, new Comparator<Density>() {
			@Override
			public int compare(Density o1, Density o2) {
				// TODO Auto-generated method stub
				if(o1.density > o2.density) {
					return 1;
				}else if(o1.density < o2.density) {
					return -1;
				}
				return 0;
			}
		});
		Arrays.sort(densities, new Comparator<Density>() {

			@Override
			public int compare(Density o1, Density o2) {
				// TODO Auto-generated method stub
				if(o1.distance > o2.distance) {
					return 1;
				}else if(o1.distance < o2.distance){
					return -1;
				}
				return 0;
			}
		});
		Arrays.sort(densities,new Comparator<Density>() {

			@Override
			public int compare(Density o1, Density o2) {
				// TODO Auto-generated method stub
					double area1 = o1.density/(o1.distance*o1.distance*4);
					double area2 = o2.density/(o2.distance*o2.distance*4);
					if(area1 > area2 ) {
						return 1;
					}else if(area1 < area2 ) {
						return -1;
					}
				return 0;
			}	
		});
		
		for(int ii = 0; ii < densities.length; ++ii) {
			System.out.println("density:"+densities[ii].density+" "+densities[ii].distance);
		}
		
	}




	

}
