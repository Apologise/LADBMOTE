package DPCD;

import java.util.ArrayList;
import java.util.Comparator;

import javax.xml.bind.ValidationEvent;

import smote.SETTING;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.core.expressionlanguage.common.IfElseMacro;

public class Searchcore {
	public static int[] func(Instances data) {
		int[] test = new int[data.size()];
		double[] density = new double[data.size()];
		for(int i = 0; i < data.size(); ++i) {
			for(int j = 0; j < data.size(); ++j) {
				density[i] += calDensity(data.get(i), data.get(j));
			}
		}
		return test;
		
	}

	public static double calCutDistance(Instances data) {
		double cd = 0.0f;
		double[][] distance = new double[data.size()][data.size()];
		ArrayList<Double> distance1 = new ArrayList<Double>();
		for(int i = 0; i < data.size(); ++i) {
			for(int j = 0; j < i; ++j) {
				distance1.add(calDistance(data.get(i), data.get(j)));
			}
		}
		distance1.sort(new Comparator<Double>() {

			@Override
			public int compare(Double o1, Double o2) {
				// TODO Auto-generated method stub
				if(o1 > o2) {
					return 1;
				}else if(o1 < o2) {
					return -1;
				}
				return 0;
			}
		
			
		});;
		for(Double a: distance1) {
			System.out.println(a);
		}
		int size = (int)(data.size()*(data.size()-1)*0.5);
		System.out.println(Math.floor(size*0.02));
		int index = (int) (Math.ceil(size*1)-1);
		cd = distance1.get(index);
		return cd;
	}
	
	public static double calDistance(Instance a, Instance b) {
		double result= 0.0f;
		for(int i = 0; i < a.numAttributes(); ++i) {
			double temp = a.value(i)-b.value(i);
			result += temp*temp;
		}
		return result;
	}
	public static double calDensity(Instance a, Instance b) {
		double result = 0.0f;
		result+= Math.exp(calDensity(a,b)/SETTING.dc*SETTING.dc);
		return result;
	}
	public static void main(String[] args) throws Exception {
		// TODO Auto-generated method stub
		Instances data = DataSource.read("dataset/test.arff");
		System.out.println(calCutDistance(data));
	}

}
