package smote;

import java.util.ArrayList;
import java.util.List;

import org.ujmp.core.util.matrices.SystemEnvironmentMatrix;

import weka.core.Instance;
import weka.core.Instances;
import weka.filters.unsupervised.attribute.MakeIndicator;

//计算少数类样本的概率
public class CalP {
	public static double[] calProbility(Instances minority, Instances majority) {
		double[] prob = new double[minority.size()];
		double[] density = new double[minority.size()];
		double[] distance = new double[minority.size()];
		for(int i = 0; i < minority.size(); ++i) {
			density[i] = calNeighbors(minority.get(i), minority);
			distance[i] = calToMajDistance(minority.get(i),majority);
		}
		SETTING.DH_distance = calCH_distance(minority, majority);
		double temp_density = 0.0f;
		double temp_distance = 0.0f;
		for(int i = 0; i < density.length; ++i) {
			temp_density += density[i];
			temp_distance += distance[i];
		}
	
		return distance;
	}
	/*
	public static double function(double distance, double DH) {
		double result;
		if(distance < DH) {
			result = 1/distance;
		}else {
			result = 
		}
	}
	*/
	public static double calDistance(Instance a, Instance b) {
		double distance = 0.0f;
		for(int i = 0; i < a.numAttributes()-1; ++i) {
			double temp = a.value(i) - b.value(i);
			distance += temp*temp;
		}
		distance = Math.sqrt(distance)/(a.numAttributes()-1);
		return distance;
	}
	public static double calNeighbors(Instance inst,Instances minority) {
		double result = 0.0f;
		//先计算该点到每个点之间的距离
		ArrayList<Double> distance = new ArrayList<>();
		for(Instance i:minority) {
			distance.add(calDistance(inst, i));
		}
		int[] flag = new int[distance.size()];
		for (int i = 0; i < SETTING.K+1; ++i) {

			double temp = Double.MAX_VALUE;
			int index = -1;
			for (int j = 0; j < distance.size(); ++j) {
				if(flag[j] == 0 && temp > distance.get(j)) {
					index = j;
					temp = distance.get(j);
					flag[j] = 1;
				}
			}
			if(index != -1) {
				result += temp;
			}
		}	
		return result/SETTING.K;
		
	}
	
	public static double calToMajDistance(Instance inst, Instances majority) {
		double result = 0.0f;
		for(Instance i:majority) {
			result += calDistance(i,inst);
		}
		return result;
	}
	public static double calCH_distance(Instances majority, Instances minority) {
		double result = 0;
		for(Instance min: minority) {
			for(Instance maj: majority) {
				result += calDistance(min,maj);
			}
		}
		result = result/minority.size();
		return result;
	}
	
	public static void main(String[] args) throws Exception {
		// TODO Auto-generated method stub
//		Instances data = LoadData.loadData("dataset/test.arff");
		SETTING.K = 3;
		String[] dataSets = {"yeast1"};
		String[] trainSet = Dataset.chooseDataset(dataSets[0], 0);
		String[] testSet = Dataset.chooseDataset(dataSets[0], 1);
		Instances data = LoadData.loadData(trainSet[1]);
		Instances test = LoadData.loadData(testSet[1]);
		
		Instances majoritySamples = new Instances(data);
		majoritySamples.clear();
		Instances minoritySamples = new Instances(data);
		minoritySamples.clear();
		int[] ans = new int[2];
		for (int i = 0; i < data.size(); ++i) {
			ans[(int) data.get(i).classValue()]++;
		}
		int flagclass = -1;
		if (ans[0] < ans[1]) {
			flagclass = 0;
		} else {
			flagclass = 1;
		}
		for (Instance inst : data) {
			if ((int) inst.classValue() == flagclass) {
				minoritySamples.add(inst);
			} else {
				majoritySamples.add(inst);
			}
		}
		for(Instance a: majoritySamples) {
			for(Instance b: minoritySamples) {
				System.out.println(String.format("%.3f", calDistance(a,b)));
			}
		}

	}

}
