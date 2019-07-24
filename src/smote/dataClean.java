package smote;

import java.util.ArrayList;
import java.util.List;
import static util.Utils.*;

import javax.management.InstanceAlreadyExistsException;
import javax.print.attribute.standard.PrinterLocation;

import weka.core.Instance;
import weka.core.Instances;

public class dataClean {
	
	public void cleanData(Instances inputData) {
		int count = 0;
		for(Instance i:inputData) {
			if(numOfKnn(i,inputData) == 0) {
				count++;
				inputData.remove(i);
			}
		}
	}
	
	public boolean isNoiseSample(Instance inst) {
		return false;
		
	}
	public int numOfKnn(Instance inst, Instances inputData) {
		int ans = 0;
		List<Double> distances = new ArrayList<>();
		int[] flag = new int[inputData.size()];
		for(int i = 0; i < inputData.size(); ++i) {
			flag[i] = 0;
		}
		flag[inputData.indexOf(inst)] = 1;
		for(Instance i: inputData) {
			distances.add(calDistance(inst, i));
//			print(calDistance(inst, i)+" ");
		}
	//	println("");
		for(int i = 0; i < SETTING.K; ++i) {
			double temp = Double.MAX_VALUE;
			int index = -1;
			for(int j = 0; j < distances.size(); ++j) {
				if(flag[j] == 0) {
					if(temp >= distances.get(j)) {
						temp = distances.get(j);
						index = j;
					}
				}
			}
			for(int j = 0; j < distances.size(); ++j) {
				if(temp == distances.get(j)) {
					flag[j] = 1;
//					print(j+" ");
					if((int)(inputData.get(j).classValue()-inst.classValue()) == 0) {
						ans++;
					}
				}
			}
		}
		return ans;
	}
	public double calDistance(Instance a, Instance b) {
		double distance = 0.0f;
		if(a == b) {
			return 0x7fffffff;
		}
		for(int i = 0; i < a.numAttributes()-1; ++i) {
			distance += (a.value(i) - b.value(i))*(a.value(i) - b.value(i));
		}
		distance = Math.sqrt(distance);
		return distance;
	}
	public static void main(String[] args) throws Exception {
		// TODO Auto-generated method stub
		String filePath = "dataset/ecoli1.arff";
		Instances inputData = LoadData.loadData(filePath);
		for(Instance inst:inputData) {
			System.out.println(inst.toString());
		}

	}

}
