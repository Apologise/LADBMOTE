package smote;

import java.security.Guard;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.function.DoubleToLongFunction;

import static util.Utils.*;
import javax.print.attribute.standard.PrinterLocation;

import org.omg.CORBA.PUBLIC_MEMBER;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.functions.SMO;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.ht.GaussianConditionalSufficientStats;
import weka.core.Instance;
import weka.core.Instances;
import weka.knowledgeflow.DefaultCallbackNotifierDelegate;


public class GenerateSampleKernel {
	
	public static void generateSample(Instance inst,Instances inputData, List<Instances> output ,int N) {
		List<Integer> knn = kNeighbors(inputData, inst);
		for(int i = 0; i < output.size(); ++i) {
			int IR = N;
			while(IR != 0) {
				double[] values = new double[inputData.numAttributes()];
				for(int j = 0; j < inputData.numAttributes()-1; ++j) {
					double gap = Math.random();
					List<Integer> newnn = kNeighborskernel(inputData, inst, inputData.get(knn.get(i)),gap);
						
					double diff = inputData.get(knn.get(i)).value(j)-inst.value(j);
					values[j] = inst.value(j) + gap*diff;
				}
				values[inputData.numAttributes()-1] = inputData.get(0).classValue();
				
				output.get(i).add(inputData.get(0).copy(values));
			
				IR--;
			}
		}
	}
	public static Instances generateSample1(Instances inputData, int N) {
		
		Instances systhetic = new Instances(inputData);
		systhetic.clear();
		for(int i = 0; i < inputData.size(); ++i) {
			List<Integer> knn = kNeighbors(inputData, inputData.get(i));
			int IR = N;
			while(IR != 0) {
				double[] values = new double[inputData.numAttributes()];
				Random random = new Random();
				int index = random.nextInt(SETTING.K);
				for(int j = 0; j < inputData.numAttributes()-1; ++j) {
					double gap = Math.random();
					double diff = inputData.get(knn.get(index)).value(j)-inputData.get(i).value(j);
					values[j] = inputData.get(i).value(j) + gap*diff;
				}
				values[inputData.numAttributes()-1] = inputData.get(0).classValue();
				systhetic.add(inputData.get(0).copy(values));
				IR--;
			}
		}
		return systhetic;
	}
	
	public static List<Integer> kNeighbors(Instances inputData, Instance inst) {
		List<Integer> knn = new ArrayList<>();
		List<Double> distances = new ArrayList<>(); 
		for(int i = 0; i < inputData.size(); ++i) {
			distances.add(calDistance(inst, inputData.get(i)));
		}
		int[] flag = new int[inputData.size()];
		flag[inputData.indexOf(inst)] = 1;
		for(int i = 0; i < SETTING.K; ++i) {

			double temp = Double.MAX_VALUE;
			int index = -1;
			for(int j = 0; j < inputData.size(); ++j) {
			
				if(flag[j] == 0) {
					if(temp > distances.get(j)) {
						temp = distances.get(j);
						index = j;
					}
				}
			}
			flag[index] = 1;
			knn.add(index);
		}
		return  knn;
		
	}
	public static List<Integer> kNeighborskernel(Instances inputData, Instance a, Instance b, double delta) {
		List<Integer> knn = new ArrayList<>();
		List<Double> distances = new ArrayList<>(); 
		for(int i = 0; i < inputData.size(); ++i) {
			distances.add(calDistanceKernel(a,b,inputData.get(i),delta));
		}
		int[] flag = new int[inputData.size()];
	
	//	flag[inputData.indexOf()] = 1;
		for(int i = 0; i < SETTING.K; ++i) {

			double temp = Double.MAX_VALUE;
			int index = -1;
			for(int j = 0; j < inputData.size(); ++j) {
			
				if(flag[j] == 0) {
					if(temp > distances.get(j)) {
						temp = distances.get(j);
						index = j;
					}
				}
			}
			flag[index] = 1;
			knn.add(index);
		}
		return  knn;
		
	}
	public static List<Integer> kNeighborsfornew(Instances inputData){
		List<Integer> knn = new ArrayList<>();
		
		return knn;
	}
	public static double calDistance(Instance a, Instance b) {
		double distance = 0.0f;
		if(a == b) {
			return Double.MAX_VALUE;
		}
		for(int i = 0; i < a.numAttributes()-1; ++i) {
			distance += (a.value(i) - b.value(i))*(a.value(i) - b.value(i));
		}
		distance = Math.sqrt(distance);
		return distance;
	}
	
	public static double calDistanceKernel(Instance a, Instance b,Instance other, double delta) {
		double distance = 0.0f;
		distance += 1-2*(delta-1)*gauss(other, a)-2*delta*gauss(other,b)+
				(delta-1)*(delta-1)*gauss(a,a)+2*delta*(1-delta)*gauss(a,b);
		return distance;
	}
	public static double gauss(Instance a, Instance b) {
		double result;
		if(a == b) {
			result = 1;
		}else {
			double temp = 0.0f;
			for(int i = 0; i < a.numAttributes()-1; ++i) {
				temp += (a.value(i) - b.value(i))*(a.value(i) - b.value(i));
			}
			temp  = temp/(2*SETTING.theta*SETTING.theta);
			result = Math.exp(-temp);
		}
		return result;
	}
	
	public static void func(List<Integer> knn) {
		
	}
	public static void main(String[] args) throws Exception {
		// TODO Auto-generated method stub
		
		String filePath = "dataset/glass0.arff";
		Instances data = LoadData.loadData(filePath);
	
		Instances majoritySamples = new Instances(data);
		majoritySamples.clear();
		Instances minoritySamples = new Instances(data);
		minoritySamples.clear();
		int[] ans = new int[2];
		for(int i = 0; i < data.size(); ++i) {
				ans[(int)data.get(i).classValue()]++;
		
		}
		int IR = ans[1]/ans[0];
		println(ans[1]+" "+ans[0]);
		
		for(Instance inst: data) {
			if((int)inst.classValue() == 0) {
				minoritySamples.add(inst);
			}else {
				majoritySamples.add(inst);
			}
		}
		println(minoritySamples.size());
		println(IR);
		Instances generate = generateSample1(minoritySamples, 1);
		for(int i = 0; i < data.size();  ++i) {
			generate.add(data.get(i));
		}
		println(generate.size());
		Classifier classifier = new J48();
		Evaluation evaluation = new Evaluation(generate);
		evaluation.crossValidateModel(classifier, generate,10,new Random(1));
			println(evaluation.toSummaryString());
			println(evaluation.areaUnderROC(0));
		}

}
