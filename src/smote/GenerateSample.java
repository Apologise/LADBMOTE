package smote;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.FormatFlagsConversionMismatchException;
import java.util.List;
import java.util.Random;
import java.util.function.DoubleToLongFunction;

import static util.Utils.*;

import javax.print.attribute.Size2DSyntax;
import javax.print.attribute.standard.PrinterLocation;
import javax.security.auth.kerberos.DelegationPermission;

import org.omg.CORBA.PUBLIC_MEMBER;
import org.ujmp.core.util.matrices.SystemEnvironmentMatrix;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.functions.SMO;
import weka.classifiers.trees.J48;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.unsupervised.attribute.MakeIndicator;


public class GenerateSample {
	
	public static void generateSample(Instance inst,Instances inputData,Instances majority, List<Instances> output ,int N) {
		List<Integer> knn = calNeighborsWithDensity(inst, inputData, majority);
		for(int i = 0; i < output.size(); ++i) {
			int IR = N;
			while(IR != 0) {
				double[] values = new double[inputData.numAttributes()];
				for(int j = 0; j < inputData.numAttributes()-1; ++j) {
					
					double gap = Math.random();
					/*
					if(gap<0.2) {
						gap = 0.2;
					}else if(gap >0.8) {
						gap = 0.8;
					}
					*/
					double diff = inputData.get(knn.get(i)).value(j)-inst.value(j);
					values[j] = inst.value(j) + gap*diff;
				}
				values[inputData.numAttributes()-1] = inputData.get(0).classValue();
				output.get(i).add(inputData.get(0).copy(values));
				IR--;
			}
		}
	}
	
	public static double calGap(Instance a, Instance b, Instances majority) {
		double gap = 0.5;
		ArrayList<Instance> knn_majority = new ArrayList<>();
		//得到a与b之间的所有多数类样本
		double r = calDistance(a,b)/2;
		//求得中点
		double[] mean = new double[a.numAttributes()-1];
		for(int i = 0; i < a.numAttributes()-1; ++i) {
			mean[i] = (a.value(i)+b.value(i))/2.0;
		}
		for(Instance inst: majority) {
			double temp_distance = calDistance(mean, inst);
			if(temp_distance < r) {
				knn_majority.add(inst);
			}
		}
		if(knn_majority.size()!=0) {
			System.out.println("======="+knn_majority.size());
		}
		gap = calGapForError(a,b,knn_majority);
		return gap;
	}
	
	//计算a,b两点之间的密度
	public static int calDensity(Instance a, Instance b, Instances majority ) {
		int density = 0;
		//得到a与b之间的所有多数类样本
		double r = calDistance(a,b)/2;
		//求得中点
		double[] mean = new double[a.numAttributes()-1];
		for(int i = 0; i < a.numAttributes()-1; ++i) {
			mean[i] = (a.value(i)+b.value(i))/2.0;
		}
		for(int i = 0; i<majority.size(); ++i) {
			double temp_distance = calDistance(mean, majority.get(i));
			if(temp_distance < r) {
				density++;
			}
		}
		return density;
	}
	
	//根据两点之间的密度来选择近邻
	public static List<Integer> calNeighborsWithDensity(Instance inst, Instances minority, Instances majority){
		List<Integer> knn = new ArrayList<Integer>();
		Density[] den = new Density[minority.size()];
		//计算出所有点到inst之间的距离与密度
		int[] density = new int[minority.size()];
	///	System.out.println(minority.size());
		for(int ii = 0; ii < minority.size(); ++ii) {
			den[ii] = new Density();
			den[ii].density = calDensity(inst, minority.get(ii), majority);
			den[ii].distance = calDistance(inst, minority.get(ii));
			den[ii].index = ii;
	
		}
		//先对den数组进行density升序排序
		Arrays.sort(den, new Comparator<Density>() {
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
		Arrays.sort(den, new Comparator<Density>() {

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
		Arrays.sort(den,new Comparator<Density>() {

			@Override
			public int compare(Density o1, Density o2) {
				// TODO Auto-generated method stub
					double area1 = o1.density/(o1.distance*o1.distance*4+1);
					double area2 = o2.density/(o2.distance*o2.distance*4+1);
					if(area1 > area2 ) {
						return 1;
					}else if(area1 < area2 ) {
						return -1;
					}
				return 0;
			}	
		});

		for(int ii = 0,cnt=0; ii < den.length&& cnt < SETTING.K;++ii) {
			if(den[ii].density < 1e-5&& den[ii].distance < 1e-5) {

			}else {
				knn.add(den[ii].index);
				cnt++;
			}
		}
		/*
		System.out.print("[");
		for(int ii = 0; ii < knn.size(); ++ii) {
			if(ii== 0) {
				System.out.print(knn.get(ii));
			}else {
				System.out.print(","+knn.get(ii));
			}
		}
		System.out.print("],");
		*/
		return knn;
	}
	
	
	
	public static double calGapForError(Instance a, Instance b,ArrayList<Instance> knn_majority) {
		double gap=0.2;
		if(knn_majority.size()==0) {
			gap = 0.5;
			return gap;
		}
		for(double i = 0.2; i<=0.8; i=i+0.01) {
			double[] temp = new double[a.numAttributes()-1];
			for(int j = 0; j < temp.length; ++j) {
				temp[j] = a.value(j)+i*(b.value(j)-a.value(j));
			}
			double error = 0.0f;
			double error_max = 0.0f;
			for(Instance inst:knn_majority) {
				
				error += calDistance(temp,inst);
			}
			if(error > error_max) {
				error_max = error;
				gap = i;
			}
		}
		return gap;
	}
	/*
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
	*/
	
	//根据到终点的距离来算近邻
	public static List<Integer> kNeighbors(Instances inputData, Instances majority,Instance inst) {
		List<Integer> knn = new ArrayList<>();
		List<Double> distances = new ArrayList<>(); 
		for(int i = 0; i < inputData.size(); ++i) {
			double tempDistance = calDistance(inst, inputData.get(i))
					+calDistanceWithinR(inst, inputData.get(i), majority);
			distances.add(tempDistance);
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
	public static double calDistance(Instance a, Instance b) {
		double distance = 0.0f;
		
		for(int i = 0; i < a.numAttributes()-1; ++i) {
			distance += (a.value(i) - b.value(i))*(a.value(i) - b.value(i));
		}
		distance = Math.sqrt(distance);
		return distance;
	}
	
	public static double calDistance(double[] mean, Instance a) {
		double distance = 0.0f;
		for(int i = 0; i < a.numAttributes()-1; ++i) {
			double temp = a.value(i)-mean[i];
			distance += temp*temp;
		}
		distance = Math.sqrt(distance);
		return distance;
	}
	
	//判断两个样本是否相等
	public static boolean InstanceisEqual(Instance a, Instance b) {
		boolean flag = true;
		for(int i = 0; i < a.numAttributes()-1; ++i) {
			double eps = 1e-6;
			if(Math.abs(a.value(i)-b.value(i)) < eps) {
				
			}else {
				flag = false;
				break;
			}
		}
		return flag;
	}
	//计算在圆圈内的距离
	public static double calDistanceWithinR(Instance a, Instance b, Instances majority) {
		double result = 0.0f;
		double r = calDistance(a,b)/2;
		//求得中点
		double[] mean = new double[a.numAttributes()-1];
		for(int i = 0; i < a.numAttributes()-1; ++i) {
			mean[i] = (a.value(i)+b.value(i))/2.0;
		}
		for(Instance inst: majority) {
			double temp_distance = calDistance(mean, inst);
			if(temp_distance < r) {
				result = result +  temp_distance;
			}
		}
		return result;
	}
	
	public static void main(String[] args) throws Exception {
		// TODO Auto-generated method stub
		
		String filePath = "dataset/testfile.arff";
		Instances data = LoadData.loadData(filePath);
	
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
		
		for (Instance inst : minoritySamples) {
			calNeighborsWithDensity(inst, minoritySamples, majoritySamples);
		}
	
	}

}
