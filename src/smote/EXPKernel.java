package smote;

import java.util.ArrayList;
import weka.filters.supervised.instance.*;
import java.util.List;
import java.util.Random;
import weka.filters.supervised.instance.ClassBalancer;
import javax.print.attribute.standard.PrinterLocation;
import javax.print.attribute.standard.PrinterMessageFromOperator;

import org.omg.CORBA.PUBLIC_MEMBER;
import org.ujmp.core.intmatrix.calculation.DiscretizeStandardBinning;
import org.ujmp.core.util.matrices.SystemEnvironmentMatrix;

//import org.ujmp.core.Matrix;
//import org.ujmp.core.bigdecimalmatrix.DenseBigDecimalMatrix;

import Jama.Matrix;
import Jama.QRDecomposition;
import Jama.SingularValueDecomposition;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.functions.SMO;
import weka.classifiers.meta.AdaBoostM1;
import weka.classifiers.meta.Vote;
import weka.classifiers.trees.J48;
import weka.core.Instance;
import weka.core.Instances;
//import weka.core.matrix.Matrix;
//import weka.core.matrix.SingularValueDecomposition;
import weka.core.SelectedTag;
import weka.filters.supervised.instance.SMOTE;

public class EXPKernel {
	public static void generateOriginSample(Instances alldata, Instances minority, Instance inst,
			List<Instances> output, int N) {

		List<Integer> knn = kNearestNeihbors(minority, inst);

		// 随机选取一个近邻
		for (int i = 0; i < output.size(); ++i) {
			int IR = N;

			while (IR != 0) {
				int flag = 0;
				double[] value = new double[alldata.numAttributes()];
				List<Double> distance = new ArrayList<>();
				double gap = Math.random();
				// 求生成点的近邻
				List<Integer> knnorigion = kernelNearestNeihbors(minority, inst, minority.get(knn.get(i)), gap,
						distance);
				distance = generateDistance(distance, flag);
				if (flag == 1) {
					
				} else {
					
					value = generateSample(minority, knnorigion, distance);
					if(isArrayExistNAN(value)) {

						
					}else {

					value[alldata.numAttributes() - 1] = minority.get(0).classValue();
					output.get(i).add(minority.get(0).copy(value));
					}
					IR--;
					
				}
			}
		}
	}
	public static void printlnArray(double[] x) {
		for(int i = 0; i< x.length; ++i)
			System.out.print(x[i]+",");
		System.out.println();
		
	}
	public static boolean isArrayExistNAN(double[] x) {
		for(int i = 0; i < x.length; ++i) {
			if(Double.isNaN(x[i])||Double.isInfinite(x[i])) {
				return true;
			}
		}
		return false;		
			
	}
	
	public static List<Double> generateDistance1(List<Double> distance, int flag){
		for (int i = 0; i < distance.size(); ++i) {

			if (1 - 0.5 * distance.get(i) < 0) {
				flag = 1;
				return distance;
			}
			double temp = -2 * SETTING.theta * SETTING.theta * (Math.log(1 - 0.5 *distance.get(i)));
		
			distance.set(i, temp*temp);
			// System.out.println("eemp:"+temp);
		}
		return distance;
	}
	public static List<Double> generateDistance(List<Double> distance, int flag) {

		for (int i = 0; i < distance.size(); ++i) {

			if ((1 - 0.5 * distance.get(i)) < 0) {
				flag = 1;
				return distance;
			}
			double temp = -2 * SETTING.theta * SETTING.theta * (Math.log(1 - 0.5 * distance.get(i)));
			distance.set(i, temp*temp);
			// System.out.println("eemp:"+temp);
		}
		return distance;
	}

	public static Matrix normalizeMatrix(double[][] matrix,int t){
		
		Matrix Mat_matrix = new Matrix(matrix);
		Matrix one_Vector = new Matrix(t,1,1.0);
		Matrix center_Matrix = Matrix.identity(t,t).minus(one_Vector.times(one_Vector.transpose()).times(1.0/t));
		//输出中心化矩阵
	
		Mat_matrix=Mat_matrix.times(center_Matrix);
		return Mat_matrix;
	}
	public static double[] generateSample(Instances alldata, List<Integer> knn, List<Double> distances) {
		// 将k近邻按照矩阵的方式排列
		int t = knn.size();

		double[][] _X = new double[alldata.numAttributes() - 1][knn.size()];
		
		for (int i = 0; i < knn.size(); ++i) {
			for (int j = 0; j < alldata.numAttributes() - 1; ++j) {
				_X[j][i] = alldata.get(knn.get(i)).value(j);
			}
		
		}

		double[] xmean = new double[alldata.numAttributes() - 1];
		for (int i = 0; i < alldata.numAttributes() - 1; ++i) {
			double result = 0.0f;
			for (int j = 0; j < knn.size(); ++j) {
				result += _X[i][j];
			}
			xmean[i] = result / (knn.size());
		//	System.out.print(xmean[i] + "||");
		}

		//System.out.println("\n====");
		//System.out.println("均值数据");
		//for (int i = 0; i < alldata.numAttributes() - 1; ++i) {
		//	System.out.print(xmean[i] + " ");
		//}
		//System.out.println();
		Matrix XH = normalizeMatrix(_X,t);
	//	XH.print(XH.getRowDimension(), XH.getColumnDimension());
		SingularValueDecomposition svd = new SingularValueDecomposition(XH);
		// XH.print(XH.getRowDimension(), XH.getColumnDimension());
		Matrix E1 = svd.getU();
		Matrix S = svd.getS();
		Matrix V = svd.getV();
		// System.out.println("===");
		// E1.print(E1.getRowDimension(), E1.getColumnDimension());
		// S.print(S.getRowDimension(), S.getColumnDimension());
		// V.print(V.getRowDimension(), V.getColumnDimension());
		// System.out.println("===");
		double[] d = new double[t];
		for (int i = 0; i < t; ++i) {
			d[i] = distances.get(knn.get(i));
		}
		double[] d0 = new double[t];
		Matrix SV = S.times(V);
		int row = SV.getRowDimension();
		int colSV = SV.getColumnDimension();

		for (int i = 0; i < t; ++i) {
			double temp = 0.0f;
			for (int j = 0; j < row; ++j) {
				temp += SV.get(j, i) * SV.get(j, i);
			}
			d0[i] = temp;
		}
		// 计算中间项
		double[] d01 = new double[t];
		for (int i = 0; i < t; ++i) {
			d01[i] = 0.5 * (d0[i] - d[i]);
		}

		for (int i = 0; i < S.getRowDimension(); ++i) {
			for (int j = 0; j < S.getColumnDimension(); ++j) {
				if (Math.abs(S.get(i, j)) < 0.000005) {
					S.set(i, j, 0);
				} else {
					S.set(i, j, 1.0 / S.get(i, j));
				}
			}
		}
		//S.print(S.getRowDimension(), S.getColumnDimension());
		// System.out.println("求逆");
		// S.print(S.getRowDimension(), S.getColumnDimension());
		Matrix vector = new Matrix(d01, 1);

		// Matrix I = Matrix.identity(S.getRowDimension(),S.getColumnDimension());
		// System.out.println();
		// I.print(I.getRowDimension(), I.getColumnDimension());
		// invS.print(invS.getRowDimension(), invS.getColumnDimension());
		Matrix temp = E1.times(S).times(V).times(vector.transpose());
		// System.out.println("test-temp");
		// temp.print(temp.getRowDimension(), temp.getColumnDimension());
		double[] value = new double[alldata.numAttributes()];
		for (int i = 0; i < alldata.numAttributes() - 1; ++i) {
			value[i] = temp.get(i, 0) + xmean[i];
			/*
			if (value[i] < 0) {
				value[i] = -value[i];
			}
			*/

		}
		return value;
	}

	// 计算出生成样本在核空间内K个近邻(在整个数据集中，而不是少数类)
	public static List<Integer> kernelNearestNeihbors(Instances alldata, Instance samplei, Instance samplej, double gap,
			List<Double> distances) {
		List<Integer> knn = new ArrayList<>();

		for (int i = 0; i < alldata.size(); ++i) {
			distances.add(kernelGenerateDistance1(alldata.get(i), samplei, samplej, gap));
		}
		int[] flag = new int[alldata.size()];

		for (int i = 0; i < SETTING.K2; ++i) {

			double temp = Double.MAX_VALUE;
			int index = -1;
			for (int j = 0; j < alldata.size(); ++j) {
				if (flag[j] == 0) {
					if (temp > distances.get(j)) {
						temp = distances.get(j);
						index = j;
					}
				}
			}
			flag[index] = 1;
			knn.add(index);
		}
		return knn;
	}


	public static double kernelGenerateDistance1(Instance inst, Instance i, Instance j, double gap) {
		double distance;

		distance =(gap-1)*(gap-1)*expkernel(i,i)+(2*gap-2*gap*gap)*expkernel(i,j)+gap*gap*expkernel(j,j)+expkernel(inst,inst)+
				2*(gap-1)*expkernel(inst,i)-2*gap*expkernel(inst,j);
		return distance;
	}

	// 计算核距离矩阵
	public static List<Integer> kNearestNeihbors(Instances inputData, Instance inst) {
		List<Integer> knn = new ArrayList<>();
		List<Double> distances = new ArrayList<>();
		for (int i = 0; i < inputData.size(); ++i) {
			distances.add(expkernelDistance(inst, inputData.get(i)));
		}
		int[] flag = new int[inputData.size()];

		flag[inputData.indexOf(inst)] = 1;
		for (int i = 0; i < SETTING.K; ++i) {

			double temp = Double.MAX_VALUE;
			int index = -1;
			for (int j = 0; j < inputData.size(); ++j) {

				if (flag[j] == 0) {
					if (temp > distances.get(j)) {
						temp = distances.get(j);
						index = j;
					}
				}
			}
			flag[index] = 1;
			knn.add(index);
		}
		return knn;
	}


	//线性和算距离
	public static double expkernelDistance(Instance a, Instance b) {
		double result = 0.0f;
		result = expkernel(a, a) - 2 * expkernel(a, b) + expkernel(b, b);
		return result;
	}

	
	
	
	
	//镜像基核函数
	public static double expkernel(Instance a, Instance b) {

		double temp = 0.0f;
		for (int i = 0; i < a.numAttributes() - 1; ++i) {
			temp += (a.value(i) - b.value(i)) * (a.value(i) - b.value(i));
		}
		temp = Math.exp(-Math.sqrt(temp) / (2.0 * SETTING.theta * SETTING.theta));
		return temp;
	}

	public static void main(String[] args) throws Exception {
		// TODO Auto-generated method stub
		/*String[] dataSets = { "ecoli1", "ecoli2", "ecoli3", "ecoli4","ecoli0v1","ecoli0137v26", "yeast1", "yeast3", "yeast4", "yeast5", "yeast6",
				"glass0", "glass1", "glass2", "glass4", "glass5", "glass6", "glass016v2", "glass016v5",
				"glass0123vs456","newthyroid1", "newthyroid2",
				"pima" };
		*/
		/*
		String[] dataSets = {"yeast2v8","yeast1458v7","yeast1v7",
				"shuttle0v4","pageblocks13v4","vehicle1",
				"shuttle2v5","poker9v7"};
		/*
		 * 
		 */
String[] dataSets = {
				
				"glass1","ecoli0v1","pima","glass0","yeast1","vehicle1",
				"glass0123vs456","ecoli1","newthyroid1","newthyroid2",
				
				"ecoli2","glass6","yeast3","ecoli3","glass016v2",
				"glass2","shuttle0v4","yeast1v7","glass4","ecoli4",
				"pageblocks13v4","glass016v5","yeast1458v7","glass5",
				"yeast2v8","yeast4","poker9v7","yeast5","yeast6","shuttle2v5"};
		for(int dataset = 0; dataset < dataSets.length; ++dataset) {
		
		double final_result[][] = new double[6][20];
		double[] AUC_Score = new double[5];
		System.out.println(dataSets[dataset]);
		String[] trainSet = Dataset.chooseDataset(dataSets[dataset], 0);
		String[] testSet = Dataset.chooseDataset(dataSets[dataset], 1);
		
		double thetapara[] = {0.005,0.01,0.05,0.1,0.5,1 };
		double result[] = new double[thetapara.length];

		for (int th = 0; th < thetapara.length; ++th) {
			SETTING.theta = thetapara[th];
			for (int cnt = 0; cnt < 20; ++cnt) {

				for (int fold = 0; fold < 5; ++fold) {
					Instances data = LoadData.loadData(trainSet[fold]);
					Instances test = LoadData.loadData(testSet[fold]);
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

					// println(ans[0] + " " + ans[1]);
					int[] n = new int[minoritySamples.size()];
					int generatesize = ans[1 - flagclass] - ans[flagclass];
					for (int i = 0; i < minoritySamples.size(); ++i) {
						n[i] = (int) Math.floor(generatesize / ans[flagclass]);
					}
					int flag = n[0];

					int reminder = generatesize - (int) Math.floor(generatesize / ans[flagclass]) * ans[flagclass];
					// println(minoritySamples.size());
					// println(reminder);

					// println("reminder:" + reminder);
					for (int i = 0; i < reminder;) {
						Random rand = new Random();
						int index = rand.nextInt(minoritySamples.size());
						if (n[index] == flag) {
							n[index]++;
							i++;
						} else {
						}

					}

					// println("总共需要生成" + count);
					/*
					 * for(int i = 0; i < minoritySamples.size(); ++i) { print(n[i]); }
					 */
					// 选取test数据，选取比例为正负类各10%

					List<Instances> systhetic = new ArrayList<Instances>();
					for (int i = 0; i < SETTING.K; ++i) {
						Instances temp = new Instances(data);
						temp.clear();
						systhetic.add(temp);
					}

					for (Instance inst : minoritySamples) {
						generateOriginSample(minoritySamples, minoritySamples, inst, systhetic,
								n[minoritySamples.indexOf(inst)]);
					}
					/*
					 * for(Instances insts: systhetic) { for(int j = 0; j < ans[0]*IR+ans[0]-ans[1];
					 * ++j) { Random random = new Random(); int rd = random.nextInt(insts.size());
					 * insts.remove(rd); } }
					 */
			

				
					
				
					// 将K个少数类的分类器加上少数类和多数类
					for (Instances insts : systhetic) {
						for (int i = 0; i < minoritySamples.size(); ++i) {
							insts.add(minoritySamples.get(i));
						}
						for (int i = 0; i < majoritySamples.size(); ++i) {
							insts.add(majoritySamples.get(i));
						}
					}

					// 使用集成规则来聚类
					// AdaBoostM1[] classifier = new AdaBoostM1[SETTING.K];
					Classifier[] classifier = new Classifier[SETTING.K];
					 String[] options =" -P 100 -S 1 -I 10 -W J48 -- -C 0.25 -M 2".split(" ");
					for (int i = 0; i < SETTING.K; ++i) {
					//	 classifier[i] = new AdaBoostM1();
						classifier[i] = new NaiveBayes();
					//	 classifier[i].setOptions(options);
						classifier[i].buildClassifier(systhetic.get(i));
					}

					Vote ensemble = new Vote();
					SelectedTag tag = new SelectedTag(Vote.AVERAGE_RULE, Vote.TAGS_RULES);
					ensemble.setCombinationRule(tag);
					ensemble.setClassifiers(classifier);

					// 测试

					Evaluation evalC45 = new Evaluation(test);
					evalC45.evaluateModel(ensemble, test);

					AUC_Score[fold] =  Math.sqrt(evalC45.truePositiveRate(flagclass)*evalC45.trueNegativeRate(flagclass));
				}
				double score = 0.0f;
				for (int i = 0; i < 5; ++i) {
					// println("score" + AUC_Score[i]);
					score += AUC_Score[i];
				}
				result[th] = score / 5;
				// 每个参数下的结果
			//	System.out.println(th + " "+cnt);
				final_result[th][cnt] = result[th];
			}
		}
		double average_result[] = new double[6];
		for (int i = 0; i < 6; ++i) {
			double temp = 0.0f;
			for (int j = 0; j < 20; ++j) {
				temp += final_result[i][j];
			}
			average_result[i] = temp / 20;
		}
		for (int i = 0; i < thetapara.length; ++i) {
			System.out.println(thetapara[i] + ":" + average_result[i]);
		}
	}
	}

}
