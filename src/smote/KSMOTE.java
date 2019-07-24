package smote;

import static util.Utils.println;
import weka.core.Utils;

import java.io.FileWriter;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.function.DoubleToLongFunction;

import Jama.Matrix;
import Jama.SingularValueDecomposition;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.functions.SMO;
import weka.classifiers.meta.AdaBoostM1;
import weka.classifiers.meta.Vote;
import weka.classifiers.rules.JRip;
import weka.classifiers.trees.J48;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SelectedTag;
import weka.classifiers.functions.supportVector.Puk;
import weka.classifiers.lazy.IBk;

public class KSMOTE {
	
	public static void generateOriginSample( Instance inst,Instances minority, 
			Instances output, int N) {

		List<Integer> knn = kNearestNeihbors(minority, inst);

		// 随机选取一个近邻
		
			int IR = N;
		
			int k = knn.size();
			while (IR != 0) {
				int flag = 0;
				Random random = new Random();
				int nn = random.nextInt(k);
				double[] value = new double[minority.numAttributes()];
				List<Double> distance = new ArrayList<>();
				double gap = Math.random();
				// 求生成点的近邻
				List<Integer> knnorigion = kernelNearestNeihbors(minority, inst, minority.get(knn.get(nn)), gap,
						distance);
				distance = generateDistance1(distance, flag);
				if (flag == 1) {
					continue;
				} else {
					
					value = generateSample(minority, knnorigion, distance);
					value[minority.numAttributes() - 1] = minority.get(0).classValue();
					output.add(minority.get(0).copy(value));
					IR--;
				}
			}
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

	public static double[] generateSample(Instances alldata, List<Integer> knn, List<Double> distances) {
		// 将k近邻按照矩阵的方式排列
		int t = knn.size();

		double[][] _X = new double[knn.size()][alldata.numAttributes() - 1];
		
		for (int i = 0; i < knn.size(); ++i) {
			for (int j = 0; j < alldata.numAttributes() - 1; ++j) {
				_X[i][j] = alldata.get(knn.get(i)).value(j);

			}
			//System.out.println(alldata.get(knn.get(i)) + " ||");
		}

		double[] xmean = new double[alldata.numAttributes() - 1];
		for (int i = 0; i < alldata.numAttributes() - 1; ++i) {
			double result = 0.0f;
			for (int j = 0; j < knn.size(); ++j) {
				result += _X[j][i];
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
		Matrix X = new Matrix(_X);
		X = X.transpose();
		double[][] _L = new double[t][t];
		for (int i = 0; i < t; ++i) {
			for (int j = 0; j < t; ++j) {
				if (i == j) {
					_L[i][j] = 1 - 1.0 / t;
				} else {
					_L[i][j] = -1.0 / t;
				}
			}
		}
		Matrix L = new Matrix(_L);

		Matrix XH = X.times(L);
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
				if (Math.abs(S.get(i, j)) < 0.0005) {
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

	public static double kernelGenerateDistance(Instance inst, Instance samplei, Instance samplej, double gap) {
		double distance;
		distance = (1 + 2 * gap) * gauss(inst, inst) - 2 * gauss(inst, samplei) - 2 * gap * gauss(inst, samplej)
				+ (gap - 1) * (gap - 1) * gauss(samplei, samplei) + 2 * gap * (1 - gap) * gauss(samplei, samplej)
				+ gap * gap * gauss(samplej, samplej);
		return distance;
	}

	public static double kernelGenerateDistance1(Instance inst, Instance i, Instance j, double gap) {
		double distance;

		distance = 1 - 2 * (gap - 1) * gauss(inst, i) - 2 * gap * gauss(inst, j) + (gap - 1) * (gap - 1) * gauss(i, i)
				+ 2 * gap * (1 - gap) * gauss(i, j) + gap * gap * gauss(j, i);
		return distance;
	}

	// 计算核距离矩阵
	public static List<Integer> kNearestNeihbors(Instances inputData, Instance inst) {
		List<Integer> knn = new ArrayList<>();
		List<Double> distances = new ArrayList<>();
		for (int i = 0; i < inputData.size(); ++i) {
			distances.add(kernelDistance(inst, inputData.get(i)));
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

	public static double kernelDistance(Instance a, Instance b) {
		double result = 0.0f;
		result = gauss(a, a) - 2 * gauss(a, b) + gauss(b, b);
		return result;
	}


	//高斯核
	public static double gauss(Instance a, Instance b) {

		double temp = 0.0f;
		for (int i = 0; i < a.numAttributes() - 1; ++i) {
			temp += (a.value(i) - b.value(i)) * (a.value(i) - b.value(i));
		}
		temp = Math.exp(-temp / (2.0 * SETTING.theta * SETTING.theta));
		
		return temp;
	}
	public static void main(String[] args) throws Exception {
		// for(int l = 0; l < 20; ++l) {
		// TODO Auto-generated method stub
		// 选择数据集
		String[] dataSets = {
				"yeast2v8","yeast1458v7","yeast1v7",
				"shuttle0v4","pageblocks13v4","vehicle1",
				"shuttle2v5","poker9v7"};
		SETTING.theta = 0.05;
		for (int set = 0; set < dataSets.length; ++set) {
		
			String[] trainSet = Dataset.chooseDataset(dataSets[set], 0);
			String[] testSet = Dataset.chooseDataset(dataSets[set], 1);
			double AllAveragescore = 0.0f;
			for (int cnt = 0; cnt < 20; ++cnt) {
				double[] AUC_Score = new double[5];
				for (int fold = 0; fold < 5; ++fold) {
					// 载入测试数据和训练数据
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

					int count = 0;
					for (int i = 0; i < minoritySamples.size(); ++i) {
						count += n[i];
					}
					// println("总共需要生成" + count);
					/*
					 * for(int i = 0; i < minoritySamples.size(); ++i) { print(n[i]); }
					 */
					// 选取test数据，选取比例为正负类各10%

					
					
						Instances systhetic = new Instances(data);
						systhetic.clear();
						
					

					for (Instance inst : minoritySamples) {
						generateOriginSample(inst, minoritySamples, systhetic,n[minoritySamples.indexOf(inst)]);
					}
					/*
					 * for(Instances insts: systhetic) { for(int j = 0; j < ans[0]*IR+ans[0]-ans[1];
					 * ++j) { Random random = new Random(); int rd = random.nextInt(insts.size());
					 * insts.remove(rd); } }
					 */

					// 将K个少数类的分类器加上少数类和多数类
					
						for (int i = 0; i < minoritySamples.size(); ++i) {
							systhetic.add(minoritySamples.get(i));
						}
						for (int i = 0; i < majoritySamples.size(); ++i) {
							systhetic.add(majoritySamples.get(i));
						}
					

					// 使用集成规则来聚类
						
					J48 cls = new J48();
					//cls.setOptions(Utils.splitOptions("-C 1.0 -L 0.0010 -P 1.0E-12 -N 0 -V -1 -W 1 -K \"weka.classifiers.functions.supportVector.RBFKernel -C 250007 -G 0.01\""));
					cls.buildClassifier(systhetic);
					
					// 测试

					Evaluation evalC45 = new Evaluation(data);
					evalC45.evaluateModel(cls,test);
					/*
					 * println(evalC45.toSummaryString("\nResult\n\n", false));
					 * println(evalC45.areaUnderROC(0)); FileWriter fw = new
					 * FileWriter("dataset/基分类器C45_ecoli3.dat", true);
					 * 
					 * fw.write("\n==============\n"); fw.write("参数设定:\n");
					 * fw.write("基分类器: C4.5\n"); fw.write("K值:"+SETTING.K+"\n");
					 * fw.write("IR:"+ans[1]/ans[0]+"\n");
					 * fw.write("AUC:"+evalC45.areaUnderROC(0)+"\n");
					 * fw.write("\n==============\n"); fw.close();
					 */
					AUC_Score[fold] =evalC45.areaUnderROC(flagclass);
							//Math.sqrt(evalC45.truePositiveRate(flagclass)*evalC45.trueNegativeRate(flagclass));

				}
				double score = 0.0f;
				for (int i = 0; i < 5; ++i) {
					// println("score" + AUC_Score[i]);
					score += AUC_Score[i];
				}
				AllAveragescore += score/5;
			}
			println(dataSets[set] +": " +AllAveragescore / 20);
		}
	}

}
