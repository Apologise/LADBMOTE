package smote;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

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
import weka.classifiers.functions.SMO;
import weka.classifiers.meta.AdaBoostM1;
import weka.classifiers.meta.Vote;
import weka.classifiers.trees.J48;
import weka.core.Instance;
import weka.core.Instances;
//import weka.core.matrix.Matrix;
//import weka.core.matrix.SingularValueDecomposition;
import weka.core.SelectedTag;

public class MultiKernelTest {
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
					continue;
				} else {
					
					value = generateSample(minority, knnorigion, distance);
					value[alldata.numAttributes() - 1] = minority.get(0).classValue();
					output.get(i).add(minority.get(0).copy(value));
					IR--;
				}
			}
		}
	}
	public static List<Double> generateDistance(List<Double> distance, int flag) {

		for (int i = 0; i < distance.size(); ++i) {

			if ((1 - 0.5 * distance.get(i)) < 0) {
				flag = 1;
				return distance;
			}
			double temp = -2 * SETTING.theta * SETTING.theta * (Math.log(1 - 0.5 * distance.get(i)));
			distance.set(i, temp);
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

			if (value[i] < 0) {
				value[i] = -value[i];
			}

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
	//线性和算距离
	public static double linearkernelDistance(Instance a, Instance b) {
		double result = 0.0f;
		result = linear(a, a) - 2 * linear(a, b) + linear(b, b);
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
	
	
	//线性核
	public static double linear(Instance a, Instance b) {
			double temp = 0.0f;
			for(int i = 0; i < a.numAttributes()-1; ++i) {
				temp += a.value(i)*b.value(i);
			}
			temp += SETTING.linearc;
			return 0;
	}
	//

	public static void main(String[] args) throws Exception {
		// TODO Auto-generated method stub
		/*String[] dataSets = { "ecoli1", "ecoli2", "ecoli3", "ecoli4","ecoli0v1","ecoli0137v26", "yeast1", "yeast3", "yeast4", "yeast5", "yeast6",
				"glass0", "glass1", "glass2", "glass4", "glass5", "glass6", "glass016v2", "glass016v5",
				"glass0123vs456","newthyroid1", "newthyroid2",
				"pima" };
				*/
				
		String[] dataSets = {"ecoli1"};
		for(int dataset = 0; dataset < dataSets.length; ++dataset) {
		
	
		double[] AUC_Score = new double[5];
		System.out.println(dataSets[dataset]);
		String[] trainSet = Dataset.chooseDataset(dataSets[dataset], 0);
		String[] testSet = Dataset.chooseDataset(dataSets[dataset], 1);
		
		
		


	
					Instances data = LoadData.loadData(trainSet[0]);
					Instances test = LoadData.loadData(testSet[0]);
					
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
					
					//多核
					//1高斯核
					List<List<Instances>> allKernel = new ArrayList<List<Instances>>();
					for(int kernel = 0; kernel < 6; ++kernel) {
						List<Instances> Gausssysthetic = new ArrayList<Instances>();
						for (int i = 0; i < SETTING.K; ++i) {
							Instances temp = new Instances(data);
							temp.clear();
							Gausssysthetic.add(temp);
						}
						allKernel.add(Gausssysthetic);
					}
					//高斯核生成的K个数据集
					for (Instance inst : minoritySamples) {
						GuassKernel.generateOriginSample(minoritySamples, minoritySamples, inst, allKernel.get(0),
								n[minoritySamples.indexOf(inst)]);
						
					}
					for (Instances insts : allKernel.get(0)) {
						for (int i = 0; i < minoritySamples.size(); ++i) {
							insts.add(minoritySamples.get(i));
						}
						for (int i = 0; i < majoritySamples.size(); ++i) {
							insts.add(majoritySamples.get(i));
						}
					}
					for (Instance inst : minoritySamples) {
						EXPKernel.generateOriginSample(minoritySamples, minoritySamples, inst, allKernel.get(1),
								n[minoritySamples.indexOf(inst)]);
					}
					for (Instances insts : allKernel.get(1)) {
						for (int i = 0; i < minoritySamples.size(); ++i) {
							insts.add(minoritySamples.get(i));
						}
						for (int i = 0; i < majoritySamples.size(); ++i) {
							insts.add(majoritySamples.get(i));
						}
					}
					for (Instance inst : minoritySamples) {
						LaplaKernel.generateOriginSample(minoritySamples, minoritySamples, inst, allKernel.get(2),
								n[minoritySamples.indexOf(inst)]);
					}
					for (Instances insts : allKernel.get(2)) {
						for (int i = 0; i < minoritySamples.size(); ++i) {
							insts.add(minoritySamples.get(i));
						}
						for (int i = 0; i < majoritySamples.size(); ++i) {
							insts.add(majoritySamples.get(i));
						}
					}
					for (Instance inst : minoritySamples) {
						RadialKernel.generateOriginSample(minoritySamples, minoritySamples, inst, allKernel.get(3),
								n[minoritySamples.indexOf(inst)]);
					}
					for (Instances insts : allKernel.get(3)) {
						for (int i = 0; i < minoritySamples.size(); ++i) {
							insts.add(minoritySamples.get(i));
						}
						for (int i = 0; i < majoritySamples.size(); ++i) {
							insts.add(majoritySamples.get(i));
						}
					}
					for (Instance inst : minoritySamples) {
						MultiTwoKernel.generateOriginSample(minoritySamples, minoritySamples, inst, allKernel.get(4),
								n[minoritySamples.indexOf(inst)]);
					}
					for (Instances insts : allKernel.get(4)) {
						for (int i = 0; i < minoritySamples.size(); ++i) {
							insts.add(minoritySamples.get(i));
						}
						for (int i = 0; i < majoritySamples.size(); ++i) {
							insts.add(majoritySamples.get(i));
						}
					}
					for (Instance inst : minoritySamples) {
						InverseMultiTwoKernel.generateOriginSample(minoritySamples, minoritySamples, inst, allKernel.get(5),
								n[minoritySamples.indexOf(inst)]);
					}
					for (Instances insts : allKernel.get(5)) {
						for (int i = 0; i < minoritySamples.size(); ++i) {
							insts.add(minoritySamples.get(i));
						}
						for (int i = 0; i < majoritySamples.size(); ++i) {
							insts.add(majoritySamples.get(i));
						}
					}
					
					double[][] bestkernel = new double[6][SETTING.K];
					for(int i = 0; i < 6; ++i) {
						for(int j = 0; j < SETTING.K; ++j) {
							Classifier J45 = new J48();
							J45.buildClassifier(allKernel.get(i).get(j));
				//			System.out.println("大小："+minoritySamples.size()+" "+majoritySamples.size() +" "+ allKernel.get(i).get(j).size());
							Evaluation eval45 = new Evaluation(allKernel.get(i).get(j));
							eval45.evaluateModel(J45, test);
							bestkernel[i][j] = eval45.areaUnderROC(flagclass);
							System.out.print(bestkernel[i][j]+ " ");
						}
						System.out.println();
					}
					
					
					//从二维矩阵中选出每列最大值得最佳序号
					int[] bestkernelIndex = new int[SETTING.K];
					for(int i = 0; i < SETTING.K; ++i) {
						double temp = 0.0f;
						int tempIndex = -1;
						for(int j = 0; j < 6; ++j) {
							
							if(bestkernel[j][i] > temp) {
								temp = bestkernel[j][i];
								tempIndex = j;
							}
						}
						bestkernelIndex[i] = tempIndex;
	
					}
					
					
					Classifier[] classifier = new Classifier[SETTING.K];
					 String[] options =" -P 100 -S 1 -I 10 -W J48 -- -C 0.25 -M 2".split(" ");
					for (int i = 0; i < SETTING.K; ++i) {
						// classifier[i] = new AdaBoostM1();
						classifier[i] = new J48();
					//	 classifier[i].setOptions(options);
						classifier[i].buildClassifier(allKernel.get(bestkernelIndex[i]).get(i));
					}

					Vote ensemble = new Vote();
					SelectedTag tag = new SelectedTag(Vote.AVERAGE_RULE, Vote.TAGS_RULES);
					ensemble.setCombinationRule(tag);
					ensemble.setClassifiers(classifier);

					// 测试

					Evaluation evalC45 = new Evaluation(test);
					evalC45.evaluateModel(ensemble, test);
					System.out.println("ROC:"+evalC45.areaUnderROC(flagclass));
		}
	}
								 

}
