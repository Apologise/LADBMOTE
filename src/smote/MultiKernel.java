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

public class MultiKernel {
	public static void main(String[] args) throws Exception {
		// TODO Auto-generated method stub
		String[] dataSets = { "ecoli1", "ecoli2", "ecoli3", "ecoli4","ecoli0v1","ecoli0137v26", "yeast1", "yeast3", "yeast4", "yeast5", "yeast6",
				"glass0", "glass1", "glass2", "glass4", "glass5", "glass6", "glass016v2", "glass016v5",
				"glass0123vs456","newthyroid1", "newthyroid2",
				"pima" };
				
			
		//String[] dataSets = {"ecoli1"};
		for(int dataset = 0; dataset < dataSets.length; ++dataset) {
		
		double final_result[][] = new double[6][20];
		double[] AUC_Score = new double[5];
		System.out.println(dataSets[dataset]);
		String[] trainSet = Dataset.chooseDataset(dataSets[dataset], 0);
		String[] testSet = Dataset.chooseDataset(dataSets[dataset], 1);
		
		double thetapara[] = { 0.005, 0.01, 0.05, 0.1, 0.5, 1 };
		double result[] = new double[thetapara.length];

		for (int th = 0; th < thetapara.length; ++th) {
			SETTING.theta = thetapara[th];
			SETTING.c = thetapara[th];
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
							Classifier J45 = new SMO();
							J45.buildClassifier(allKernel.get(i).get(j));
				//			System.out.println("大小："+minoritySamples.size()+" "+majoritySamples.size() +" "+ allKernel.get(i).get(j).size());
							Evaluation eval45 = new Evaluation(allKernel.get(i).get(j));
							eval45.evaluateModel(J45, test);
							bestkernel[i][j] = eval45.areaUnderROC(flagclass);
						//	System.out.print(bestkernel[i][j]+ " ");
						}
				//		System.out.println();
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
					/*
					 * for(Instances insts: systhetic) { for(int j = 0; j < ans[0]*IR+ans[0]-ans[1];
					 * ++j) { Random random = new Random(); int rd = random.nextInt(insts.size());
					 * insts.remove(rd); } }
					 */
			

				
					
				
		
					
					// 使用集成规则来聚类
					
					 
					Classifier[] classifier = new Classifier[SETTING.K];
					 String[] options =" -P 100 -S 1 -I 10 -W J48 -- -C 0.25 -M 2".split(" ");
					for (int i = 0; i < SETTING.K; ++i) {
						// classifier[i] = new AdaBoostM1();
						classifier[i] = new SMO();
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

					AUC_Score[fold] = evalC45.areaUnderROC(flagclass);
				}
				double score = 0.0f;
				for (int i = 0; i < 5; ++i) {
					
					score += AUC_Score[i];
				}
				result[th] = score / 5;
				// 每个参数下的结果
		
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
