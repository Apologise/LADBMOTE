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
import weka.classifiers.bayes.net.search.fixed.NaiveBayes;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.classifiers.functions.SMO;
import weka.classifiers.lazy.IBk;
import weka.classifiers.meta.Vote;
import weka.classifiers.rules.JRip;
import weka.classifiers.trees.J48;
import weka.core.Instance;
import weka.core.Instances;
//import weka.core.matrix.Matrix;
//import weka.core.matrix.SingularValueDecomposition;
import weka.core.SelectedTag;

public class MultiKernelMultiPara {
	public static void main(String[] args) throws Exception {
		// TODO Auto-generated method stub
		String[] dataSets = {
				"glass5"};
		 int[] para_K = {4};
		 for(int para = 0; para < para_K.length; ++para) {
			 SETTING.K = para_K[para];
			 SETTING.K2 = 4;
		
		for (int dataset = 0; dataset < dataSets.length; ++dataset) {
			double max_auc = 0, min_auc = 0x7fffffff;
			double[] AUC_Score = new double[5];

			String[] trainSet = Dataset.chooseDataset(dataSets[dataset], 0);
			String[] testSet = Dataset.chooseDataset(dataSets[dataset], 1);
			
			double thetapara[] = {0.005,0.01,0.05,0.1,0.5, 1 };
			double[] tempscore = new double[20];
			
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
					List<List<List<Instances>>> allParaKernel = new ArrayList<List<List<Instances>>>();
					for (int k = 0; k < thetapara.length; ++k) {
						SETTING.theta = thetapara[k];
						SETTING.c = thetapara[k];
						List<List<Instances>> allKernel = new ArrayList<List<Instances>>();
						for (int kernel = 0; kernel < 6; ++kernel) {
							List<Instances> Gausssysthetic = new ArrayList<Instances>();
							for (int i = 0; i < SETTING.K; ++i) {
								Instances temp = new Instances(data);
								temp.clear();
								Gausssysthetic.add(temp);
							}
							allKernel.add(Gausssysthetic);
						}
						// 高斯核生成的K个数据集
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
							MultiTwoKernel.generateOriginSample(minoritySamples, minoritySamples, inst,
									allKernel.get(4), n[minoritySamples.indexOf(inst)]);
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
							InverseMultiTwoKernel.generateOriginSample(minoritySamples, minoritySamples, inst,
									allKernel.get(5), n[minoritySamples.indexOf(inst)]);
						}
						for (Instances insts : allKernel.get(5)) {
							for (int i = 0; i < minoritySamples.size(); ++i) {
								insts.add(minoritySamples.get(i));
							}
							for (int i = 0; i < majoritySamples.size(); ++i) {
								insts.add(majoritySamples.get(i));
							}
						}
						allParaKernel.add(allKernel);

					}
					//寻找每个方块中最优的分类器
					double[][][] bestkernel = new double[thetapara.length][6][SETTING.K];
					for (int k = 0; k < thetapara.length; ++k) {
						for (int i = 0; i < 6; ++i) {
							for (int j = 0; j < SETTING.K; ++j) {

								Classifier C45 = new J48();
								C45.buildClassifier(allParaKernel.get(k).get(i).get(j));
						
								Evaluation eval45 = new Evaluation(allParaKernel.get(k).get(i).get(j));

								eval45.evaluateModel(C45, test);
								bestkernel[k][i][j] = eval45.areaUnderROC(flagclass);
							}
							// System.out.print(bestkernel[i][j]+ " ");
						}
						// System.out.println();
					}

					// 从二维矩阵中选出每列最大值得最佳序号
					int[] bestkernelIndexk = new int[SETTING.K];
					int[] bestkernelIndexj = new int[SETTING.K];
					for (int i = 0; i < SETTING.K; ++i) {
						double temp = 0.0f;
						int tempIndexk = -1;
						int tempIndexj = -1;
						for (int k = 0; k < thetapara.length; ++k) {
							for (int j = 0; j < 6; ++j) {

								if (bestkernel[k][j][i] > temp) {
									temp = bestkernel[k][j][i];
									tempIndexk = k;
									tempIndexj = j;
								}
							}
						}
						bestkernelIndexk[i] = tempIndexk;
						bestkernelIndexj[i] = tempIndexj;

					}
					/*
					 * for(Instances insts: systhetic) { for(int j = 0; j < ans[0]*IR+ans[0]-ans[1];
					 * ++j) { Random random = new Random(); int rd = random.nextInt(insts.size());
					 * insts.remove(rd); } }
					 */

					// 使用集成规则来聚类

					Classifier[] classifier = new Classifier[SETTING.K];
					String[] options = " -P 100 -S 1 -I 10 -W J48 -- -C 0.25 -M 2".split(" ");
					for (int i = 0; i < SETTING.K; ++i) {
						// classifier[i] = new AdaBoostM1();
						classifier[i] =new J48();
						// classifier[i].setOptions(options);
						classifier[i].buildClassifier(
								allParaKernel.get(bestkernelIndexk[i]).get(bestkernelIndexj[i]).get(i));
					}

					Vote ensemble = new Vote();
					SelectedTag tag = new SelectedTag(Vote.AVERAGE_RULE, Vote.TAGS_RULES);
					ensemble.setCombinationRule(tag);
					ensemble.setClassifiers(classifier);

					// 测试

					Evaluation evalC45 = new Evaluation(data);
					evalC45.evaluateModel(ensemble, test);

					AUC_Score[fold] = evalC45.areaUnderROC(flagclass);
				}
				double score = 0.0f;
				for (int i = 0; i < 5; ++i) {

					score += AUC_Score[i];
				}
				tempscore[cnt] = score / 5;
				if(tempscore[cnt] > max_auc) {
					max_auc = tempscore[cnt];
				}
				if(tempscore[cnt] < min_auc) {
					min_auc = tempscore[cnt];
				}

			}
			double finalscore = 0.0f;
			for(int i = 0; i < 20; ++i) {
				finalscore+= tempscore[i];
			}
			System.out.println(para_K[para]+"-"+dataSets[dataset]+": "+finalscore/20 + "min:"+min_auc+" max:"+max_auc);
			
		}

	}
	}

}







