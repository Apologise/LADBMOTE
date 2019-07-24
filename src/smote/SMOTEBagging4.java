package smote;

import static util.Utils.println;

import java.io.FileWriter;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.function.DoubleToLongFunction;

import util.Utils;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.classifiers.functions.SMO;
import weka.classifiers.lazy.IBk;
import weka.classifiers.meta.AdaBoostM1;
import weka.classifiers.meta.Vote;
import weka.classifiers.rules.JRip;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.RandomForest;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SelectedTag;

public class SMOTEBagging4 {
	public static void generateSample1(Instance inst,Instances inputData, Instances output ,int N) {
		List<Integer> knn = kNeighbors(inputData, inst);
		
			int IR = N;
			int k = knn.size();
			while(IR != 0) {
				Random random = new Random();
				int nn = random.nextInt(k);
				double[] values = new double[inputData.numAttributes()];
				for(int j = 0; j < inputData.numAttributes()-1; ++j) {
					double gap = Math.random();
					double diff = inputData.get(knn.get(nn)).value(j)-inst.value(j);
					values[j] = inst.value(j) + gap*diff;
				}
				values[inputData.numAttributes()-1] = inputData.get(0).classValue();
				
				output.add(inputData.get(0).copy(values));
			
				IR--;
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
	public static void main(String[] args) throws Exception {
		// for(int l = 0; l < 20; ++l) {
		// TODO Auto-generated method stub
		// 选择数据集
		String[] dataSets = {
				"glass1","ecoli0v1","pima","glass0","yeast1","vehicle1",
				"glass0123vs456","ecoli1","newthyroid1","newthyroid2",
				"ecoli2","glass6","yeast3","ecoli3","glass016v2",
				"glass2","shuttle0v4","yeast1v7","glass4","ecoli4",
				"pageblocks13v4","glass016v5","yeast1458v7","glass5",
				"yeast2v8","yeast4","poker9v7","yeast5","yeast6","shuttle2v5"
				};

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

					
					
						List<Instances> synthetic = new ArrayList<>();
						for(int i = 0; i < 4; ++i) {
							Instances temp = new Instances(data);
							temp.clear();
							
							for (Instance inst : minoritySamples) {
								generateSample1(inst, minoritySamples, temp,n[minoritySamples.indexOf(inst)]);
							}
							for (int j = 0; j < minoritySamples.size(); ++j) {
								temp.add(minoritySamples.get(j));
							}
							for (int j = 0; j < majoritySamples.size(); ++j) {
								temp.add(majoritySamples.get(j));
							}
							synthetic.add(temp);
						}
						
					

					
					/*
					 * for(Instances insts: systhetic) { for(int j = 0; j < ans[0]*IR+ans[0]-ans[1];
					 * ++j) { Random random = new Random(); int rd = random.nextInt(insts.size());
					 * insts.remove(rd); } }
					 */

					// 将K个少数类的分类器加上少数类和多数类
					
						
					

					// 使用集成规则来聚类
						Classifier[] cls = new Classifier[4];
					for(int i = 0; i < 4; ++i) {
						cls[i] = new MultilayerPerceptron();
						cls[i].buildClassifier(synthetic.get(i));
					}
					
				
					// 测试6

					Vote ensemble = new Vote();
					SelectedTag tag = new SelectedTag(Vote.AVERAGE_RULE, Vote.TAGS_RULES);
					ensemble.setCombinationRule(tag);
					ensemble.setClassifiers(cls);
					Evaluation evalC45 = new Evaluation(data);
					evalC45.evaluateModel(ensemble, test);
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
					AUC_Score[fold] = evalC45.areaUnderROC(flagclass);

				}
				double score = 0.0f;
				for (int i = 0; i < 5; ++i) {
					// println("score" + AUC_Score[i]);
					score += AUC_Score[i];
				}
				AllAveragescore += score/5;
			}
			System.out.print(String.format("%.3f",AllAveragescore / 20)+",");
		}
	}

}
