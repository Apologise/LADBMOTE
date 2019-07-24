package smote;

import java.awt.Event;
import java.io.File;
import java.util.Random;

import javax.swing.plaf.basic.BasicInternalFrameTitlePane.SystemMenuBar;

import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;
import weka.core.converters.ArffLoader;
import weka.core.neighboursearch.kdtrees.KMeansInpiredMethod;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.classifiers.functions.SMO;
import weka.classifiers.lazy.IBk;
import weka.classifiers.rules.JRip;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.RandomForest;
import weka.clusterers.SimpleKMeans;


public class CBS {
	public static void main1(String[] args) throws Exception {
		// TODO Auto-generated method stub
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
					
					//对多数类进行聚类
					SimpleKMeans kMeans = new SimpleKMeans();
					//设置聚类个数为少数类的样本数
					Remove remove = new Remove();
					String[] options = Utils.splitOptions("-R "+majoritySamples.numAttributes());
					remove.setOptions(options);
					remove.setInputFormat(majoritySamples);
					 Instances clusterdata = Filter.useFilter(majoritySamples, remove);
					 
					kMeans.setNumClusters(minoritySamples.size());
					kMeans.buildClusterer(clusterdata);
					Instances centers = kMeans.getClusterCentroids();
					centers.insertAttributeAt(majoritySamples.attribute(majoritySamples.numAttributes()-1),centers.numAttributes());
					centers.setClassIndex(centers.numAttributes()-1);
					//将产生的中心点作为多数类
					Instances systhetic = new Instances(test);
					systhetic.clear();
					//将少数类和中心点加入到systhetic中
					for(Instance inst: minoritySamples) {
						systhetic.add(inst);
					}
					for(Instance inst: centers) {
						inst.setClassValue(1-flagclass);
						systhetic.add(inst);
					}

					//建立分类器，测试分类效果
					Classifier j48 = new J48();
					j48.buildClassifier(systhetic);
					Evaluation eval = new Evaluation(systhetic);
					eval.evaluateModel(j48, test);
					AUC_Score[fold] = eval.areaUnderROC(flagclass);
				}
				double finalscore = 0.0f;
				for(int i = 0; i < AUC_Score.length; ++i) {
					finalscore += AUC_Score[i];
				}
				AllAveragescore += finalscore/5;
			}
			System.out.print(AllAveragescore/20+",");
		}
	}
	public static void main(String[] args) throws Exception {
		// TODO Auto-generated method stub
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
					
					
					//对多数类进行聚类
					SimpleKMeans kMeans = new SimpleKMeans();
					//设置聚类个数为少数类的样本数
					Remove remove = new Remove();
					String[] options = Utils.splitOptions("-R "+majoritySamples.numAttributes());
					remove.setOptions(options);
					remove.setInputFormat(majoritySamples);
					 Instances clusterdata = Filter.useFilter(majoritySamples, remove);
					 
					kMeans.setNumClusters(minoritySamples.size());
					kMeans.buildClusterer(clusterdata);
					Instances centers = kMeans.getClusterCentroids();
					centers.insertAttributeAt(majoritySamples.attribute(majoritySamples.numAttributes()-1),centers.numAttributes());
					Instances centersNN = new Instances(centers);
					centersNN.clear();
					//找到每个中心点的最近的样本
					for(Instance inst:centers) {
						int index = getNearestInstance(inst, majoritySamples);
						centersNN.add(majoritySamples.get(index));
					
					}
					centersNN.setClassIndex(centersNN.numAttributes()-1);
					//将产生的中心点作为多数类
					Instances systhetic = new Instances(test);
					systhetic.clear();
					//将少数类和中心点加入到systhetic中
					for(Instance inst: minoritySamples) {
						systhetic.add(inst);
					}
					for(Instance inst: centersNN) {
						inst.setClassValue(1-flagclass);
						systhetic.add(inst);
					}
					//建立分类器，测试分类效果
					Classifier j48 = new MultilayerPerceptron();
					j48.buildClassifier(systhetic);
					Evaluation eval = new Evaluation(systhetic);
					eval.evaluateModel(j48, test);
					AUC_Score[fold] = eval.areaUnderROC(flagclass);
				}
				double finalscore = 0.0f;
				for(int i = 0; i < AUC_Score.length; ++i) {
					finalscore += AUC_Score[i];
				}
				AllAveragescore += finalscore/5;
			}
			System.out.print(String.format("%.3f",AllAveragescore / 20)+",");
		}
	}
	
	//选择每个中心点最近的样本(多数类样本)
	public static int getNearestInstance(Instance inst, Instances input) {
		int index = -1;
		double distance = Double.MAX_VALUE;
		for(int i = 0; i < input.size(); ++i) {
			double temp = getdistance(inst, input.get(i));
			if(temp < distance) {
				index = i;
				distance = temp;
			}
		}
		return index;
	}
	public static double getdistance(Instance a, Instance b) {
		double distance = 0.0f;
		if(a == b) {return Double.MAX_VALUE;};
		for(int i = 0; i < a.numAttributes()-1; ++i) {
			distance += (a.value(i)-b.value(i))*(a.value(i)-b.value(i));
		}
		return distance;
	}
}
