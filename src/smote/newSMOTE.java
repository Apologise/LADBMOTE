package smote;

import static util.Utils.println;

import java.util.Random;

import weka.classifiers.Evaluation;
import weka.classifiers.trees.J48;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.supervised.instance.SMOTE;

public class newSMOTE {

	public static void main(String[] args) throws Exception {
		// TODO Auto-generated method stub
		String[] dataSets = {"pima"};
		String[] trainSet = Dataset.chooseDataset(dataSets[0], 0);
		String[] testSet = Dataset.chooseDataset(dataSets[0], 1);
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
			
				weka.filters.supervised.instance.SMOTE sm = new weka.filters.supervised.instance.SMOTE(); 
				double percentage = sm.getPercentage();
				sm.setNearestNeighbors(5);
				sm.setPercentage(percentage);
				sm.setRandomSeed((int)(Math.random()*10));
				sm.setInputFormat(data);
				Instances ndata = weka.filters.Filter.useFilter(data, sm);
				System.out.println(ndata.numInstances());
				
					
					
					
				

				

				// 将K个少数类的分类器加上少数类和多数类
				
				

				// 使用集成规则来聚类
				
				J48 cls = new J48();
				cls.buildClassifier(ndata);
			
				// 测试

				Evaluation evalC45 = new Evaluation(ndata);
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
				AUC_Score[fold] = evalC45.areaUnderROC(flagclass);

			}
			double score = 0.0f;
			for (int i = 0; i < 5; ++i) {
				// println("score" + AUC_Score[i]);
				score += AUC_Score[i];
			}
			AllAveragescore += score/5;
		}
		println(dataSets[0]+": " +AllAveragescore / 20);
	
	}


}
