package smote;
import static util.Utils.*;

import java.io.FileWriter;
import java.math.BigInteger;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import javax.management.modelmbean.RequiredModelMBean;
import javax.print.DocFlavor.INPUT_STREAM;
import javax.print.attribute.standard.PrinterLocation;


import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.functions.SMO;
import weka.classifiers.meta.AdaBoostM1;
import weka.classifiers.meta.Vote;
import weka.classifiers.trees.J48;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SelectedTag;
import weka.core.neighboursearch.NearestNeighbourSearch;
import weka.core.pmml.jaxbbindings.NearestNeighborModel;
import weka.estimators.MahalanobisEstimator;

public class main {
	public static void main(String[] args) throws Exception {
	//	for(int l = 0; l < 20; ++l) {
		// TODO Auto-generated method stub
		String filePath = "5-fold-glass-0-1-6-vs-5/glass-0-1-6-vs-5-5-2tra.arff";
		String testpath = "5-fold-glass-0-1-6-vs-5/glass-0-1-6-vs-5-5-2tst.arff";
		Instances data = LoadData.loadData(filePath);
		dataClean dc = new dataClean();
//		dc.cleanData(data);
	
		Instances majoritySamples = new Instances(data);
		majoritySamples.clear();
		Instances minoritySamples = new Instances(data);
		minoritySamples.clear();
		int[] ans = new int[2];
		for(int i = 0; i < data.size(); ++i) {
				ans[(int)data.get(i).classValue()]++;
		}
		
		for(Instance inst: data) {
			if((int)inst.classValue() == 0) {
				minoritySamples.add(inst);
			}else {
				majoritySamples.add(inst);
			}
		}
		
		List<Instances> trainSample = new ArrayList<Instances>();
		List<Instances> testSample = new ArrayList<Instances>();
		int testsamples = (int)(data.size()*0.1);
		

	
		println(ans[0] + " "+ans[1]);
		int[] n = new int[minoritySamples.size()];
		int generatesize = ans[1]-ans[0];
		for(int i = 0; i < minoritySamples.size();++i) {
			n[i] = (int) Math.floor(generatesize/ans[0]);
		}
		int flag = n[0];
		
		int reminder = generatesize - (int)Math.floor(generatesize/ans[0])*ans[0];
		println(minoritySamples.size());
		println(reminder);
		
		println("reminder:"+reminder);
		for(int i = 0; i < reminder; ) {
			println(i);
			Random rand = new Random();
			int index = rand.nextInt(minoritySamples.size());
			if(n[index] == flag) {
				n[index]++;
				i++;
			}else {
			}
			
		}
		
		
		int count = 0;
		for(int i = 0; i < minoritySamples.size();++i) {
			count+=n[i];
		}
		println("总共需要生成"+count);
		/*
		for(int i = 0; i < minoritySamples.size(); ++i) {
			print(n[i]);
		}
		*/
		//选取test数据，选取比例为正负类各10%
		
		Instances test = new Instances(data);
		test.clear();
		
		

		List<Instances> systhetic = new ArrayList<Instances>();
		for(int i = 0; i < SETTING.K; ++i) {
			Instances temp = new Instances(data);
			temp.clear();
			systhetic.add(temp);
		}
		
		println("shaoshu:"+minoritySamples.size());
		for(Instance inst: minoritySamples) {
			GenerateSample.generateSample(inst,minoritySamples,systhetic,n[minoritySamples.indexOf(inst)]);
		}
	/*	for(Instances insts: systhetic) {
			for(int j = 0; j < ans[0]*IR+ans[0]-ans[1]; ++j) {
				Random random = new Random();
				int rd = random.nextInt(insts.size());
				insts.remove(rd);
			}
		}*/
		println(ans[0]+" "+ans[1]);
		for(int i = 0; i < SETTING.K; ++i) {
			println(systhetic.get(i).size());
		}
		//将K个少数类的分类器加上少数类和多数类
		for(Instances insts: systhetic) {
			for(int i = 0; i < minoritySamples.size(); ++i) {
				insts.add(minoritySamples.get(i));
			}
			for(int i = 0; i < majoritySamples.size(); ++i) {
				insts.add(majoritySamples.get(i));
			}
		}
	
		int class0 = 0, class1 = 0;
		for(int i = 0; i < systhetic.get(0).size(); ++i) {
			if((int)systhetic.get(0).get(i).classValue() == 1) {
				class0++;
			}else {
				class1++;
			}
		}
		println("test"+class1+" "+class0);
		//使用集成规则来聚类
		//使用朴素分类器
		Classifier[] classifier= new Classifier[SETTING.K];
		for(int i = 0; i < SETTING.K; ++i) {
			classifier[i] = new J48();
			classifier[i].buildClassifier(systhetic.get(i));
		}
	
		
		Vote ensemble = new Vote();
		SelectedTag tag = new SelectedTag(Vote.AVERAGE_RULE, Vote.TAGS_RULES);
		ensemble.setCombinationRule(tag);
		ensemble.setClassifiers(classifier);
		
		//测试
		test = LoadData.loadData("5-fold-glass-0-1-6-vs-5/glass-0-1-6-vs-5-5-2tst.arff");
	//	dc.cleanData(test);
		Evaluation evalC45 = new Evaluation(test);
		evalC45.evaluateModel(ensemble,test);
		println(evalC45.toSummaryString("\nResult\n\n", false));
		println(evalC45.areaUnderROC(0));
		FileWriter fw = new FileWriter("dataset/基分类器C45_ecoli3.dat", true);
	    
	    fw.write("\n==============\n");
	    fw.write("参数设定:\n");
	    fw.write("基分类器: C4.5\n");
		fw.write("K值:"+SETTING.K+"\n");
		fw.write("IR:"+ans[1]/ans[0]+"\n");
	    fw.write("AUC:"+evalC45.areaUnderROC(0)+"\n");
	    fw.write("\n==============\n");
	    fw.close(); 
	    
//		}
	}

}
