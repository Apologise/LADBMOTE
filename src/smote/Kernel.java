package smote;

import weka.core.Instance;
import weka.core.Instances;
import weka.core.neighboursearch.balltrees.MedianOfWidestDimension;
import java.util.ArrayList;
import java.util.List;

import org.ujmp.core.importer.MatrixImporter;

import Jama.EigenvalueDecomposition;
import Jama.Matrix;
import Jama.SingularValueDecomposition;

public class Kernel {
	//计算输入空间的原始距离矩阵
	public  static List<ArrayList<Double>> calMatrixK(Instances input){
		List<ArrayList<Double>> kMatrix = new  ArrayList<ArrayList<Double>>();
		for(Instance inst_1: input) {
			ArrayList<Double> distance = new ArrayList<Double>();
			for(Instance inst_2: input) {
				distance.add(calDistance(inst_1,inst_2));
			}
			kMatrix.add(distance);
		}
		return kMatrix;
	}
	//计算两个点在输入空间之间的距离
	public static double calDistance(Instance a, Instance b) {
		double distance = 0.0f;
		for(int i = 0; i < a.numAttributes()-1; ++i) {
			double gap = a.value(i)-b.value(i);
			distance += gap*gap;
		}
		return Math.sqrt(distance);
	}
	public static void printMatrix(List<ArrayList<Double>> Matrix) {
		for(int i = 0; i < Matrix.size(); ++i) {
			for(int j = 0; j < Matrix.get(i).size(); ++j) {
				System.out.print(Matrix.get(i).get(j)+" ");
			}
			System.out.println();
		}
	}
	public static double[][] listToArray(List<ArrayList<Double>> Matrix) {
		//获取矩阵的行与列,然后将list类型的数组转为array类型
		int row = Matrix.size(),col = Matrix.get(0).size();
		double[][] array_matrix = new double[row][col]; 
		for(int i = 0; i < row; ++i) {
			for(int j = 0; j < col; ++j) {
				array_matrix[i][j] = Matrix.get(i).get(j);
			}
		}
		return array_matrix;
	}
	public static Matrix normalizeMatrix(double[][] matrix){
		int col = matrix.length;
		Matrix Mat_matrix = new Matrix(matrix);
		Matrix one_Vector = new Matrix(col,1,1.0);
		Matrix center_Matrix = Matrix.identity(col,col).minus(one_Vector.times(one_Vector.transpose()).times(1.0/col));
		//输出中心化矩阵
		Mat_matrix = center_Matrix.times(Mat_matrix).times(Mat_matrix);
		return Mat_matrix;
	}
	//对已经进行均质化的样本进行特征分解
	public static void EigDecomposition(Matrix mat) {
		EigenvalueDecomposition eig = new EigenvalueDecomposition(mat);
		
	}
	//计算生成的样本在核空间的K2个近邻
	public static void main(String[] args) throws Exception {
		// TODO Auto-generated method stub
		String[] dataSets = { "ecoli1", "ecoli2", "ecoli3", "ecoli4","ecoli0v1","ecoli0137v26", "yeast1", "yeast3", "yeast4", "yeast5", "yeast6",
				"glass0", "glass1", "glass2", "glass4", "glass5", "glass6", "glass016v2", "glass016v5",
				"glass0123vs456","newthyroid1", "newthyroid2",
				"pima" };
		String[] trainSet = Dataset.chooseDataset(dataSets[1], 0);
		String[] testSet = Dataset.chooseDataset(dataSets[1], 1);
		Instances inputdata = LoadData.loadData("dataset/iris.2D.arff");
		List<ArrayList<Double>> kMatrix =  calMatrixK(inputdata);
//		printMatrix(kMatrix);
		double[][] test = {{5,15},{3,6}};
		Matrix test_1 = normalizeMatrix(test);
		
		
		
	}

}
