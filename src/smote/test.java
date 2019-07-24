package smote;



import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class test {

	public static void main(String[] args) throws Exception {
		// TODO Auto-generated method stub
		Instances data = DataSource.read("dataset/iris.2D.arff");
		for(Instance inst: data) {
			System.out.println("1:"+inst.toString());
			System.out.println("2:"+data.get(getNearestInstance(inst, data)).toString());
			
		}
	}
	//选择每个中心点最近的样本(多数类样本)
		public static int getNearestInstance(Instance inst, Instances input) {
			int index = -1;
			System.out.println("=======");
			double distance = Double.MAX_VALUE;
			for(int i = 0; i < input.size(); ++i) {
				double temp = getdistance(inst, input.get(i));
				System.out.print(" "+temp);
				if(temp < distance) {
					index = i;
					distance = temp;
				}
			}
			System.out.println("=======");
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
