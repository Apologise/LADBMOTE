package smote;
import static util.Utils.*;
import weka.core.converters.ConverterUtils.DataSource;
import javax.print.attribute.standard.PrinterMessageFromOperator;
import weka.core.Instance;
import weka.core.Instances;

public class LoadData {
	public static Instances loadData(String filePath) throws Exception {
		Instances data = DataSource.read(filePath);
		if(data.classIndex() == -1) {
			data.setClassIndex(data.numAttributes()-1);
		}
		return data;
	}
	
	public static void main(String[] args) throws Exception {
		String filePath = "5-fold-abalone19/abalone19-5-1tra.arff";
		Instances data = loadData(filePath);
		System.out.println(data.get(2).value(0));
	}
	
}
