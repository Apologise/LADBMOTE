package smote;

import java.lang.reflect.Array;

public class TryTest {
	public static int func(int i, int[] array) {
		array[i] = array[i]+1;
		return array[i];
	}
	public static void main(String[] args) {
		// TODO Auto-generated method stub
		int[] a = {1,2,3,4,5};
		for(int i = 0; i < 10; ++i) {
		try {
			System.out.println(func(i,a));
		} catch (Exception e) {
			// TODO: handle exception
			System.out.println("错误");
		}
		}
		
		
	}

}
