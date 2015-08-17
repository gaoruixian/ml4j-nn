package org.ml4j;




public class MatrixFunctions {

	
	public static DoubleMatrix pow(DoubleMatrix doubleMatrix, int i) {
		return new DoubleMatrix(DoubleMatrix.strategy.pow(doubleMatrix.matrix, i));
	}

	public static DoubleMatrix log(DoubleMatrix doubleMatrix) {
		return new DoubleMatrix(DoubleMatrix.strategy.log(doubleMatrix.matrix));
	}

	public static DoubleMatrix expi(DoubleMatrix doubleMatrix) { 
		
		DoubleMatrix.strategy.expi(doubleMatrix.matrix);
		return doubleMatrix;
	}

	public static DoubleMatrix powi(DoubleMatrix doubleMatrix, int d) {
		
		DoubleMatrix.strategy.powi(doubleMatrix.matrix,d);
		return doubleMatrix;

	}
	
	public static DoubleMatrix logi(DoubleMatrix doubleMatrix) {
		
		DoubleMatrix.strategy.logi(doubleMatrix.matrix);
		return doubleMatrix;
	}

}
