package org.ml4j;




public class MatrixFunctions {

	
	public static DoubleMatrix pow(DoubleMatrix doubleMatrix, int i) {
		return new DoubleMatrix(DoubleMatrixConfig.getDoubleMatrixStrategy().pow(doubleMatrix.matrix, i));
	}

	public static DoubleMatrix log(DoubleMatrix doubleMatrix) {
		return new DoubleMatrix(DoubleMatrixConfig.getDoubleMatrixStrategy().log(doubleMatrix.matrix));
	}

	public static DoubleMatrix expi(DoubleMatrix doubleMatrix) { 
		
		DoubleMatrixConfig.getDoubleMatrixStrategy().expi(doubleMatrix.matrix);
		return doubleMatrix;
	}

	public static DoubleMatrix powi(DoubleMatrix doubleMatrix, int d) {
		
		DoubleMatrixConfig.getDoubleMatrixStrategy().powi(doubleMatrix.matrix,d);
		return doubleMatrix;

	}
	
	public static DoubleMatrix logi(DoubleMatrix doubleMatrix) {
		
		DoubleMatrixConfig.getDoubleMatrixStrategy().logi(doubleMatrix.matrix);
		return doubleMatrix;
	}

}
