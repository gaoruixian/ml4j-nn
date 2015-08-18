package org.ml4j.jblas;

import org.jblas.DoubleMatrix;
import org.jblas.MatrixFunctions;




public class JBlasMatrixFunctions {

	
	public static JBlasDoubleMatrix pow(JBlasDoubleMatrix doubleMatrix, int i) {
		
		return new JBlasDoubleMatrix(MatrixFunctions.pow(doubleMatrix.matrix, i));
	}

	public static JBlasDoubleMatrix log(JBlasDoubleMatrix doubleMatrix) {
		return new JBlasDoubleMatrix(MatrixFunctions.log(doubleMatrix.matrix));
	}

	public static JBlasDoubleMatrix exp(JBlasDoubleMatrix x) { 
		return new JBlasDoubleMatrix(MatrixFunctions.exp(x.matrix));
	}
	
	public static JBlasDoubleMatrix sigmoid(JBlasDoubleMatrix x) { 

		DoubleMatrix result = new DoubleMatrix();
		result = x.matrix;
		result = MatrixFunctions.expi(result.mul(-1));
		result = result.add(1);
		result = MatrixFunctions.powi(result, -1);
		return new JBlasDoubleMatrix(result);
	}
	
	/**
	 * Applies the <i>exponential</i> function element-wise on this
	 * matrix. Note that this is an in-place operation.
	 * @see JBlasMatrixFunctions#exp(DoubleMatrix)
	 * @return this matrix
	 */		
	public static JBlasDoubleMatrix expi(JBlasDoubleMatrix x) { 
		

		/*# mapfct('Math.exp') #*/
//RJPP-BEGIN------------------------------------------------------------
	   for (int i = 0; i < x.getLength(); i++)
	      x.put(i, (double) Math.exp(x.get(i)));

	   return x;
//RJPP-END--------------------------------------------------------------
	}

	public static JBlasDoubleMatrix powi(JBlasDoubleMatrix x, int d) {
		

		if (d == 2.0)
		{
			JBlasDoubleMatrix result = x.muli(x);
			return result;
		}
		else {
			for (int i = 0; i < x.getLength(); i++)
				x.put(i, (double) Math.pow(x.get(i), d));

			return x;
		}		
	}
	
	public static JBlasDoubleMatrix logi(JBlasDoubleMatrix x) {
		
		/*# mapfct('Math.log') #*/
//RJPP-BEGIN------------------------------------------------------------
	   for (int i = 0; i < x.getLength(); i++)
	      x.put(i, (double) Math.log(x.get(i)));
	  
	   return x;
//RJPP-END--------------------------------------------------------------
	}

}
