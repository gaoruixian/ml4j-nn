package org.ml4j.cuda;



public class MatrixFunctions {

	
	public static DoubleMatrix pow(DoubleMatrix doubleMatrix, int i) {
		
		if (i == 2) return doubleMatrix.mul(doubleMatrix);
		return powi(doubleMatrix.dup(), i);
	}

	public static DoubleMatrix log(DoubleMatrix x) {
		return logi(x.dup()); 
	}


	
	/**
	 * Applies the <i>exponential</i> function element-wise on this
	 * matrix. Note that this is an in-place operation.
	 * @see MatrixFunctions#exp(DoubleMatrix)
	 * @return this matrix
	 */		
	public static DoubleMatrix expi(DoubleMatrix x) { 
		/*# mapfct('Math.exp') #*/
//RJPP-BEGIN------------------------------------------------------------
	   for (int i = 0; i < x.getLength(); i++)
	      x.put(i, (double) Math.exp(x.get(i)));
	   return x;
//RJPP-END--------------------------------------------------------------
	}

	public static DoubleMatrix powi(DoubleMatrix x, int d) {
		
		if (d == 2.0)
			return x.muli(x);
		else {
			for (int i = 0; i < x.getLength(); i++)
				x.put(i, (double) Math.pow(x.get(i), d));
			return x;
		}		
	}
	
	public static DoubleMatrix logi(DoubleMatrix x) {
		/*# mapfct('Math.log') #*/
//RJPP-BEGIN------------------------------------------------------------
	   for (int i = 0; i < x.getLength(); i++)
	      x.put(i, (double) Math.log(x.get(i)));
	   return x;
//RJPP-END--------------------------------------------------------------
	}

}
