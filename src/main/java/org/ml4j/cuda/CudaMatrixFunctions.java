package org.ml4j.cuda;

import org.nd4j.linalg.api.ndarray.INDArray;




public class CudaMatrixFunctions {

	
	public static CudaDoubleMatrix pow(CudaDoubleMatrix doubleMatrix, int i) {
		
		if (i == 2) return doubleMatrix.mul(doubleMatrix);
		return powi(doubleMatrix.dup(), i);
	}

	public static CudaDoubleMatrix log(CudaDoubleMatrix x) {
		 		return new CudaDoubleMatrix(org.nd4j.linalg.ops.transforms.Transforms.log(x.matrix));

		//return logi(x.dup()); 
	}
	
	
	public static CudaDoubleMatrix logi(CudaDoubleMatrix x) {
 		INDArray result = org.nd4j.linalg.ops.transforms.Transforms.log(x.matrix);
 		x.setMatrix(result);
 		return x;
	}


	public static CudaDoubleMatrix exp(CudaDoubleMatrix x)
	{
		CudaDoubleMatrix result =  new CudaDoubleMatrix(org.nd4j.linalg.ops.transforms.Transforms.exp(x.matrix));
		return result;
	}
	
	public static CudaDoubleMatrix expi(CudaDoubleMatrix x)
	{
		INDArray result = org.nd4j.linalg.ops.transforms.Transforms.exp(x.matrix);
		x.setMatrix(result);
		return x;
	}
	
	public static CudaDoubleMatrix sigmoid(CudaDoubleMatrix x)
	{
		CudaDoubleMatrix result =  new CudaDoubleMatrix(org.nd4j.linalg.ops.transforms.Transforms.sigmoid(x.matrix));
		return result;
	}
	
	
	
	
//	/**
//	 * Applies the <i>exponential</i> function element-wise on this
//	 * matrix. Note that this is an in-place operation.
//	 * @see MatrixFunctions#exp(DoubleMatrix)
//	 * @return this matrix
//	 */		
//	public static DoubleMatrix expi(DoubleMatrix x) { 
//		/*# mapfct('Math.exp') #*/
////RJPP-BEGIN------------------------------------------------------------
//	   for (int i = 0; i < x.getLength(); i++)
//	      x.put(i, (double) Math.exp(x.get(i)));
//	   return x;
////RJPP-END--------------------------------------------------------------
//	}

	public static CudaDoubleMatrix powi(CudaDoubleMatrix x, int d) {
		
		if (d == 2.0)
			return x.muli(x);
		else {
			for (int i = 0; i < x.getLength(); i++)
				x.put(i, (double) Math.pow(x.get(i), d));
			return x;
		}		
	}
	
	
	
//	public static DoubleMatrix logi(DoubleMatrix x) {
//		/*# mapfct('Math.log') #*/
////RJPP-BEGIN------------------------------------------------------------
//	   for (int i = 0; i < x.getLength(); i++)
//	      x.put(i, (double) Math.log(x.get(i)));
//	   return x;
////RJPP-END--------------------------------------------------------------
//	}

}
