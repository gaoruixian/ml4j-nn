package org.ml4j;

import org.ml4j.cuda.CudaMatrixAdapter;
import org.ml4j.jblas.JBlasMatrixAdapterFactory;

public class CustomCudaConcatenationMatrixAdapterFactory extends JBlasMatrixAdapterFactory {

	@Override
	public MatrixAdapter createHorizontalConcatenation(MatrixAdapter matrix, MatrixAdapter matrix2) {
		
		if (matrix instanceof CudaMatrixAdapter || matrix2 instanceof CudaMatrixAdapter)
		{
			super.createHorizontalConcatenation(matrix.asCudaMatrix(), matrix2.asCudaMatrix());
		}
		
		return super.createHorizontalConcatenation(matrix, matrix2);
	}


	@Override
	public MatrixAdapter createVerticalConcatenation(MatrixAdapter matrix, MatrixAdapter matrix2) {
		if (matrix instanceof CudaMatrixAdapter || matrix2 instanceof CudaMatrixAdapter)
		{
			super.createVerticalConcatenation(matrix.asCudaMatrix(), matrix2.asCudaMatrix());
		}
		return super.createVerticalConcatenation(matrix, matrix2);
	}

	
	
}
