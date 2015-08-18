package org.ml4j.jblas;

import org.ml4j.DefaultMatrixAdapterStrategy;
import org.ml4j.MatrixAdapter;

import Jama.Matrix;

public class NoJblasPresentMatrixAdapterStrategy extends DefaultMatrixAdapterStrategy {

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;

	@Override
	public MatrixAdapter mmul(MatrixAdapter matrix, MatrixAdapter matrix2) {
		Matrix result = toJAMAMatrix(matrix).times(toJAMAMatrix(matrix2));
		return new JBlasMatrixAdapter(result.getArray());
	}
	
	private Matrix toJAMAMatrix(MatrixAdapter matrix)
	{
		return new Matrix(matrix.toArray2(),matrix.getRows(),matrix.getColumns());
	}

	
	
}
