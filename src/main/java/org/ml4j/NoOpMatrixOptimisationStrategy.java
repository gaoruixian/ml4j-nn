package org.ml4j;

public class NoOpMatrixOptimisationStrategy implements MatrixOptimisationStrategy {

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;

	@Override
	public DoubleMatrix optimise(DoubleMatrix matrix) {
		return matrix;
	}

}
