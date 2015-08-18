package org.ml4j;

public class NoOpMatrixOptimisationStrategy implements MatrixOptimisationStrategy {

	@Override
	public DoubleMatrix optimise(DoubleMatrix matrix) {
		return matrix;
	}

}
