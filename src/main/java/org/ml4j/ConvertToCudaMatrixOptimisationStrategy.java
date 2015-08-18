package org.ml4j;

public class ConvertToCudaMatrixOptimisationStrategy implements MatrixOptimisationStrategy {

	@Override
	public DoubleMatrix optimise(DoubleMatrix matrix) {
		return matrix.asCudaMatrix();
	}

}
