package org.ml4j;

import java.io.Serializable;

public interface MatrixOptimisationStrategy extends Serializable{

	public DoubleMatrix optimise(DoubleMatrix matrix);
}
