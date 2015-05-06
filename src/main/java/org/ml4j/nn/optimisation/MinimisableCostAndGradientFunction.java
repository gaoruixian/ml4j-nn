package org.ml4j.nn.optimisation;

import org.jblas.DoubleMatrix;

public interface MinimisableCostAndGradientFunction {

	/**
	 * Returns a Tuple with first element the Cost, and the second element the
	 * Gradients of given input matrix
	 */
	public Tuple<Double, DoubleMatrix> evaluateCost(DoubleMatrix input);
}
