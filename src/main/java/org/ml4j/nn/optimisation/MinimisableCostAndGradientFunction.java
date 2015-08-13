package org.ml4j.nn.optimisation;

import org.ml4j.DoubleMatrices;
import org.ml4j.DoubleMatricesFactory;
import org.ml4j.cuda.DoubleMatrix;

public interface MinimisableCostAndGradientFunction {

	/**
	 * Returns a Tuple with first element the Cost, and the second element the
	 * Gradients of given input matrix
	 */
	public Tuple<Double, DoubleMatrices<DoubleMatrix>> evaluateCost(DoubleMatrices<DoubleMatrix> input);
	
	public int[][] getRetrainableTopologies();
	
	public DoubleMatricesFactory<DoubleMatrix> getDoubleMatricesFactory();
}
