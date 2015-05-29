package org.ml4j.nn.activationfunctions;

import org.jblas.DoubleMatrix;
import org.ml4j.nn.costfunctions.CostFunction;
import org.ml4j.nn.costfunctions.SSECostFunction;

public class LinearActivationFunction implements ActivationFunction {

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;

	@Override
	public DoubleMatrix activate(DoubleMatrix input) {
		return input;
	}

	@Override
	public DoubleMatrix activationGradient(DoubleMatrix input) {
		return DoubleMatrix.ones(input.getRows(),input.getColumns());
	}

	@Override
	public CostFunction getDefaultCostFunction() {
		return new SSECostFunction();
	}

}
