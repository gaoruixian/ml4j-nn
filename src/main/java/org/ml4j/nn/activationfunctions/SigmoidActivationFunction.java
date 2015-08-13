package org.ml4j.nn.activationfunctions;

import org.ml4j.cuda.DoubleMatrix;
import org.ml4j.nn.costfunctions.CostFunction;
import org.ml4j.nn.costfunctions.CrossEntropyCostFunction;
import org.ml4j.nn.util.NeuralNetworkUtils;

public class SigmoidActivationFunction implements DifferentiableActivationFunction {

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;

	@Override
	public DoubleMatrix activate(DoubleMatrix input) {
		return NeuralNetworkUtils.sigmoid(input);
	}

	@Override
	public DoubleMatrix activationGradient(DoubleMatrix input) {
		return NeuralNetworkUtils.sigmoidGradiant(input);
	}

	@Override
	public CostFunction getDefaultCostFunction() {
		return new CrossEntropyCostFunction();
	}

}
