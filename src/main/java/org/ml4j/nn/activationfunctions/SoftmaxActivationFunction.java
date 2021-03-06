package org.ml4j.nn.activationfunctions;

import org.ml4j.DoubleMatrix;
import org.ml4j.nn.costfunctions.CostFunction;
import org.ml4j.nn.costfunctions.MultClassCrossEntropyCostFunction;
import org.ml4j.nn.util.NeuralNetworkUtils;

public class SoftmaxActivationFunction implements DifferentiableActivationFunction {

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;

	@Override
	public DoubleMatrix activate(DoubleMatrix input) {
		return NeuralNetworkUtils.softmax(input);
	}

	@Override
	public DoubleMatrix activationGradient(DoubleMatrix input) {
		return NeuralNetworkUtils.softmaxGradient(input);
	}

	@Override
	public CostFunction getDefaultCostFunction() {
		return new MultClassCrossEntropyCostFunction();
	}

}
