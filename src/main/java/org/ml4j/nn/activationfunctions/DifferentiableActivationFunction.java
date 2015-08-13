package org.ml4j.nn.activationfunctions;

import org.ml4j.cuda.DoubleMatrix;
import org.ml4j.nn.costfunctions.CostFunction;

public interface DifferentiableActivationFunction extends ActivationFunction {

	public DoubleMatrix activationGradient(DoubleMatrix input);

	public CostFunction getDefaultCostFunction();
}
