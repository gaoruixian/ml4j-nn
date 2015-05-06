package org.ml4j.nn.activationfunctions;

import org.jblas.DoubleMatrix;
import org.ml4j.nn.costfunctions.CostFunction;

public interface ActivationFunction {

	public DoubleMatrix activate(DoubleMatrix input);

	public DoubleMatrix activationGradient(DoubleMatrix input);

	// public CostFunction getDefaultCostFunction(DoubleMatrix setX,
	// DoubleMatrix setY,
	// int [] setTopology, double setLambda);

	public CostFunction getDefaultCostFunction();
}
