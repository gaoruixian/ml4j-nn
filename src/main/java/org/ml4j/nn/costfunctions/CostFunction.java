package org.ml4j.nn.costfunctions;

import org.ml4j.DoubleMatrix;

public interface CostFunction {

	public double getCost(DoubleMatrix desiredOutputs, DoubleMatrix actualOutputs);

}
