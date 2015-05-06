package org.ml4j.nn.costfunctions;

import org.jblas.DoubleMatrix;

public interface CostFunction {

	public double getCost(DoubleMatrix desiredOutputs, DoubleMatrix actualOutputs);

}
