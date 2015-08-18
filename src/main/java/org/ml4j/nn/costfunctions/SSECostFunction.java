package org.ml4j.nn.costfunctions;

import org.ml4j.DoubleMatrix;

public class SSECostFunction implements CostFunction {

	public double getCost(DoubleMatrix desiredOutputs, DoubleMatrix actualOutputs) {
		int m = desiredOutputs.getRows();
		DoubleMatrix E = desiredOutputs.sub(actualOutputs);
		DoubleMatrix J_part = E.mul(E);
		return J_part.sum() / (2 * m);
	}

}
