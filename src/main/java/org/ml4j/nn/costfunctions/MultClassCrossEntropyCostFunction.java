package org.ml4j.nn.costfunctions;

import org.jblas.DoubleMatrix;

public class MultClassCrossEntropyCostFunction implements CostFunction {

	public double getCost(DoubleMatrix desiredOutputs, DoubleMatrix actualOutputs) {
		int m = desiredOutputs.getRows();

		DoubleMatrix J_part = (desiredOutputs.mul(-1).mul(limitLog(actualOutputs))).rowSums();

		return J_part.sum() / (2 * m);

	}

	private double limit(double p) {
		p = Math.min(p, 1 - 0.000000000000001);
		p = Math.max(p, 0.000000000000001);
		return p;
	}

	private DoubleMatrix limitLog(DoubleMatrix m) {
		DoubleMatrix x = m.dup();
		for (int i = 0; i < x.getLength(); i++)
			x.put(i, (double) Math.log(limit(x.get(i))));
		return x;
	}
}
