package org.ml4j.nn.costfunctions;

import org.jblas.DoubleMatrix;
import org.jblas.MatrixFunctions;

public class CrossEntropyCostFunction implements CostFunction {

	public double getCost(DoubleMatrix desiredOutputs, DoubleMatrix actualOutputs) {
		int m = desiredOutputs.getRows();

		DoubleMatrix J_part = (desiredOutputs.mul(-1).mul(MatrixFunctions.log(actualOutputs)).sub(desiredOutputs
				.mul(-1).add(1).mul(MatrixFunctions.log(actualOutputs.mul(-1).add(1))))).rowSums();
		return J_part.sum() / (2 * m);

	}

}
