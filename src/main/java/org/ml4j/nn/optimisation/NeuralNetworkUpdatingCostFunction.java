package org.ml4j.nn.optimisation;

import org.jblas.DoubleMatrix;
import org.ml4j.nn.NeuralNetwork;
import org.ml4j.nn.costfunctions.CostFunction;

public class NeuralNetworkUpdatingCostFunction implements MinimisableCostAndGradientFunction {

	private DoubleMatrix X; // Training input matrix
	private DoubleMatrix Y; // Training output matrix
	private double[] lambda; // Used for regularization
	private NeuralNetwork neuralNetwork;
	private CostFunction costFunction;

	/**
	 * Constructs a cost function with given neural network variables.
	 */
	public NeuralNetworkUpdatingCostFunction(DoubleMatrix setX, DoubleMatrix setY, int[] setTopology,
			double[] setLambda, NeuralNetwork neuralNetwork, CostFunction costFunction) {
		X = new DoubleMatrix().copy(setX);
		Y = new DoubleMatrix().copy(setY);
		lambda = setLambda;
		this.neuralNetwork = neuralNetwork;
		this.costFunction = costFunction;
	}

	@Override
	public Tuple<Double, DoubleMatrix> evaluateCost(DoubleMatrix thetas) {

		neuralNetwork.updateThetasForRetrainableLayers(thetas, true);
		return neuralNetwork.calculateCostAndGradientsForRetrainableLayers(X, Y, lambda, costFunction);
	}

}
