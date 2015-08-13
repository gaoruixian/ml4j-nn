package org.ml4j.nn.optimisation;

import org.ml4j.DoubleMatrices;
import org.ml4j.DoubleMatricesFactory;
import org.ml4j.cuda.DoubleMatrix;
import org.ml4j.nn.BaseFeedForwardNeuralNetwork;
import org.ml4j.nn.costfunctions.CostFunction;

public class NeuralNetworkUpdatingCostFunction implements MinimisableCostAndGradientFunction {

	private DoubleMatrix X; // Training input matrix
	private DoubleMatrix Y; // Training output matrix
	private double[] lambda; // Used for regularization
	private BaseFeedForwardNeuralNetwork<?,?> neuralNetwork;
	private CostFunction costFunction;
	
	private int[][] topology;
	
	private DoubleMatricesFactory<DoubleMatrix> doubleMatricesFactory;

	/**
	 * Constructs a cost function with given neural network variables.
	 */
	public NeuralNetworkUpdatingCostFunction(DoubleMatrix setX, DoubleMatrix setY, int[][] setTopology,
			double[] setLambda, BaseFeedForwardNeuralNetwork<?,?> neuralNetwork, CostFunction costFunction,DoubleMatricesFactory<DoubleMatrix> doubleMatricesFactory) {
		X = new DoubleMatrix().copy(setX);
		Y = new DoubleMatrix().copy(setY);
		lambda = setLambda;
		this.neuralNetwork = neuralNetwork;
		this.costFunction = costFunction;
		this.topology = setTopology;
		this.doubleMatricesFactory = doubleMatricesFactory;
	}


	public int[][] getRetrainableTopologies() {
		return topology;
	}

	@Override
	public Tuple<Double, DoubleMatrices<DoubleMatrix>> evaluateCost(DoubleMatrices<DoubleMatrix>  thetas) {

		neuralNetwork.updateThetasForRetrainableLayers(thetas, true);
		return neuralNetwork.calculateCostAndGradientsForRetrainableLayers(X, Y, lambda, costFunction);
	}


	@Override
	public DoubleMatricesFactory<DoubleMatrix> getDoubleMatricesFactory() {
		return doubleMatricesFactory;
	}

}
