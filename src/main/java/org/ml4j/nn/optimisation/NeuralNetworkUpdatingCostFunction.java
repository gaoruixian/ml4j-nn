package org.ml4j.nn.optimisation;

import java.util.Vector;

import org.jblas.DoubleMatrix;
import org.ml4j.nn.NeuralNetwork;
import org.ml4j.nn.NeuralNetworkLayer;
import org.ml4j.nn.costfunctions.CostFunction;
import org.ml4j.nn.util.NeuralNetworkUtils;

public class NeuralNetworkUpdatingCostFunction implements MinimisableCostAndGradientFunction {

	private DoubleMatrix X; // Training input matrix
	private DoubleMatrix Y; // Training output matrix
	private int[] topology; // Neural network topology
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
		topology = setTopology;
		lambda = setLambda;
		this.neuralNetwork = neuralNetwork;
		this.costFunction = costFunction;
	}

	@Override
	public Tuple<Double, DoubleMatrix> evaluateCost(DoubleMatrix thetas) {

		setThetaOnNetwork(thetas);
		return neuralNetwork.calculateCostAndGradients(X, Y, lambda, costFunction);

	}

	private Vector<DoubleMatrix> setThetaOnNetwork(DoubleMatrix thetas) {
		Vector<DoubleMatrix> Theta = NeuralNetworkUtils.reshapeToList(thetas, topology);
		int ind = 0;
		for (NeuralNetworkLayer layer : neuralNetwork.getLayers()) {
			layer.setThetas(Theta.get(ind++));
		}
		return Theta;
	}

}
