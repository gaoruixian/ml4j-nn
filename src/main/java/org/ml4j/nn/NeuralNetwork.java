package org.ml4j.nn;

import org.jblas.DoubleMatrix;
import org.ml4j.nn.costfunctions.CostFunction;

public class NeuralNetwork extends BaseNeuralNetwork<NeuralNetwork> {

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;

	/**
	 * Helper function to compute the accuracy of predictions give said
	 * predictions and correct output matrix
	 */
	
	public NeuralNetwork(NeuralNetworkLayer... layers)
	{
		super(layers);
	}
	

	public String getAccuracy(DoubleMatrix trainingDataMatrix, DoubleMatrix trainingLabelsMatrix) {

		DoubleMatrix predictions = forwardPropagate(trainingDataMatrix).getPredictions();
		return computeAccuracy(predictions, trainingLabelsMatrix) + "";

	}
	
	protected double computeAccuracy(DoubleMatrix predictions, DoubleMatrix Y) {
		return ((predictions.mul(Y)).sum()) * 100 / Y.getRows();
	}

	@Override
	public void train(DoubleMatrix inputs, DoubleMatrix desiredOutputs, double[] lambdas, int max_iter) {
		// TODO Auto-generated method stub
		super.train(inputs, desiredOutputs, lambdas, max_iter);
	}

	@Override
	public void train(DoubleMatrix inputs, DoubleMatrix desiredOutputs, double lambda, int max_iter) {
		// TODO Auto-generated method stub
		super.train(inputs, desiredOutputs, lambda, max_iter);
	}

	@Override
	public void train(DoubleMatrix inputs, DoubleMatrix desiredOutputs, double lambda, CostFunction costFunction,
			int max_iter) {
		// TODO Auto-generated method stub
		super.train(inputs, desiredOutputs, lambda, costFunction, max_iter);
	}

	@Override
	public void train(DoubleMatrix inputs, DoubleMatrix desiredOutputs, double[] lambdas, CostFunction costFunction,
			int max_iter) {
		// TODO Auto-generated method stub
		super.train(inputs, desiredOutputs, lambdas, costFunction, max_iter);
	}

	@Override
	protected NeuralNetwork createFromLayers(NeuralNetworkLayer[] layers) {
		return new NeuralNetwork(layers);
	}
	
	
	
}
