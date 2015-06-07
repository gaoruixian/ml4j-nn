package org.ml4j.nn;

import java.util.List;

import org.jblas.DoubleMatrix;
import org.ml4j.nn.costfunctions.CostFunction;

public class FeedForwardNeuralNetwork extends BaseFeedForwardNeuralNetwork<FeedForwardNeuralNetwork> {

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;

	
	
	public FeedForwardNeuralNetwork(FeedForwardLayer... layers)
	{
		super(layers);
	}
	
	public FeedForwardNeuralNetwork(BaseFeedForwardNeuralNetwork<?> n)
	{
		super(n);
	}
	
	public FeedForwardNeuralNetwork(List<FeedForwardLayer> layers)
	{
		super(layers);
	}
	

	/**
	 * Helper function to compute the accuracy of predictions give said
	 * predictions and correct output matrix
	 */
	public String getAccuracy(DoubleMatrix trainingDataMatrix, DoubleMatrix trainingLabelsMatrix) {

		DoubleMatrix predictions = forwardPropagate(trainingDataMatrix).getPredictions();
		return computeAccuracy(predictions, trainingLabelsMatrix) + "";

	}
	
	protected double computeAccuracy(DoubleMatrix predictions, DoubleMatrix Y) {
		return ((predictions.mul(Y)).sum()) * 100 / Y.getRows();
	}

	@Override
	public void train(DoubleMatrix inputs, DoubleMatrix desiredOutputs, double[] lambdas, int max_iter) {
		super.train(inputs, desiredOutputs, lambdas, max_iter);
	}

	@Override
	public void train(DoubleMatrix inputs, DoubleMatrix desiredOutputs, double lambda, int max_iter) {
		super.train(inputs, desiredOutputs, lambda, max_iter);
	}

	@Override
	public void train(DoubleMatrix inputs, DoubleMatrix desiredOutputs, double lambda, CostFunction costFunction,
			int max_iter) {
		super.train(inputs, desiredOutputs, lambda, costFunction, max_iter);
	}

	@Override
	public void train(DoubleMatrix inputs, DoubleMatrix desiredOutputs, double[] lambdas, CostFunction costFunction,
			int max_iter) {
		super.train(inputs, desiredOutputs, lambdas, costFunction, max_iter);
	}

	@Override
	protected FeedForwardNeuralNetwork createFromLayers(FeedForwardLayer[] layers) {
		return new FeedForwardNeuralNetwork(layers);
	}
	
	
	
}
