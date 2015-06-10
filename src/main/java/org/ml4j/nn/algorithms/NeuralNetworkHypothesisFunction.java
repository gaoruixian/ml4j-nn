package org.ml4j.nn.algorithms;

import org.jblas.DoubleMatrix;
import org.ml4j.algorithms.HypothesisFunction;
import org.ml4j.nn.FeedForwardNeuralNetwork;

public class NeuralNetworkHypothesisFunction implements HypothesisFunction<double[], double[]> {

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;

	protected FeedForwardNeuralNetwork neuralNetwork;

	public NeuralNetworkHypothesisFunction(FeedForwardNeuralNetwork neuralNetwork) {
		this.neuralNetwork = neuralNetwork;
	}

	public FeedForwardNeuralNetwork getNeuralNetwork() {
		return neuralNetwork;
	}

	@Override
	public double[] predict(double[] arg0) {

		DoubleMatrix inputs = new DoubleMatrix(arg0).transpose();
		double[] predictions = neuralNetwork.forwardPropagate(inputs).getOutputs().toArray();

		return predictions;
	}
	
	public double[][] predict(double[][] arg0)
	{
		DoubleMatrix inputs = new DoubleMatrix(arg0);
		double[][] predictions = neuralNetwork.forwardPropagate(inputs).getOutputs().toArray2();
		return predictions;
	}

	/**
	 * Helper function to compute the accuracy of predictions using calculated predictions
	 * predictions and correct output matrix
	 * 
	 * @param trainingDataMatrix The training examples to compute accuracy for
	 * 
	 * @param trainingLabelsMatrix The desired output labels
	 * 
	 * @return
	 */
	public double getAccuracy(double[][] trainingDataMatrix, double[][] trainingLabelsMatrix) {

		return neuralNetwork.getAccuracy(new DoubleMatrix(trainingDataMatrix), new DoubleMatrix(trainingLabelsMatrix));

	}

	/**
	 * Helper function to compute the accuracy of predictions using calculated predictions
	 * predictions and correct output matrix
	 * 
	 * @param trainingDataMatrix The training examples to compute accuracy for
	 * 
	 * @param trainingLabelsMatrix The desired output labels
	 * 
	 * @return The accuracy of the network
	 */
	public double getAccuracy(DoubleMatrix trainingDataMatrix, DoubleMatrix trainingLabelsMatrix) {

		return neuralNetwork.getAccuracy(trainingDataMatrix, trainingLabelsMatrix);

	}

}
