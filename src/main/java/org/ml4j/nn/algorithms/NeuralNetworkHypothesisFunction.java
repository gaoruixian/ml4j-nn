package org.ml4j.nn.algorithms;

import org.jblas.DoubleMatrix;
import org.ml4j.algorithms.HypothesisFunction;
import org.ml4j.nn.NeuralNetwork;

public class NeuralNetworkHypothesisFunction implements HypothesisFunction<double[], double[]> {

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;

	protected NeuralNetwork neuralNetwork;

	public NeuralNetworkHypothesisFunction(NeuralNetwork neuralNetwork) {
		this.neuralNetwork = neuralNetwork;
	}

	public NeuralNetwork getNeuralNetwork() {
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
		DoubleMatrix inputs = new DoubleMatrix(arg0).transpose();
		double[][] predictions = neuralNetwork.forwardPropagate(inputs).getOutputs().toArray2();
		return predictions;
	}

	/**
	 * Helper function to compute the accuracy of predictions give said
	 * predictions and correct output matrix
	 */

	public String getAccuracy(double[][] trainingDataMatrix, double[][] trainingLabelsMatrix) {

		return neuralNetwork.getAccuracy(new DoubleMatrix(trainingDataMatrix), new DoubleMatrix(trainingLabelsMatrix));

	}

	public String getAccuracy(DoubleMatrix trainingDataMatrix, DoubleMatrix trainingLabelsMatrix) {

		return neuralNetwork.getAccuracy(trainingDataMatrix, trainingLabelsMatrix);

	}

}
