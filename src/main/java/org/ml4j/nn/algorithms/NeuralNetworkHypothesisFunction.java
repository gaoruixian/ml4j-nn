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
	
	public NeuralNetwork getNeuralNetwork()
	{
		return neuralNetwork;
	}

	@Override
	public double[] predict(double[] arg0) {

		DoubleMatrix inputs = new DoubleMatrix(arg0).transpose();
		// inputs =
		// DoubleMatrix.concatHorizontally(DoubleMatrix.ones(inputs.rows,1),inputs);
		double[] predictions = neuralNetwork.forwardPropagate(inputs).getOutputs().toArray();

		return predictions;
	}

	/**
	 * Helper function to compute the accuracy of predictions give said
	 * predictions and correct output matrix
	 */
	public static double computeAccuracy(DoubleMatrix predictions, DoubleMatrix Y) {
		return ((predictions.mul(Y)).sum()) * 100 / Y.getRows();
	}

	public String getAccuracy(double[][] trainingDataMatrix, double[][] trainingLabelsMatrix) {

		DoubleMatrix Y = new DoubleMatrix(trainingLabelsMatrix);
		DoubleMatrix predictions = neuralNetwork.forwardPropagate(new DoubleMatrix(trainingDataMatrix)).getOutputs();
		return computeAccuracy(predictions, Y) + "";

	}

}
