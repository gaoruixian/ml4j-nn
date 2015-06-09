package org.ml4j.nn.algorithms;

import org.jblas.DoubleMatrix;
import org.ml4j.nn.FeedForwardNeuralNetwork;

public class NeuralNetworkAlgorithm {

	private FeedForwardNeuralNetwork neuralNetwork;

	public NeuralNetworkAlgorithm(FeedForwardNeuralNetwork neuralNetwork) {
		this.neuralNetwork = neuralNetwork;

	}

	public NeuralNetworkHypothesisFunction getHypothesisFunction(double[][] inputs, double[][] outputs,
			NeuralNetworkAlgorithmTrainingContext context) {
		
		if (context.getCostFunction() != null) {
			neuralNetwork.train(new DoubleMatrix(inputs), new DoubleMatrix(outputs),
					neuralNetwork.createLayerRegularisations(context.getRegularizationLambda()),
					context.getCostFunction(), context.getMaxIterations());
		} else {
			neuralNetwork.train(new DoubleMatrix(inputs), new DoubleMatrix(outputs),
					neuralNetwork.createLayerRegularisations(context.getRegularizationLambda()),
					context.getMaxIterations());
		}

		return new NeuralNetworkHypothesisFunction(neuralNetwork);
	}

}
