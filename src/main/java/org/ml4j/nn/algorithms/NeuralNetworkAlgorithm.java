package org.ml4j.nn.algorithms;

import java.util.List;

import org.jblas.DoubleMatrix;
import org.ml4j.nn.NeuralNetwork;
import org.ml4j.nn.NeuralNetworkLayer;
import org.ml4j.nn.costfunctions.CostFunction;

public class NeuralNetworkAlgorithm {

	private NeuralNetwork neuralNetwork;

	public NeuralNetworkAlgorithm(NeuralNetwork neuralNetwork) {
		this.neuralNetwork = neuralNetwork;

	}

	private CostFunction getCostFunction(NeuralNetworkAlgorithmTrainingContext context) {
		if (context.getCostFunction() == null) {
			List<NeuralNetworkLayer> layers = neuralNetwork.getLayers();
			NeuralNetworkLayer outerLayer = layers.get(layers.size() - 1);
			return outerLayer.getActivationFunction().getDefaultCostFunction();
		} else {
			return context.getCostFunction();
		}
	}

	private double[] createLayerRegularisations(double regularisationLamdba) {
		double[] layerRegularisations = new double[neuralNetwork.getLayers().size()];
		for (int i = 0; i < layerRegularisations.length; i++) {
			layerRegularisations[i] = regularisationLamdba;
		}
		return layerRegularisations;
	}

	public NeuralNetworkHypothesisFunction getHypothesisFunction(double[][] inputs, double[][] outputs,
			NeuralNetworkAlgorithmTrainingContext context) {

		neuralNetwork.train(new DoubleMatrix(inputs), new DoubleMatrix(outputs),
				createLayerRegularisations(context.getRegularizationLambda()), getCostFunction(context),
				context.getMaxIterations());

		return new NeuralNetworkHypothesisFunction(neuralNetwork);
	}

}
