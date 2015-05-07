package org.ml4j.nn;

import org.jblas.DoubleMatrix;
import org.ml4j.nn.activationfunctions.ActivationFunction;
import org.ml4j.nn.costfunctions.CostFunction;

public class AutoEncoder extends BaseNeuralNetwork<AutoEncoder> {

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;

	public AutoEncoder(int inputNeuronCount,int hiddenNeuronCount,ActivationFunction encodingActivationFunction,ActivationFunction decodingActivationFunction)
	{
		this(new NeuralNetworkLayer(inputNeuronCount,hiddenNeuronCount,encodingActivationFunction),new NeuralNetworkLayer(hiddenNeuronCount,hiddenNeuronCount,decodingActivationFunction));
	}
	public AutoEncoder(NeuralNetworkLayer... layers)
	{
		super(layers);
		NeuralNetworkLayer firstLayer = layers[0];
		NeuralNetworkLayer lastLayer = layers[layers.length - 1];
		if (firstLayer.getInputNeuronCount() != lastLayer.getOutputNeuronCount())
		{
			throw new IllegalArgumentException("Input layer neuron count must be the same as output activations");
		}
	}
	
	public void addLayer(NeuralNetworkLayer layer)
	{
		getLayers().add(layer);
	}
	
	public void train(DoubleMatrix inputs, double[] lambdas, int max_iter) {
		train(inputs, lambdas, max_iter);
	}

	public void train(DoubleMatrix inputs, double lambda, int max_iter) {
		train(inputs, lambda, max_iter);
	}

	public void train(DoubleMatrix inputs, double lambda, CostFunction costFunction,
			int max_iter) {
		train(inputs, lambda, costFunction, max_iter);
	}

	public void train(DoubleMatrix inputs, double[] lambdas, CostFunction costFunction,
			int max_iter) {
		super.train(inputs, inputs, lambdas, costFunction, max_iter);
	}
	@Override
	protected AutoEncoder createFromLayers(NeuralNetworkLayer[] layers) {
		return new AutoEncoder(layers);
	}
	
	
}
