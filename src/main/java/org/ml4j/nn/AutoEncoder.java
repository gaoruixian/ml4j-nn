package org.ml4j.nn;

import java.util.List;

import org.jblas.DoubleMatrix;
import org.ml4j.nn.activationfunctions.DifferentiableActivationFunction;
import org.ml4j.nn.costfunctions.CostFunction;

public class AutoEncoder extends BaseNeuralNetwork<AutoEncoder> {

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;

	public AutoEncoder(int inputNeuronCount,int hiddenNeuronCount,DifferentiableActivationFunction encodingActivationFunction,DifferentiableActivationFunction decodingActivationFunction)
	{
		this(new NeuralNetworkLayer(inputNeuronCount,hiddenNeuronCount,encodingActivationFunction),new NeuralNetworkLayer(hiddenNeuronCount,inputNeuronCount,decodingActivationFunction));
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
	
	public AutoEncoder(List<NeuralNetworkLayer> layers)
	{
		super(layers);
		NeuralNetworkLayer firstLayer = getFirstLayer();
		NeuralNetworkLayer lastLayer = getOuterLayer();
		if (firstLayer.getInputNeuronCount() != lastLayer.getOutputNeuronCount())
		{
			throw new IllegalArgumentException("Input layer neuron count must be the same as output activations");
		}
	}  
	
	public AutoEncoder(AutoEncoder encoder)
	{
		super(encoder);
		NeuralNetworkLayer firstLayer = getFirstLayer();
		NeuralNetworkLayer lastLayer = getOuterLayer();
		if (firstLayer.getInputNeuronCount() != lastLayer.getOutputNeuronCount())
		{
			throw new IllegalArgumentException("Input layer neuron count must be the same as output activations");
		}
	}

	
	public void train(DoubleMatrix inputs, double[] lambdas, int max_iter) {
		super.train(inputs, inputs, lambdas, max_iter);
	}

	public void train(DoubleMatrix inputs, double lambda, int max_iter) {
		super.train(inputs, inputs, lambda, max_iter);
	}

	public void train(DoubleMatrix inputs, double lambda, CostFunction costFunction,
			int max_iter) {
		super.train(inputs, inputs, lambda, costFunction, max_iter);
	}

	public void train(DoubleMatrix inputs, double[] lambdas, CostFunction costFunction,
			int max_iter) {
		super.train(inputs, inputs, lambdas, costFunction, max_iter);
	}
	@Override
	protected AutoEncoder createFromLayers(NeuralNetworkLayer[] layers) {
		return new AutoEncoder(layers);
	}
	
	public StackedAutoEncoder stack(AutoEncoder... autoEncoders) {
		AutoEncoder[] encoders = new AutoEncoder[1 + autoEncoders.length];
		encoders[0] = this;
		for (int i = 0; i < autoEncoders.length;i++)
		{
			encoders[i+1] = autoEncoders[i];
		}
		return new StackedAutoEncoder(encoders);
	}
	
	public static StackedAutoEncoder createStackedAutoEncoder(AutoEncoder... autoEncoders)
	{
		return new StackedAutoEncoder(autoEncoders);
	}
	
	public NeuralNetwork cloneAndRemoveOuterLayer()
	{
		NeuralNetworkLayer[] layers = new NeuralNetworkLayer[getNumberOfLayers() -2];
		for (int i = 0; i < layers.length;i++)
		{
			layers[i] = this.getLayers().get(i);
		}
		return new NeuralNetwork(layers);
	}
	
	public NeuralNetwork cloneAndReplaceOuterLayer(NeuralNetworkLayer layer)
	{
		NeuralNetworkLayer[] layers = new NeuralNetworkLayer[getNumberOfLayers() -1];
		for (int i = 0; i < layers.length;i++)
		{
			layers[i] = this.getLayers().get(i);
		}
		layers[layers.length -1] = layer;
		return new NeuralNetwork(layers);
	}
	
	public double[][] encodeToLayer(double[][] numericFeaturesMatrix,int toLayerIndex) {
		return forwardPropagateFromTo(numericFeaturesMatrix, 0, toLayerIndex).getOutputs().toArray2();
	}
	
	public double[][] decodeFromLayer(double[][] numericFeaturesMatrix,int fromLayerIndex) {
		return forwardPropagateFromTo(numericFeaturesMatrix, fromLayerIndex, getNumberOfLayers() - 1).getOutputs().toArray2();
	}
	
	public double[] decodeFromLayer(double[] encodedFeatures,int fromLayerIndex) {
		return forwardPropagateFromTo(encodedFeatures, fromLayerIndex, getNumberOfLayers() - 1).getOutputs().toArray();

	}
	public double[] encodeToLayer(double[] numericFeatures,int toLayer) {
		return forwardPropagateFromTo(numericFeatures, 0, toLayer).getOutputs().toArray();
	}
	
	
	
}
