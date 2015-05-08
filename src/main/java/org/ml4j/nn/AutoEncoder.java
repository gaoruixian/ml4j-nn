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
	
	public boolean isSymmetricTopology()
	{
		if (this.getLayers().size() % 2 != 0)
		{
			return false;
		}
		int half = this.getLayers().size()/2;
		for (int i = 0; i < half; i++)
		{
			if ((this.getLayers().get(i).getInputNeuronCount() != this.getLayers().get(half * 2 - i - 1).getOutputNeuronCount())
				|| (this.getLayers().get(i).getOutputNeuronCount() != this.getLayers().get(half *2- i - 1).getInputNeuronCount()))
				{
					return false;
				}
		}
		return true;
	}
	
	
	public AutoEncoder appendHiddenLayers(boolean createNewAutoEncoder,int[] topology,ActivationFunction[] activationFunctions)
	{
		AutoEncoder encoder = createNewAutoEncoder ? new AutoEncoder(this.getLayers().toArray(new NeuralNetworkLayer[this.getLayers().size()])) : this ;
		NeuralNetworkLayer finalLayer = getLayers().get(getLayers().size() -1);
		int inputNeuronCount = finalLayer.getInputNeuronCount();
		NeuralNetworkLayer[] appendedLayers = new NeuralNetworkLayer[topology.length + 1];
		for (int i = 0; i < topology.length; i++)
		{
			appendedLayers[i] = new NeuralNetworkLayer(inputNeuronCount,topology[i],activationFunctions[i]);
			inputNeuronCount = topology[i];
		}
		appendedLayers[topology.length] = new NeuralNetworkLayer(inputNeuronCount,finalLayer.getOutputNeuronCount(),activationFunctions[appendedLayers.length - 1]);
		return encoder.replaceFinalLayer(createNewAutoEncoder,appendedLayers);
	}
	
	public AutoEncoder replaceFinalLayer(boolean createNewAutoEncoder,NeuralNetworkLayer... layers)
	{
		AutoEncoder encoder = createNewAutoEncoder ? new AutoEncoder(this.getLayers().toArray(new NeuralNetworkLayer[this.getLayers().size()])) : this ;
		for (NeuralNetworkLayer layer : layers)
		{	
			encoder.getLayers().add(encoder.getLayers().size() -1,layer);
		}
		encoder.getLayers().remove(encoder.getLayers().size() -1);
		return encoder;
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
	
	
	
	
	
	
}
