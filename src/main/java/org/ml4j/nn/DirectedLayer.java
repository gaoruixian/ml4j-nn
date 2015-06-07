package org.ml4j.nn;

import org.jblas.DoubleMatrix;
import org.ml4j.nn.activationfunctions.DifferentiableActivationFunction;

public abstract class DirectedLayer<L extends DirectedLayer<?>> extends BaseLayer<L>{

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;

	public DirectedLayer(boolean retrainable) {
		super(retrainable);
	}

	protected abstract NeuralNetworkLayerActivation forwardPropagate(
			DoubleMatrix inputActivations);
	
	protected int inputNeuronCount;
	protected int outputNeuronCount;
	
	public DirectedLayer(int inputNeuronCount,int outputNeuronCount,DifferentiableActivationFunction activationFunction)
	{
		super(true);
		this.inputNeuronCount = inputNeuronCount;
		this.outputNeuronCount = outputNeuronCount;
		this.activationFunction = activationFunction;
	}
	
	

	public int getInputNeuronCount() {
		return inputNeuronCount;
	}

	public int getOutputNeuronCount() {
		return outputNeuronCount;
	}

	protected DifferentiableActivationFunction activationFunction;

	public DifferentiableActivationFunction getActivationFunction() {
		return activationFunction;
	}
	
	
	
	
}
