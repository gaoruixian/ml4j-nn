package org.ml4j.nn;

import java.io.Serializable;

public abstract class DeepBeliefNetwork<N extends DeepBeliefNetwork<N>> extends BaseNeuralNetwork<RestrictedBoltzmannLayer,N> implements Serializable {

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;
	protected RestrictedBoltzmannMachineStack  unsupervisedRbmStack;

	protected DeepBeliefNetwork(RestrictedBoltzmannMachineStack unsupervisedRbmStack)
	{
		super(getLayers(unsupervisedRbmStack));
		this.unsupervisedRbmStack = unsupervisedRbmStack;
	}
	
	protected DeepBeliefNetwork(RestrictedBoltzmannMachineStack unsupervisedRbmStack,RestrictedBoltzmannMachine supervisedLayer)
	{
		super(getLayers(unsupervisedRbmStack,supervisedLayer));
		this.unsupervisedRbmStack = unsupervisedRbmStack;
	}
	
	private static RestrictedBoltzmannLayer[] getLayers(RestrictedBoltzmannMachineStack unsupervisedStack)
	{
		RestrictedBoltzmannLayer[] layers = new RestrictedBoltzmannLayer[unsupervisedStack.size()];
		int i = 0;
		for (RestrictedBoltzmannMachine rbm : unsupervisedStack)
		{
			layers[i++] = rbm.getLayer();
		}
		return layers;
	}
	
	public abstract FeedForwardNeuralNetwork createFeedForwardNeuralNetwork();
	
	private static RestrictedBoltzmannLayer[] getLayers(RestrictedBoltzmannMachineStack unsupervisedStack,RestrictedBoltzmannMachine supervisedRbm)
	{
		RestrictedBoltzmannLayer[] layers = new RestrictedBoltzmannLayer[unsupervisedStack.size() + 1];
		int i = 0;
		for (RestrictedBoltzmannMachine rbm : unsupervisedStack)
		{
			layers[i++] = rbm.getLayer();
		}
		layers[i++] = supervisedRbm.getLayer();
		return layers;
	}
}
