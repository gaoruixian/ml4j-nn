package org.ml4j.nn;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public abstract class BaseNeuralNetwork<L extends BaseLayer<?>,N extends BaseNeuralNetwork<L,N>> {

	protected List<L> layers;
	
	public int getNumberOfLayers()
	{
		return layers.size();
	}
	
	protected BaseNeuralNetwork(L[] layers)
	{
		this.layers = Arrays.asList(layers);
	}
	
	protected BaseNeuralNetwork(List<L> layers)
	{
		this.layers = new ArrayList<L>();
		this.layers.addAll(layers);
	}
	
	public List<L> getLayers()
	{
		return layers;
	}

}
