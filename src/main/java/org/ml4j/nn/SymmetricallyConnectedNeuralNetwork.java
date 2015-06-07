package org.ml4j.nn;


public class SymmetricallyConnectedNeuralNetwork<L extends UndirectedLayer<L>,N extends SymmetricallyConnectedNeuralNetwork<L,N>> extends BaseNeuralNetwork<L,N> {

	public SymmetricallyConnectedNeuralNetwork(L[] layers)
	{
		super(layers);
	}
}
