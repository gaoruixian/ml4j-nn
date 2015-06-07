package org.ml4j.nn;

import org.jblas.DoubleMatrix;
import org.ml4j.nn.activationfunctions.ActivationFunction;


public abstract class BipartiteUndirectedGraph<L extends BipartiteUndirectedGraph<L>> extends UndirectedLayer<L> {

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;


	protected int visibleNeuronCount;
	protected int hiddenNeuronCount;

	protected DoubleMatrix thetas;
	
	protected ActivationFunction visibleActivationFunction;
	protected ActivationFunction hiddenActivationFunction;

	public BipartiteUndirectedGraph(int visibleNeuronCount,int hiddenNeuronCount,DoubleMatrix thetas,ActivationFunction visibleActivationFunction,ActivationFunction hiddenActivationFunction,boolean retrainable)
	{
		super(retrainable);
		this.visibleNeuronCount = visibleNeuronCount;
		this.hiddenNeuronCount = hiddenNeuronCount;
		this.visibleActivationFunction = visibleActivationFunction;
		this.hiddenActivationFunction = hiddenActivationFunction;
		this.thetas = thetas;

	}
	
	public void setThetas(DoubleMatrix thetas) {
		this.thetas = thetas;
	}
	
	public int getVisibleNeuronCount() {
		return visibleNeuronCount;
	}

	public int getHiddenNeuronCount() {
		return hiddenNeuronCount;
	}
	

	public DoubleMatrix getClonedThetas() {

		DoubleMatrix ret = thetas.dup();
		return ret;
	}

	

}
