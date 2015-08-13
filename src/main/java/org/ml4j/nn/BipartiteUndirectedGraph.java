/*
 * Copyright 2015 the original author or authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.ml4j.nn;

import org.ml4j.cuda.DoubleMatrix;
import org.ml4j.nn.activationfunctions.ActivationFunction;

/**
 * @author Michael Lavelle
 * 
 * An UndirectedLayer which separates neurons into a visible group and a hidden group.
 * 
 * There are no visible-visible connections, or hidden-hidden connections.
 * 
 * The connections between visible and hidden units are undirected - each pair of visible-hidden units has a symmetric
 * connection weight between them which determines how information flows in either direction.
 * 
 * The connection weight between visible unit i and hidden unit j, w(i,j) = thetas.get(i,j)
 *
 * @param <L> The type of BipartiteUndirectedGraph<L> this instance represents
 *  
 *  */
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
		if (thetas != null && (thetas.getRows() != visibleNeuronCount + 1 || thetas.getColumns() != hiddenNeuronCount + 1))
		{
			throw new IllegalArgumentException("Thetas incorrect size of " + thetas.getRows() + ":" + thetas.getColumns() + " - required is:" +  (visibleNeuronCount + 1) + ":" + (hiddenNeuronCount + 1));
		}

	}
	
	public void setThetas(DoubleMatrix thetas) {
		this.thetas = thetas;
	}
	
	/**
	 * 
	 * @return
	 */
	public int getVisibleNeuronCount() {
		return visibleNeuronCount;
	}

	public int getHiddenNeuronCount() {
		return hiddenNeuronCount;
	}
	
	/**
	 * A clone of the weights matrix
	 * 
	 * The matrix dimensions are visibleNeuronCount + 1:hiddenNeuronCount + 1
	 *
	 * @return A duplicated matrix of weights connecting visible and hidden neurons
	 * - including bias units.
	 * 
	 */
	public DoubleMatrix getClonedThetas() {

		DoubleMatrix ret = thetas.dup();
		return ret;
	}

	

}
