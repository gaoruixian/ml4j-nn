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

import org.jblas.DoubleMatrix;
import org.ml4j.nn.activationfunctions.DifferentiableActivationFunction;

/**
 * Represents a Directed Layer of a NeuralNetwork - a layer through which information propagates
 * from input neurons to output neurons in one direction
 * 
 * @author Michael Lavelle
 *
 * @param <L> The type of DirectedLayer<L> this instance represents
 */
public abstract class DirectedLayer<L extends DirectedLayer<?>> extends BaseLayer<L>{

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;
	
	protected final boolean hasBiasUnit;
	protected int inputNeuronCount;
	protected int outputNeuronCount;
	
	protected final double inputDropout;


	protected DifferentiableActivationFunction activationFunction;

	/**
	 * DirectedLayer constructor
	 * 
	 * @param inputNeuronCount The number of input neurons, not including any bias unit
	 * @param outputNeuronCount The number of output neurons
	 * @param activationFunction  The activation function which is applied to the 
	 * inputs after they have been multiplied by their weights 
	 * to product the output neuron activities
	 * @param biasUnit Whether this layer contains an additional inputs bias unit, as well as the input neurons specified by inputNeuronCount
	 * @param retrainable Whether this layer can be (re)trained further.
	 */
	public DirectedLayer(int inputNeuronCount,int outputNeuronCount,DifferentiableActivationFunction activationFunction,boolean biasUnit,boolean retrainable,double inputDropout)
	{
		super(retrainable);
		if (activationFunction == null) throw new IllegalArgumentException("Activation function passed to layer cannot be null");
		this.inputNeuronCount = inputNeuronCount;
		this.outputNeuronCount = outputNeuronCount;
		this.activationFunction = activationFunction;
		this.hasBiasUnit = biasUnit;
		this.inputDropout  = inputDropout;
	}
	
	/**
	 * If true, the total input neuron count = (inputNeuronCount + 1)
	 * 
	 * @return A flag to indicate whether the directed layer has an input bias unit which
	 * is in addition to the number specified by the inputNeuronCount attribute.
	 * 
	 */
	public boolean hasBiasUnit() {
		return hasBiasUnit;
	}
	
	public double createDropoutScaling(boolean training)
	{
		return training ? 1 : 1d/inputDropout;
	}
	
	public DoubleMatrix createDropoutMask(DoubleMatrix inputs,boolean training)
	{
		DoubleMatrix dropoutMask = DoubleMatrix.ones(inputs.getRows(),inputs.getColumns());
		if (training && inputDropout != 1)
		{
			for (int i = 0; i < dropoutMask.getRows(); i++)
			{
				for (int j = 0; i < dropoutMask.getColumns(); i++)
				{
					if (Math.random() > inputDropout)
					{
						dropoutMask.put(i, j,0);
					}
				}
			}
		}
		return dropoutMask;
	}


	/**
	 * Propagate information through this DirectedLayer
	 * 
	 * 
	 * @param inputActivations A matrix of inputActivations ( including
	 * the always-1 activations of the bias unit if  hasBiasUnit() == true ).
	 * 
	 * Each row is a specification of activations for all input units ( including
	 * the bias unit if hasBiasUnit() == true ), with a column for each unit.
	 * 
	 * 
	 * @return A NeuralNetworkLayerActivation instance specifying how
	 * the information propagated through the layer.
	 */
	protected abstract NeuralNetworkLayerActivation<L> forwardPropagate(
			DoubleMatrix inputActivations,boolean training);
	

	/**
	 * @return The count of input neurons ( not including any possible bias unit)
	 */
	public int getInputNeuronCount() {
		return inputNeuronCount;
	}

	/**
	 * 
	 * @return The count of output neurons
	 */
	public int getOutputNeuronCount() {
		return outputNeuronCount;
	}


	/**
	 * @return The activation function which is applied to the 
	 * inputs after they have been multiplied by their weights 
	 * to product the output neuron activities.
	 * 
	 */
	public DifferentiableActivationFunction getActivationFunction() {
		return activationFunction;
	}
	
	public abstract DoubleMatrix getClonedThetas();

	protected abstract void updateThetas(DoubleMatrix doubleMatrix, int layerIndex, boolean permitFurtherRetrains);
	
	public void applyGradientWeightConstraints(DoubleMatrix gradients)
	{
		// No-op by default
	}
}
