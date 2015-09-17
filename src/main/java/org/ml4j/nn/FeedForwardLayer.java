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

import java.io.Serializable;

import org.ml4j.DoubleMatrix;
import org.ml4j.MatrixOptimisationStrategy;
import org.ml4j.NoOpMatrixOptimisationStrategy;
import org.ml4j.nn.activationfunctions.DifferentiableActivationFunction;

/**
 * A DirectedLayer which composes input neurons and output neurons into a directed acyclic bipartite graph.
 * 
 * There are no input-input connections or output-output connections,  only input-output connections.
 * 
 * The connection between input neuron i and output neuron j is represented by an input->output weight, w(i,j).
 * 
 * Please note that the DoubleMatrix containing the weights is now 
 * the shape that may be naturally assumed  - ie. w(i,j) = thetas.get(i,j);
 * 
 * @author Michael Lavelle
 *
 */
public class FeedForwardLayer extends DirectedLayer<FeedForwardLayer> implements Serializable {

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;

	private DoubleMatrix thetas;
	protected DoubleMatrix thetasMask;
	protected Double initialThetaScaling;
	
	
	private MatrixOptimisationStrategy forwardPropagationInputMatrixStrategy
	 = new NoOpMatrixOptimisationStrategy();
	
	
	public void setInitialThetaScaling(double v)
	{
		this.initialThetaScaling = v;
	}
	
	public double getInitialThetaScaling()
	{
		return initialThetaScaling != null ? initialThetaScaling : 0.05d;
	}

	public void setForwardPropagationInputMatrixStrategy(MatrixOptimisationStrategy forwardPropagationInputMatrixStrategy) {
		this.forwardPropagationInputMatrixStrategy = forwardPropagationInputMatrixStrategy;
	}

	private void applyThetasMask()
	{
		this.thetas.muli(thetasMask);
	}

	public FeedForwardLayer dup(boolean retrainable) {
		FeedForwardLayer dup = new FeedForwardLayer(inputNeuronCount, outputNeuronCount, this.getClonedThetas(),
				thetasMask,activationFunction, hasBiasUnit(),retrainable,inputDropout);
		return dup;
	}
	
	
	
	/**
	 * Activates the output neurons, by forward propagating information from
	 * the input neuron activities, as specified by a double[][] array.
	 * 
	 * @param layerInputs The activations of the input units ( not including bias units). Many
	 * activations can be forward propagated in parallel using the rows of this matrix, with
	 * each column representing each input neuron.
	 * 
	 * @return The activations of the output units once information has been propagated.
	 * If multiple activation rows were input for parallel processing, the output
	 * will have a row for each parallel output activation.
	 */
	public DoubleMatrix activate(double[][] layerInputsArrays)
	{
		DoubleMatrix layerInputs= new DoubleMatrix(layerInputsArrays);
		if (hasBiasUnit())
		{
			layerInputs = DoubleMatrix.concatHorizontally(DoubleMatrix.ones(layerInputs.getRows(),1), layerInputs);
		}
		return forwardPropagate(layerInputs).getOutputActivations();
	}

	/**
	 * Activates the output neurons, by forward propagating information from
	 * the input neurons as specified by a DoubleMatrix
	 * 
	 * @param layerInputs The activations of the input units ( not including bias units). Many
	 * activations can be forward propagated in parallel using the rows of this matrix, with
	 * each column representing each input neuron.
	 * 
	 * @return The activations of the output units once information has been propagated.
	 * If multiple activation rows were input for parallel processing, the output
	 * will have a row for each parallel output activation.
	 */
	public DoubleMatrix activate(DoubleMatrix layerInputs)
	{
		if (hasBiasUnit())
		{
			layerInputs = DoubleMatrix.concatHorizontally(DoubleMatrix.ones(layerInputs.getRows(),1), layerInputs);
		}
		return forwardPropagate(layerInputs).getOutputActivations();
	}
	
	/**
	 * Propagate information through this FeedForwardLayer
	 * 
	 * @param inputActivations A matrix of inputActivations ( including
	 * the always-1 activations of the bias unit if  hasBiasUnit() == true ).
	 * 
	 * Each row is a specification of activations for all input units ( including
	 * the bias unit if hasBiasUnit() == true ), with a column for each unit.
	 * 
	 * @return A NeuralNetworkLayerActivation instance specifying how
	 * the information propagated through the layer.
	 */
	protected NeuralNetworkLayerActivation<FeedForwardLayer> forwardPropagate(DoubleMatrix layerInputsWithIntercept,boolean training) {
		
		if (layerInputsWithIntercept.getColumns() != getInputNeuronCount() + (hasBiasUnit() ? 1 : 0))
		{
			throw new IllegalArgumentException("Layer forward propogation requires inputs matrix with intercepts with number of rows = " + (getInputNeuronCount() + 1));
		}
		
		DoubleMatrix dropoutMask = this.createDropoutMask(layerInputsWithIntercept, training);
		DoubleMatrix layerInputsWithInterceptAndDropout = layerInputsWithIntercept;
		
		if (training && inputDropout != 1)
		{
			layerInputsWithInterceptAndDropout = layerInputsWithInterceptAndDropout.mul(dropoutMask);
		}
		
		double dropoutScaling = createDropoutScaling(training);
		DoubleMatrix scaledThetas = thetas;
		if (dropoutScaling != 1)
		{
			scaledThetas = thetas.mul(dropoutScaling);
		}
		
		layerInputsWithInterceptAndDropout = forwardPropagationInputMatrixStrategy.optimise(layerInputsWithInterceptAndDropout);
		DoubleMatrix Z = layerInputsWithInterceptAndDropout.mmul(scaledThetas);

		DoubleMatrix acts = activationFunction.activate(Z);
		NeuralNetworkLayerActivation<FeedForwardLayer> activation = new NeuralNetworkLayerActivation<FeedForwardLayer>(this,layerInputsWithInterceptAndDropout, Z, acts,thetasMask,dropoutMask);

		return activation;
	}
	
	protected NeuralNetworkLayerActivation<FeedForwardLayer> forwardPropagate(DoubleMatrix layerInputsWithIntercept) {
		return forwardPropagate(layerInputsWithIntercept,false);
	}
		

	/**
	 * A clone of the weights matrix
	 * 
	 * The matrix dimensions are outputNeuronCount:(inputNeuronCount + hasBiasUnit() ? 1: 0)
	 *
	 * 
	 * @return A duplicated matrix of weights mapping input neurons to output neurons.
	 * Please note that the weight connecting input neuron i to output neuron j,
	 * w(i,j) = getClonedThetas().get(i,j)
	 * 
	 */
	public DoubleMatrix getClonedThetas() {

		DoubleMatrix ret = thetas.dup();
		return ret;
	}
	
	public DoubleMatrix getThetas() {

		return thetas;
	}

	/**
	 * Untrained FeedForwardLayer constructor - sets the layer to retrainable=true
	 * 
	 * @param inputNeuronCount The number of input neurons, not including any bias unit
	 * @param outputNeuronCount The number of output neurons
	 * @param activationFunction  The activation function which is applied to the 
	 * inputs after they have been multiplied by their weights 
	 * to product the output neuron activities
	 * @param biasUnit Whether this layer contains an additional inputs bias unit, as well as the input neurons specified by inputNeuronCount
	 */
	public FeedForwardLayer(int inputNeuronCount, int outputNeuronCount, DifferentiableActivationFunction activationFunction,boolean biasUnit,double inputDropout) {
		super(inputNeuronCount,outputNeuronCount,activationFunction,biasUnit,true,inputDropout);
		this.thetas = generateInitialThetas(getOutputNeuronCount(), getInputNeuronCount() + (biasUnit ? 1 : 0));
		this.thetasMask = DoubleMatrix.ones(thetas.getRows(),thetas.getColumns());
	}
	
	public FeedForwardLayer(int inputNeuronCount, int outputNeuronCount, DifferentiableActivationFunction activationFunction,boolean biasUnit) {
		super(inputNeuronCount,outputNeuronCount,activationFunction,biasUnit,true,1);
		this.thetas = generateInitialThetas(getOutputNeuronCount(), getInputNeuronCount() + (biasUnit ? 1 : 0));
		this.thetasMask = DoubleMatrix.ones(thetas.getRows(),thetas.getColumns());
	}
	
	/**
	 * Untrained FeedForwardLayer constructor - sets the layer to retrainable=true
	 * 
	 * @param inputNeuronCount The number of input neurons, not including any bias unit
	 * @param outputNeuronCount The number of output neurons
	 * @param activationFunction  The activation function which is applied to the 
	 * inputs after they have been multiplied by their weights 
	 * to product the output neuron activities
	 * @param biasUnit Whether this layer contains an additional inputs bias unit, as well as the input neurons specified by inputNeuronCount
	 * @param thetasMask Thetas mask
	 */
	protected FeedForwardLayer(int inputNeuronCount, int outputNeuronCount, DifferentiableActivationFunction activationFunction,boolean biasUnit,DoubleMatrix thetasMask,double inputDropout) {
		super(inputNeuronCount,outputNeuronCount,activationFunction,biasUnit,true,inputDropout);
		this.thetas = generateInitialThetas(getOutputNeuronCount(), getInputNeuronCount() + (biasUnit ? 1 : 0));
		this.thetasMask = thetasMask;
		applyThetasMask();
	}
	
	protected FeedForwardLayer(int inputNeuronCount, int outputNeuronCount, DifferentiableActivationFunction activationFunction,boolean biasUnit,DoubleMatrix thetasMask) {
		super(inputNeuronCount,outputNeuronCount,activationFunction,biasUnit,true,1);
		this.thetas = generateInitialThetas(getOutputNeuronCount(), getInputNeuronCount() + (biasUnit ? 1 : 0));
		this.thetasMask = thetasMask;
		applyThetasMask();
	}
	
	/**
	 * Pre-trained FeedForwardLayer constructor - initializes the weights matrix
	 * and allows retrainable flag to be custom set.
	 * 
	 * @param inputNeuronCount The number of input neurons, not including any bias unit
	 * @param outputNeuronCount The number of output neurons
	 * @param activationFunction  The activation function which is applied to the 
	 * inputs after they have been multiplied by their weights 
	 * to product the output neuron activities
	 * @param biasUnit Whether this layer contains an additional inputs bias unit, as well as the input neurons specified by inputNeuronCount
	 */
	protected FeedForwardLayer(int inputNeuronCount, int outputNeuronCount, DoubleMatrix thetas,DoubleMatrix thetasMask,
			DifferentiableActivationFunction activationFunction, boolean biasUnit,boolean retrainable,double inputDropout) {
		super(inputNeuronCount,outputNeuronCount,activationFunction,biasUnit,retrainable,inputDropout);
		if (thetas == null) throw new IllegalArgumentException("Thetas passed to layer cannot be null");
		if (thetas.getColumns() != outputNeuronCount || thetas.getRows() != (inputNeuronCount + (biasUnit ? 1 : 0))) throw new IllegalArgumentException("Thetas matrix must be of dimensions " + (inputNeuronCount + (hasBiasUnit ? 1 : 0) + ":" +  outputNeuronCount));
		this.thetas = thetas;
		this.thetasMask = thetasMask;
		applyThetasMask();

	}
	
	
	
	@Override
	public String toString() {
		return "FeedForwardLayer with " + inputNeuronCount + " input neurons " + (hasBiasUnit ? "(+ 1 bias unit)" : "") + " and " + outputNeuronCount + " output neurons";
	}

	protected FeedForwardLayer(int inputNeuronCount, int outputNeuronCount, DoubleMatrix thetas,DoubleMatrix thetasMask,
			DifferentiableActivationFunction activationFunction, boolean biasUnit,boolean retrainable) {
		super(inputNeuronCount,outputNeuronCount,activationFunction,biasUnit,retrainable,1);
		if (thetas == null) throw new IllegalArgumentException("Thetas passed to layer cannot be null");
		if (thetas.getRows() != outputNeuronCount || thetas.getRows() != (inputNeuronCount + (biasUnit ? 1 : 0))) throw new IllegalArgumentException("Thetas matrix must be of dimensions " +  (inputNeuronCount + (hasBiasUnit ? 1 : 0) + ":" + outputNeuronCount));
		this.thetas = thetas;
		this.thetasMask = thetasMask;
		applyThetasMask();
	}
	
	
	/**
	 * Pre-trained FeedForwardLayer constructor - initializes the weights matrix
	 * and allows retrainable flag to be custom set.
	 * 
	 * @param inputNeuronCount The number of input neurons, not including any bias unit
	 * @param outputNeuronCount The number of output neurons
	 * @param activationFunction  The activation function which is applied to the 
	 * inputs after they have been multiplied by their weights 
	 * to product the output neuron activities
	 * @param biasUnit Whether this layer contains an additional inputs bias unit, as well as the input neurons specified by inputNeuronCount
	 */
	public FeedForwardLayer(int inputNeuronCount, int outputNeuronCount, DoubleMatrix thetas,
			DifferentiableActivationFunction activationFunction, boolean biasUnit,boolean retrainable,double inputDropout) {
		this(inputNeuronCount,outputNeuronCount,thetas,DoubleMatrix.ones(outputNeuronCount,inputNeuronCount + (biasUnit ?  1 : 0)),activationFunction,biasUnit,retrainable,inputDropout);
	}
	
	public FeedForwardLayer(int inputNeuronCount, int outputNeuronCount, DoubleMatrix thetas,
			DifferentiableActivationFunction activationFunction, boolean biasUnit,boolean retrainable) {
		this(inputNeuronCount,outputNeuronCount,thetas,DoubleMatrix.ones(outputNeuronCount,inputNeuronCount + (biasUnit ?  1 : 0)),activationFunction,biasUnit,retrainable,1);
	}

	/**
	 * Update the weights of this layer
	 * 
	 * @param thetas The weights matrix. Please note that the weight connecting input neuron i to output neuron j,
	 * w(i,j) = getClonedThetas().get(i,j)
	 * @param layerIndex The index of the layer in the containing Neural Network
	 * 
	 * @param permitFurtherRetrains Whether to permit further retrains (weight updates) after updating the weights.
	 */
	protected void updateThetas(DoubleMatrix thetas, int layerIndex, boolean permitFurtherRetrains) {

		if (!isRetrainable()) {
			throw new IllegalStateException("Layer " + (layerIndex + 1)
					+ " has already been trained and has not been set to retrainable");
		}
		if (layerIndex < 0)
		{
			throw new IllegalArgumentException("Neural network layer index must be zero or above");
		}
		if (thetas.getColumns() != outputNeuronCount || thetas.getRows() != (inputNeuronCount + (hasBiasUnit ? 1 : 0 ))) throw new IllegalArgumentException("Thetas matrix must be of dimensions "  + (inputNeuronCount + (hasBiasUnit ? 1 : 0 ) +  ":" + outputNeuronCount ));
		this.thetas = thetas;
		if (!permitFurtherRetrains) {
			this.setRetrainable(false);
		}
		applyThetasMask();

	}

	/**
	 * Generate a set of weights, initialized to normally distributed
	 * random values.
	 * 
	 * @param r The row count of the target weight matrix : ( outputNeuronCount)
	 * @param c The columns count of the target weight matrix : ( inputNeuronCount + hasBiasUnit() ? 1 : 0 )
	 * @return An initial set of weights
	 */
	protected DoubleMatrix generateInitialThetas(int c, int r) {
		DoubleMatrix initial = DoubleMatrix.randn(r, c).mul(getInitialThetaScaling());
		return initial;
	}
	
	/**
	 * Return input activations which maximise the activation of a specified output neuron
	 * 
	 * @param outputNeuronIndex The index of the output Neuron to obtain maximising input features for
	 * @return The input features which maximise the activation of the specified output Neuron
	 * 
	 */
	public double[] getOutputNeuronActivationMaximisingInputFeatures(int outputNeuronIndex) {
		int jCount = thetas.getRows() - (hasBiasUnit ? 1 : 0);
		double[] maximisingInputFeatures = new double[jCount];
		for (int j = 0; j < jCount; j++) {
			double wij = getWij(j,outputNeuronIndex );
			double sum = 0;

			for (int j2 = 0; j2 < jCount; j2++) {
				sum = sum + Math.pow(getWij( j2,outputNeuronIndex), 2);
			}
			sum = Math.sqrt(sum);
			maximisingInputFeatures[j] = wij / sum;
		}
		return maximisingInputFeatures;
	}
	
	private double getWij(int i, int j) {
		DoubleMatrix weights = thetas;
		int iInd = i + (hasBiasUnit ? 1 : 0);
		return weights.get(iInd, j);
	}
	

}
