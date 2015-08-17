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

import java.util.ArrayList;
import java.util.List;

import org.ml4j.DoubleMatrix;
import org.ml4j.nn.activationfunctions.DifferentiableActivationFunction;
import org.ml4j.nn.costfunctions.CostFunction;

/**
 * Unsupervised feed forward Neural Network that attempts to reconstruct
 * input features after they have been mapped/compressed/encoded into alternative
 * feature domains.
 * 
 * @author Michael Lavelle
 *
 */
public class AutoEncoder extends BaseFeedForwardNeuralNetwork<FeedForwardLayer,AutoEncoder> {

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;

	public AutoEncoder(int inputNeuronCount,int hiddenNeuronCount,DifferentiableActivationFunction encodingActivationFunction,DifferentiableActivationFunction decodingActivationFunction)
	{
		this(new FeedForwardLayer(inputNeuronCount,hiddenNeuronCount,encodingActivationFunction,true),new FeedForwardLayer(hiddenNeuronCount,inputNeuronCount,decodingActivationFunction,true));
	}
	
	public AutoEncoder(FeedForwardLayer... layers)
	{
		super(layers);
		FeedForwardLayer firstLayer = layers[0];
		FeedForwardLayer lastLayer = layers[layers.length - 1];
		if (firstLayer.getInputNeuronCount() != lastLayer.getOutputNeuronCount())
		{
			throw new IllegalArgumentException("Input layer neuron count must be the same as output activations");
		}
	}  
	
	public AutoEncoder(List<FeedForwardLayer> layers)
	{
		super(layers);
		FeedForwardLayer firstLayer = getFirstLayer();
		FeedForwardLayer lastLayer = getOuterLayer();
		if (firstLayer.getInputNeuronCount() != lastLayer.getOutputNeuronCount())
		{
			throw new IllegalArgumentException("Input layer neuron count must be the same as output activations");
		}
	}  
	
	public AutoEncoder(AutoEncoder encoder)
	{
		super(encoder);
		FeedForwardLayer firstLayer = getFirstLayer();
		FeedForwardLayer lastLayer = getOuterLayer();
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
	public AutoEncoder dup(boolean allLayersRetrainable) {
		List<FeedForwardLayer> dupLayers = new ArrayList<FeedForwardLayer>();
		for (int i = 0; i < layers.size(); i++) {
			FeedForwardLayer layer = layers.get(i);
			
			dupLayers.add(layer.dup(allLayersRetrainable || layer.isRetrainable()));
		}
		return new AutoEncoder(dupLayers);
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
	
	public FeedForwardNeuralNetwork cloneAndRemoveOuterLayer()
	{
		FeedForwardLayer[] layers = new FeedForwardLayer[getNumberOfLayers() -2];
		for (int i = 0; i < layers.length;i++)
		{
			layers[i] = this.getLayers().get(i);
		}
		return new FeedForwardNeuralNetwork(layers);
	}
	
	public FeedForwardNeuralNetwork cloneAndReplaceOuterLayer(FeedForwardLayer layer)
	{
		FeedForwardLayer[] layers = new FeedForwardLayer[getNumberOfLayers() -1];
		for (int i = 0; i < layers.length;i++)
		{
			layers[i] = this.getLayers().get(i);
		}
		layers[layers.length -1] = layer;
		return new FeedForwardNeuralNetwork(layers);
	}
	
	public double[][] encodeToLayer(double[][] numericFeaturesMatrix,int toLayerIndex) {
		return forwardPropagateFromTo(numericFeaturesMatrix, 0, toLayerIndex,false).getOutputs().toArray2();
	}
	
	public double[][] decodeFromLayer(double[][] numericFeaturesMatrix,int fromLayerIndex) {
		return forwardPropagateFromTo(numericFeaturesMatrix, fromLayerIndex, getNumberOfLayers() - 1,false).getOutputs().toArray2();
	}
	
	public double[] decodeFromLayer(double[] encodedFeatures,int fromLayerIndex) {
		return forwardPropagateFromTo(encodedFeatures, fromLayerIndex, getNumberOfLayers() - 1,false).getOutputs().toArray();

	}
	public double[] encodeToLayer(double[] numericFeatures,int toLayer) {
		return forwardPropagateFromTo(numericFeatures, 0, toLayer,false).getOutputs().toArray();
	}
	
	
	
}
