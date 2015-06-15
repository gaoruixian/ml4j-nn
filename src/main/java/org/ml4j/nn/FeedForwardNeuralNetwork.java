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

import org.jblas.DoubleMatrix;
import org.ml4j.nn.costfunctions.CostFunction;

/**
 * A supervised FeedForwardNeuralNetwork which predicts labels from input features.
 * 
 * @author Michael Lavelle
 *
 */
public class FeedForwardNeuralNetwork extends BaseFeedForwardNeuralNetwork<FeedForwardLayer,FeedForwardNeuralNetwork> {

	private static final long serialVersionUID = 1L;

	/**
	 * FeedForwardNeuralNetwork constructor
	 * 
	 * @param layers The FeedForwardLayers contained within this NeuralNetwork
	 */
	public FeedForwardNeuralNetwork(FeedForwardLayer... layers)
	{
		super(layers);
	}
	
	/**
	 * FeedForwardNeuralNetwork constructor
	 * 
	 * @param network A network to clone
	 */
	protected FeedForwardNeuralNetwork(BaseFeedForwardNeuralNetwork<FeedForwardLayer,?> network)
	{
		super(network);
	}
	
	/**
	 * FeedForwardNeuralNetwork constructor
	 * 
	 * @param layers The FeedForwardLayers contained within this NeuralNetwork
	 */
	public FeedForwardNeuralNetwork(List<FeedForwardLayer> layers)
	{
		super(layers);
	}
	

	/**
	 * Helper function to compute the accuracy of predictions using calculated predictions
	 * predictions and correct output matrix
	 * 
	 * @param trainingDataMatrix The training examples to compute accuracy for
	 * 
	 * @param trainingLabelsMatrix The desired output labels
	 * 
	 * @return The accuracy of the network
	 */
	public double getAccuracy(DoubleMatrix trainingDataMatrix, DoubleMatrix trainingLabelsMatrix) {

		DoubleMatrix predictions = forwardPropagate(trainingDataMatrix).getPredictions();
		return computeAccuracy(predictions, trainingLabelsMatrix);

	}
	
	/**
	 * Helper function to compute the accuracy of predictions using calculated predictions
	 * predictions and correct output matrix
	 * 
	 * @param predictions The predictions
	 * 
	 * @param Y The desired output labels
	 * 
	 * @return The accuracy of the network
	 */
	protected double computeAccuracy(DoubleMatrix predictions, DoubleMatrix Y) {
		return ((predictions.mul(Y)).sum()) * 100 / Y.getRows();
	}


	@Override
	public void train(DoubleMatrix inputs, DoubleMatrix desiredOutputs, double[] lambdas, int max_iter) {
		super.train(inputs, desiredOutputs, lambdas, max_iter);
	}

	@Override
	public void train(DoubleMatrix inputs, DoubleMatrix desiredOutputs, double lambda, int max_iter) {
		super.train(inputs, desiredOutputs, lambda, max_iter);
	}

	@Override
	public void train(DoubleMatrix inputs, DoubleMatrix desiredOutputs, double lambda, CostFunction costFunction,
			int max_iter) {
		super.train(inputs, desiredOutputs, lambda, costFunction, max_iter);
	}

	@Override
	public void train(DoubleMatrix inputs, DoubleMatrix desiredOutputs, double[] lambdas, CostFunction costFunction,
			int max_iter) {
		super.train(inputs, desiredOutputs, lambdas, costFunction, max_iter);
	}
	
	@Override
	public FeedForwardNeuralNetwork dup(boolean allLayersRetrainable) {
		List<FeedForwardLayer> dupLayers = new ArrayList<FeedForwardLayer>();
		for (int i = 0; i < layers.size(); i++) {
			FeedForwardLayer layer = layers.get(i);
			
			dupLayers.add(layer.dup(allLayersRetrainable || layer.isRetrainable()));
		}
		return new FeedForwardNeuralNetwork(dupLayers);
	}
	

}
