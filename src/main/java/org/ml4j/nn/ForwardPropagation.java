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

import java.util.List;

import org.jblas.DoubleMatrix;
import org.ml4j.nn.costfunctions.CostFunction;

public class ForwardPropagation {

	private DoubleMatrix outputs;
	private List<NeuralNetworkLayerActivation> activations;

	public ForwardPropagation(DoubleMatrix outputs, List<NeuralNetworkLayerActivation> activations) {
		this.outputs = outputs;
		this.activations = activations;
	}

	public List<NeuralNetworkLayerActivation> getActivations() {
		return activations;
	}

	public DoubleMatrix getOutputs() {
		return outputs;
	}
	public BackPropagation backPropagate(FeedForwardNeuralNetwork neuralNetwork,DoubleMatrix desiredOutputs,
			double[] lambdas)
	{
		return neuralNetwork.backPropagate(this, desiredOutputs,lambdas);
	}
	
	
	public DoubleMatrix getPredictions()
	{
		DoubleMatrix hypothesis = outputs;
		int [] maxIndicies= hypothesis.rowArgmaxs();
		int rows = hypothesis.getRows();
		int cols = hypothesis.getColumns();
		DoubleMatrix prediction = DoubleMatrix.zeros(rows,cols);
		for (int i = 0; i< rows; i++)
		{
			prediction.put(i,maxIndicies[i],1);
		}
		return prediction;
		
	}
	
	

	public double getCostWithRetrainableLayerRegularisation(DoubleMatrix desiredOutputs, double[] lambda,
			CostFunction cf) {

		DoubleMatrix X = activations.get(0).getInputActivations();

		int m = X.getRows();

		DoubleMatrix H = getOutputs();
		double J = cf.getCost(desiredOutputs, H);

		// Calculate regularization part of cost.
		int i = 0;
		for (NeuralNetworkLayerActivation layerActivation : getActivations()) {
			if (layerActivation.getLayer().isRetrainable()) {
				J = J + layerActivation.getRegularisationCost(m, lambda[i]);
			}
			i++;
		}

		return J;
	}

}
