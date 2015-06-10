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

public abstract class DirectedNeuralNetwork<L extends DirectedLayer<?>,N extends DirectedNeuralNetwork<L,N>> extends BaseNeuralNetwork<L,N> {

	protected DirectedNeuralNetwork(L[] layers) {
		super(layers);
	}
	
	protected DirectedNeuralNetwork(List<L> layers) {
		super(layers);
	}
	
	public ForwardPropagation forwardPropagate(double[][] inputs) {
		return forwardPropagate(new DoubleMatrix(inputs));
	}

	public ForwardPropagation forwardPropagate(double[] inputs) {
		return forwardPropagate(new DoubleMatrix(new double[][] { inputs }));
	}

	public ForwardPropagation forwardPropagate(DoubleMatrix inputs) {
		return forwardPropagateFromTo(inputs,0,getNumberOfLayers() -1);
	}
	
	public ForwardPropagation forwardPropagateFromTo(double[][] inputs,int fromLayerIndex,int toLayerIndex) {
		return forwardPropagateFromTo(new DoubleMatrix(inputs),fromLayerIndex,toLayerIndex);
	}
	
	protected ForwardPropagation forwardPropagateFromTo(double[] inputs,int fromLayerIndex,int toLayerIndex) {
		return forwardPropagateFromTo(new DoubleMatrix(new double[][] {inputs}),fromLayerIndex,toLayerIndex);
	}
	protected ForwardPropagation forwardPropagateFromTo(DoubleMatrix inputs,int fromLayerIndex,int toLayerIndex) {
		DoubleMatrix inputActivations = inputs;
		List<NeuralNetworkLayerActivation> layerActivations = new ArrayList<NeuralNetworkLayerActivation>();
		boolean start = false;
		boolean end = false;
		int index =0;
		for (DirectedLayer<?> layer : getLayers()) {
			start = start || index == fromLayerIndex;

			if (start && !end)
			{
				if (layer.hasBiasUnit)
				{
					inputActivations = DoubleMatrix.concatHorizontally(DoubleMatrix.ones(inputActivations.getRows(), 1),
							inputActivations);
				}
			NeuralNetworkLayerActivation activation = layer.forwardPropagate(inputActivations);
			layerActivations.add(activation);
			inputActivations = activation.getOutputActivations();
			}
			end = end || index == toLayerIndex;

			index++;
		}
		return new ForwardPropagation(inputActivations, layerActivations);
	}

	
	
}

