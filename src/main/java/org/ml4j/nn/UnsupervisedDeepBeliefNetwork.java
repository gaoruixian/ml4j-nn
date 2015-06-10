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

import org.jblas.DoubleMatrix;
import org.ml4j.nn.util.NeuralNetworkUtils;

/**
 * 
 * 
 * An unsupervised Deep Belief Network that consists of a stack of unsupervised Restricted Boltzmann Machines
 * 
 * @author Michael Lavelle
 *
 */
public class UnsupervisedDeepBeliefNetwork extends DeepBeliefNetwork<UnsupervisedDeepBeliefNetwork> implements Serializable  {

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;
	
	
	
	public UnsupervisedDeepBeliefNetwork(RestrictedBoltzmannMachineStack unsupervisedRbmStack) {
		super(unsupervisedRbmStack);
	}
	
	public UnsupervisedDeepBeliefNetwork(RestrictedBoltzmannLayer... layers) {
		super(new RestrictedBoltzmannMachineStack(layers));
	}
	
	
	public StackedAutoEncoder createStackedAutoEncoder()
	{
		return unsupervisedRbmStack.createStackedAutoEncoder();
	}
	
	
	public FeedForwardNeuralNetwork createFeedForwardNeuralNetwork(FeedForwardLayer... additionalLayers)
	{
		return unsupervisedRbmStack.createFeedForwardNeuralNetwork(additionalLayers);
	}
	
	public FeedForwardNeuralNetwork createFeedForwardNeuralNetwork()
	{
		return unsupervisedRbmStack.createFeedForwardNeuralNetwork();
	}
	
	/**
	 * Generate probabilities of activations of each of the visible units,
	 * after
	 * 
	 * @param visibleUnits
	 * @return
	 */
	public double[] generateVisibleProbabilities(double[] visibleUnits,int gibbsSamples) {
	
	
		double[] inputs = visibleUnits;
		int in = 0;
		for (RestrictedBoltzmannMachine rbm : unsupervisedRbmStack)
		{
			if (in != unsupervisedRbmStack.getNumberOfLayers() - 1)
			{
			inputs = rbm.encodeToBinary(inputs);
			}
			in++;
		}
		
		RestrictedBoltzmannMachine finalRbm = unsupervisedRbmStack.getFinalRestrictedBoltzmannMachine();
		
		for (int i = 0; i <gibbsSamples; i++)
		{
			DoubleMatrix reconstructionWithIntercept = finalRbm.pushData(new DoubleMatrix(new double[][] {inputs}));
			finalRbm.pushReconstruction(reconstructionWithIntercept);
		}
		
		inputs = NeuralNetworkUtils.removeInterceptColumn(finalRbm.getCurrentVisibleStates()).toArray();
		
		
		RestrictedBoltzmannMachineStack reversedStack = unsupervisedRbmStack.reverse();
		
		inputs = new DoubleMatrix(new double[][] {inputs}).toArray();
		int ind = 0;
		for (RestrictedBoltzmannMachine rbm : reversedStack)
		{
			if (ind != 0)
			{
			inputs = rbm.decodeToBinary(inputs);
			}
			ind++;
		}
		inputs = unsupervisedRbmStack.getFirstRestrictedBoltzmannMachine().decodeToProbabilities(new DoubleMatrix(new double[][] {inputs})).toArray();
		
		return inputs;
	}

	public void trainGreedilyLayerwise(DoubleMatrix inputs, int max_iter,int miniBatchSize,double learningRate) {
		unsupervisedRbmStack.trainGreedilyLayerwise(inputs, max_iter, miniBatchSize, learningRate);
	}
	
	
	
}
