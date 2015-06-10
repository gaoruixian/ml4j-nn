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
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Iterator;
import java.util.List;

import org.jblas.DoubleMatrix;
import org.ml4j.nn.algorithms.RestrictedBoltzmannMachineAlgorithm;
import org.ml4j.nn.algorithms.RestrictedBoltzmannMachineAlgorithmTrainingContext;
import org.ml4j.nn.algorithms.RestrictedBoltzmannMachineHypothesisFunction;

public class RestrictedBoltzmannMachineStack implements Iterable<RestrictedBoltzmannMachine>,Serializable {

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;
	
	private List<RestrictedBoltzmannMachine> rbms;
	
	public int getNumberOfLayers()
	{
		return rbms.size();
	}
	
	public RestrictedBoltzmannLayer getFinalLayer()
	{
		return getFinalRestrictedBoltzmannMachine().getLayer();
	}
	
	public RestrictedBoltzmannMachine getFinalRestrictedBoltzmannMachine()
	{
		return rbms.get(rbms.size() -1);
	}
	
	public List<RestrictedBoltzmannLayer> getLayers()
	{
		List<RestrictedBoltzmannLayer> layers = new ArrayList<RestrictedBoltzmannLayer>();
		for (RestrictedBoltzmannMachine rbm : rbms)
		{
			layers.add(rbm.getLayer());
		}
		return layers;
	}
	
	public RestrictedBoltzmannMachine getFirstRestrictedBoltzmannMachine()
	{
		return rbms.get(0);
	}
	
	public RestrictedBoltzmannMachineStack reverse()
	{
		List<RestrictedBoltzmannMachine> reversed = new ArrayList<RestrictedBoltzmannMachine>();
		reversed.addAll(reversed);
		return new RestrictedBoltzmannMachineStack(reversed);
	}
	
	
	public RestrictedBoltzmannMachineStack(List<RestrictedBoltzmannMachine> rbms)
	{
		this.rbms = rbms;
	}
	
	
	public RestrictedBoltzmannMachineStack(RestrictedBoltzmannMachine... rbms)
	{
		this.rbms = Arrays.asList(rbms);
	}
	
	public RestrictedBoltzmannMachineStack(RestrictedBoltzmannLayer... layers)
	{
		List<RestrictedBoltzmannMachine> rbms = new ArrayList<RestrictedBoltzmannMachine>();
		for (RestrictedBoltzmannLayer layer : layers)
		{
			rbms.add(new RestrictedBoltzmannMachine(layer));
		}
		this.rbms = rbms;
	}

	
	public void trainGreedilyLayerwise(DoubleMatrix inputs, int max_iter,int miniBatchSize,double learningRate,int gibbsSamples) {
		DoubleMatrix currentInputs = inputs;
		for (RestrictedBoltzmannMachine rbm : this)
		{
			RestrictedBoltzmannMachineAlgorithm alg = new RestrictedBoltzmannMachineAlgorithm(rbm,miniBatchSize);
			RestrictedBoltzmannMachineAlgorithmTrainingContext context = new RestrictedBoltzmannMachineAlgorithmTrainingContext(miniBatchSize,max_iter,learningRate,gibbsSamples);
			RestrictedBoltzmannMachineHypothesisFunction hyp = alg.getHypothesisFunction(currentInputs.toArray2(), context);
			currentInputs= new DoubleMatrix(hyp.sampleHiddenFromVisible(currentInputs.toArray2()));
		}	
	}
	
	public UnsupervisedDeepBeliefNetwork createUnsupervisedDeepBeliefNetwork()
	{
		return new UnsupervisedDeepBeliefNetwork(this);
	}
	
	public SupervisedDeepBeliefNetwork createUnsupervisedDeepBeliefNetwork(RestrictedBoltzmannMachine supervisedRbm)
	{
		return new SupervisedDeepBeliefNetwork(this,supervisedRbm);
	}
	
	public StackedAutoEncoder createStackedAutoEncoder()
	{
		List<AutoEncoder> autoEncoders = new ArrayList<AutoEncoder>();
		for (RestrictedBoltzmannMachine rbm : this)
		{
			autoEncoders.add(rbm.getLayer().createAutoEncoder());
		}
		return new StackedAutoEncoder(autoEncoders.toArray(new AutoEncoder[rbms.size()]));
	}
	
	
	public FeedForwardNeuralNetwork createFeedForwardNeuralNetwork(FeedForwardLayer... additionalLayers)
	{
		List<FeedForwardLayer> feedForwardLayers = new ArrayList<FeedForwardLayer>();
		for (RestrictedBoltzmannMachine rbm : this)
		{
			RestrictedBoltzmannLayer layer = rbm.getLayer();
			feedForwardLayers.add(layer.createVisibleToHiddenFeedForwardLayer());
		}

		feedForwardLayers.addAll(Arrays.asList(additionalLayers));
		return new FeedForwardNeuralNetwork(feedForwardLayers);
	}


	@Override
	public Iterator<RestrictedBoltzmannMachine> iterator() {
		return rbms.iterator();
	}

}
