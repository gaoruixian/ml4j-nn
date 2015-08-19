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
import java.util.List;

import org.ml4j.DoubleMatrix;
import org.ml4j.nn.activationfunctions.DifferentiableActivationFunction;
import org.ml4j.nn.algorithms.RestrictedBoltzmannMachineAlgorithm;
import org.ml4j.nn.algorithms.RestrictedBoltzmannMachineAlgorithmTrainingContext;
import org.ml4j.nn.algorithms.RestrictedBoltzmannMachineHypothesisFunction;
import org.ml4j.nn.util.NeuralNetworkUtils;

/**
 * A supervised Deep Belief Network that consists of a stack of unsupervised Restricted Boltzmann Machines, followed
 * by a final supervised Restricted Boltzmann Machine.
 * 
 * @author Michael Lavelle
 *
 */
public class SupervisedDeepBeliefNetwork extends DeepBeliefNetwork<SupervisedDeepBeliefNetwork> implements Serializable  {

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;
	
	private RestrictedBoltzmannMachineStack  unsupervisedRbmStack;
	private RestrictedBoltzmannMachine supervisedRbm;
	
	
	public SupervisedDeepBeliefNetwork(RestrictedBoltzmannMachineStack unsupervisedRbmStack,RestrictedBoltzmannMachine supervisedRbm) {
		super(unsupervisedRbmStack,supervisedRbm);
		this.unsupervisedRbmStack = unsupervisedRbmStack;
		this.supervisedRbm = supervisedRbm;
	}
	
	public FeedForwardNeuralNetwork createFeedForwardNeuralNetwork(DifferentiableActivationFunction supervisedActivationFunction)
	{
		List<FeedForwardLayer> feedForwardLayers = new ArrayList<FeedForwardLayer>();
		for (int i = 0; i < getNumberOfLayers() - 1; i++)
		{
			RestrictedBoltzmannLayer layer = getLayers().get(i);
			feedForwardLayers.add(layer.createVisibleToHiddenFeedForwardLayer());
		}
		RestrictedBoltzmannLayer finalLayer = getLayers().get(getNumberOfLayers() -1);

		int inputsLength = supervisedRbm.getLayer().getVisibleNeuronCount() + 1;
		

		int labelsLength = inputsLength - unsupervisedRbmStack.getFinalLayer().getHiddenNeuronCount() - 1;

		int rowCount = inputsLength - labelsLength;
		int[] rows = new int[rowCount];
		rows[0] = 0;
		for (int i = 1; i < rowCount; i++)
		{
			rows[i] = i + labelsLength;
		}
		int[] rows2 = new int[labelsLength];
		for (int i = 0; i < labelsLength; i++)
		{
			rows2[i] = i+1;
		}
		
		DoubleMatrix thets1 =  finalLayer.getClonedThetas().getRows(rows);
		
		DoubleMatrix thets2 =  finalLayer.getClonedThetas().transpose().getColumns(rows2);
		
		
		
		FeedForwardLayer ff1 = new FeedForwardLayer(finalLayer.getVisibleNeuronCount() - labelsLength,finalLayer.getHiddenNeuronCount() + 1, thets1,(DifferentiableActivationFunction) finalLayer.hiddenActivationFunction,true,true);
		FeedForwardLayer ff2 = new FeedForwardLayer(finalLayer.getHiddenNeuronCount() + 1,labelsLength,thets2,supervisedActivationFunction,false,true);
		feedForwardLayers.add(ff1);
		feedForwardLayers.add(ff2);

		return new FeedForwardNeuralNetwork(feedForwardLayers);
	}
	

	public double[] generateVisibleProbabilities(double[] visibleUnits,double[] labels) {
	
	
		double[] inputs = visibleUnits;
		
		for (RestrictedBoltzmannMachine rbm : unsupervisedRbmStack)
		{
			inputs = rbm.encodeToBinary(inputs);
		}
		
		
		inputs = DoubleMatrix.concatHorizontally(new DoubleMatrix(new double[][] {labels}), new DoubleMatrix(new double[][] {inputs})).toArray();
		
		int gibbsSamples = 100;
		for (int i = 0; i <gibbsSamples; i++)
		{
			DoubleMatrix reconstructionWithIntercept = supervisedRbm.pushData(new DoubleMatrix(new double[][] {inputs}));
			supervisedRbm.pushReconstruction(reconstructionWithIntercept);
		}
		
		inputs = NeuralNetworkUtils.removeInterceptColumn(supervisedRbm.getCurrentVisibleStates()).toArray();
		
		
		
		int columns = inputs.length - labels.length;
		int[] cols = new int[columns];
		for (int i = 0; i < columns; i++)
		{
			cols[i] = i + labels.length;
		}
		
		
		RestrictedBoltzmannMachineStack reversedStack = unsupervisedRbmStack.reverse();
		
		inputs = new DoubleMatrix(new double[][] {inputs}).getColumns(cols).toArray();
		int i = 0;
		for (RestrictedBoltzmannMachine rbm : reversedStack)
		{
			if ( i != reversedStack.getNumberOfLayers() -1)
			{
				inputs = rbm.decodeToBinary(inputs);
			}
			i++;
		}
		inputs = unsupervisedRbmStack.getFirstRestrictedBoltzmannMachine().decodeToProbabilities(new DoubleMatrix(new double[][] {inputs})).toArray();
		
		return inputs;
	}

	public void trainGreedilyLayerwise(DoubleMatrix inputs,DoubleMatrix labels, int max_iter,int miniBatchSize,double learningRate) {
		DoubleMatrix currentInputs = inputs;
		for (RestrictedBoltzmannMachine rbm : unsupervisedRbmStack)
		{
			RestrictedBoltzmannMachineAlgorithm alg = new RestrictedBoltzmannMachineAlgorithm(rbm,miniBatchSize);
			RestrictedBoltzmannMachineAlgorithmTrainingContext context = new RestrictedBoltzmannMachineAlgorithmTrainingContext(miniBatchSize,max_iter,learningRate);
			RestrictedBoltzmannMachineHypothesisFunction hyp = alg.getHypothesisFunction(currentInputs.toArray2(), context);
			currentInputs= new DoubleMatrix(hyp.sampleHiddenFromVisible(currentInputs.toArray2()));
		}	
		currentInputs = DoubleMatrix.concatHorizontally(labels, currentInputs);

		RestrictedBoltzmannMachineAlgorithm alg = new RestrictedBoltzmannMachineAlgorithm(supervisedRbm,miniBatchSize);
		RestrictedBoltzmannMachineAlgorithmTrainingContext context = new RestrictedBoltzmannMachineAlgorithmTrainingContext(miniBatchSize,max_iter,learningRate);
		RestrictedBoltzmannMachineHypothesisFunction hyp = alg.getHypothesisFunction(currentInputs.toArray2(), context);
		currentInputs= new DoubleMatrix(hyp.sampleHiddenFromVisible(currentInputs.toArray2()));
	}
	

	
}
