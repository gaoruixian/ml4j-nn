package org.ml4j.nn;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import org.jblas.DoubleMatrix;
import org.ml4j.nn.algorithms.RestrictedBoltzmannMachineAlgorithm;
import org.ml4j.nn.algorithms.RestrictedBoltzmannMachineAlgorithmTrainingContext;
import org.ml4j.nn.algorithms.RestrictedBoltzmannMachineHypothesisFunction;

public class RestrictedBoltzmannMachineStack extends ArrayList<RestrictedBoltzmannMachine> {

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;
	
	
	public RestrictedBoltzmannMachineStack reverse()
	{
		RestrictedBoltzmannMachineStack stack = new RestrictedBoltzmannMachineStack();
		stack.addAll(this);
		Collections.reverse(stack);
		return stack;
	}
	
	
	public RestrictedBoltzmannMachineStack(RestrictedBoltzmannMachine... rbms)
	{
		this.addAll(Arrays.asList(rbms));
	}
	
	public void trainGreedilyLayerwise(DoubleMatrix inputs, int max_iter,int miniBatchSize,double learningRate) {
		DoubleMatrix currentInputs = inputs;
		for (RestrictedBoltzmannMachine rbm : this)
		{
			RestrictedBoltzmannMachineAlgorithm alg = new RestrictedBoltzmannMachineAlgorithm(rbm,miniBatchSize);
			RestrictedBoltzmannMachineAlgorithmTrainingContext context = new RestrictedBoltzmannMachineAlgorithmTrainingContext(max_iter,miniBatchSize,learningRate);
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
		return new StackedAutoEncoder(autoEncoders.toArray(new AutoEncoder[size()]));
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

}
