package org.ml4j.nn;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

import org.jblas.DoubleMatrix;
import org.ml4j.nn.activationfunctions.DifferentiableActivationFunction;
import org.ml4j.nn.activationfunctions.SegmentedActivationFunction;
import org.ml4j.nn.algorithms.RestrictedBoltzmannMachineAlgorithm;
import org.ml4j.nn.algorithms.RestrictedBoltzmannMachineAlgorithmTrainingContext;
import org.ml4j.nn.algorithms.RestrictedBoltzmannMachineHypothesisFunction;

public class SupervisedDeepBeliefNetwork extends DeepBeliefNetwork<SupervisedDeepBeliefNetwork> implements Serializable  {

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;
	
	private RestrictedBoltzmannMachineStack  unsupervisedRbmStack;
	private RestrictedBoltzmannMachine supervisedRbm;
	
	/*
	private List<RestrictedBoltzmannMachine> getRestrictedBoltzmannMachineStack()
	{
		return rbmStack;
	}
	
	public SupervisedDeepBeliefNetwork(RestrictedBoltzmannMachine...rbms) {
		super(getLayers(rbms));
		this.rbmStack = Arrays.asList(rbms);
	}
	*/
	
	public SupervisedDeepBeliefNetwork(RestrictedBoltzmannMachineStack unsupervisedRbmStack,RestrictedBoltzmannMachine supervisedRbm) {
		super(unsupervisedRbmStack,supervisedRbm);
		this.unsupervisedRbmStack = unsupervisedRbmStack;
		this.supervisedRbm = supervisedRbm;
	}
	

	
	public FeedForwardNeuralNetwork createFeedForwardNeuralNetwork()
	{
		List<FeedForwardLayer> feedForwardLayers = new ArrayList<FeedForwardLayer>();
		for (int i = 0; i < getNumberOfLayers() - 1; i++)
		{
			RestrictedBoltzmannLayer layer = getLayers().get(i);
			feedForwardLayers.add(layer.createVisibleToHiddenFeedForwardLayer());
		}
		RestrictedBoltzmannLayer finalLayer = getLayers().get(getNumberOfLayers() -1);

		int inputsLength = supervisedRbm.getLayer().getVisibleNeuronCount() + 1;
		
		int labelsLength = inputsLength - unsupervisedRbmStack.get(unsupervisedRbmStack.size() - 1).getLayer().getHiddenNeuronCount() - 1;

		int columns = inputsLength - labelsLength;
		int[] cols = new int[columns];
		cols[0] = 0;
		for (int i = 1; i < columns; i++)
		{
			cols[i] = i + labelsLength;
		}
		int[] cols2 = new int[labelsLength];
		for (int i = 0; i < labelsLength; i++)
		{
			cols2[i] = i+1;
		}
		
		DoubleMatrix thets1 =  finalLayer.getClonedThetas().transpose().getColumns(cols).transpose();
		DoubleMatrix thets2 =  finalLayer.getClonedThetas().getRows(cols2);
		FeedForwardLayer ff1 = new FeedForwardLayer(finalLayer.getVisibleNeuronCount() - labelsLength,finalLayer.getHiddenNeuronCount() + 1, thets1.transpose(),(DifferentiableActivationFunction) finalLayer.hiddenActivationFunction,true,true);
		FeedForwardLayer ff2 = new FeedForwardLayer(finalLayer.getHiddenNeuronCount() + 1,labelsLength,thets2,(DifferentiableActivationFunction)((SegmentedActivationFunction)(finalLayer.visibleActivationFunction)).getActivationFunctions()[0],false,true);
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
		
		inputs = RestrictedBoltzmannLayer.removeInterceptColumn(supervisedRbm.getCurrentVisibleStates()).toArray();
		
		
		
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
			if ( i != reversedStack.size() -1)
			{
				inputs = rbm.decodeToBinary(inputs);
			}
			i++;
		}
		inputs = unsupervisedRbmStack.get(0).decodeToProbabilities(new DoubleMatrix(new double[][] {inputs})).toArray();
		
		return inputs;
	}

	public void trainGreedilyLayerwise(DoubleMatrix inputs,DoubleMatrix labels, int max_iter,int miniBatchSize,double learningRate) {
		DoubleMatrix currentInputs = inputs;
		for (RestrictedBoltzmannMachine rbm : unsupervisedRbmStack)
		{
			RestrictedBoltzmannMachineAlgorithm alg = new RestrictedBoltzmannMachineAlgorithm(rbm,miniBatchSize);
			RestrictedBoltzmannMachineAlgorithmTrainingContext context = new RestrictedBoltzmannMachineAlgorithmTrainingContext(max_iter,miniBatchSize,learningRate);
			RestrictedBoltzmannMachineHypothesisFunction hyp = alg.getHypothesisFunction(currentInputs.toArray2(), context);
			currentInputs= new DoubleMatrix(hyp.sampleHiddenFromVisible(currentInputs.toArray2()));
		}	
		currentInputs = DoubleMatrix.concatHorizontally(labels, currentInputs);

		RestrictedBoltzmannMachineAlgorithm alg = new RestrictedBoltzmannMachineAlgorithm(supervisedRbm,miniBatchSize);
		RestrictedBoltzmannMachineAlgorithmTrainingContext context = new RestrictedBoltzmannMachineAlgorithmTrainingContext(max_iter,miniBatchSize,learningRate);
		RestrictedBoltzmannMachineHypothesisFunction hyp = alg.getHypothesisFunction(currentInputs.toArray2(), context);
		currentInputs= new DoubleMatrix(hyp.sampleHiddenFromVisible(currentInputs.toArray2()));
	}
	

	
}
