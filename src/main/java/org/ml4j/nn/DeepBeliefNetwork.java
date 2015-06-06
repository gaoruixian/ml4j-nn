package org.ml4j.nn;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import org.jblas.DoubleMatrix;
import org.ml4j.nn.algorithms.RestrictedBoltzmannMachineAlgorithm;
import org.ml4j.nn.algorithms.RestrictedBoltzmannMachineAlgorithmTrainingContext;
import org.ml4j.nn.algorithms.RestrictedBoltzmannMachineHypothesisFunction;

public class DeepBeliefNetwork implements Serializable  {

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;
	
	private List<RestrictedBoltzmannMachine>  rbmStack;
	
	private List<RestrictedBoltzmannMachine> getRestrictedBoltzmannMachineStack()
	{
		return rbmStack;
	}
	
	public DeepBeliefNetwork(RestrictedBoltzmannMachine...rbms) {
		this.rbmStack = Arrays.asList(rbms);
	}
	
	public double[] generateVisibleProbabilities(double[] visibleUnits,double[] labels) {
	
	
		double[] inputs = visibleUnits;
		
		for (int i = 0; i < rbmStack.size() - 1; i++)
		{
			RestrictedBoltzmannMachine rbm = rbmStack.get(i);
			inputs = rbm.encodeToBinary(inputs);
		}
		
		RestrictedBoltzmannMachine finalRbm = rbmStack.get(rbmStack.size() -1);
		
		inputs = DoubleMatrix.concatHorizontally(new DoubleMatrix(new double[][] {labels}), new DoubleMatrix(new double[][] {inputs})).toArray();
		
		int gibbsSamples = 100;
		for (int i = 0; i <gibbsSamples; i++)
		{
			DoubleMatrix reconstructionWithIntercept = finalRbm.pushData(new DoubleMatrix(new double[][] {inputs}));
			finalRbm.pushReconstruction(reconstructionWithIntercept);
		}
		
		inputs = RestrictedBoltzmannLayer.removeInterceptColumn(finalRbm.getCurrentVisibleStates()).toArray();
		
		List<RestrictedBoltzmannMachine> reversedStack  = new ArrayList<RestrictedBoltzmannMachine>();
		reversedStack.addAll(rbmStack);
		Collections.reverse(reversedStack);
		
		int columns = inputs.length - labels.length;
		int[] cols = new int[columns];
		for (int i = 0; i < columns; i++)
		{
			cols[i] = i + labels.length;
		}
		
		inputs = new DoubleMatrix(new double[][] {inputs}).getColumns(cols).toArray();
		
		for (int i = 1; i < reversedStack.size() - 1; i++)
		{
			RestrictedBoltzmannMachine rbm = reversedStack.get(i);
			inputs = rbm.decodeToBinary(inputs);
		}
		inputs = rbmStack.get(0).decodeToProbabilities(new DoubleMatrix(new double[][] {inputs})).toArray();
		
		return inputs;
	}

	public void trainGreedilyLayerwise(DoubleMatrix inputs,DoubleMatrix labels, int max_iter,int miniBatchSize,double learningRate) {
		DoubleMatrix currentInputs = inputs;
		for (int i = 0; i < getRestrictedBoltzmannMachineStack().size();i++)
		{
			RestrictedBoltzmannMachine rbm = getRestrictedBoltzmannMachineStack().get(i);
			RestrictedBoltzmannMachineAlgorithm alg = new RestrictedBoltzmannMachineAlgorithm(rbm,miniBatchSize);
			RestrictedBoltzmannMachineAlgorithmTrainingContext context = new RestrictedBoltzmannMachineAlgorithmTrainingContext(max_iter,miniBatchSize,learningRate);
			RestrictedBoltzmannMachineHypothesisFunction hyp = alg.getHypothesisFunction(currentInputs.toArray2(), context);
			currentInputs= new DoubleMatrix(hyp.sampleHiddenFromVisible(currentInputs.toArray2()));
			if (i == getRestrictedBoltzmannMachineStack().size() - 2)
			{
				currentInputs = DoubleMatrix.concatHorizontally(labels, currentInputs);
			}
		
		}	
	}
	

	
}
