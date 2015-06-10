package org.ml4j.nn.algorithms;

public class RestrictedBoltzmannMachineAlgorithmTrainingContext {

	private int maxIterations;
	private int batchSize;
	private double learningRate;
	private int gibbsSamples;

	public RestrictedBoltzmannMachineAlgorithmTrainingContext(int batchSize,int maxIterations,double learningRate,int gibbsSamples)
	{
		this.maxIterations = maxIterations;
		this.batchSize = batchSize;
		this.learningRate = learningRate;
		this.gibbsSamples = gibbsSamples;
	}

	public int getBatchSize() {
		return batchSize;
	}

	public int getMaxIterations() {
		return maxIterations;
	}

	public double getLearningRate() {
		return learningRate;
	}

	public int getGibbsSamples() {
		return gibbsSamples;
	}
	
	
	
	
	
}
