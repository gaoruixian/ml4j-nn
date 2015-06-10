package org.ml4j.nn.algorithms;

public class RestrictedBoltzmannMachineAlgorithmTrainingContext {

	private int maxIterations;
	private int batchSize;
	private double learningRate;

	public RestrictedBoltzmannMachineAlgorithmTrainingContext(int batchSize,int maxIterations,double learningRate)
	{
		this.maxIterations = maxIterations;
		this.batchSize = batchSize;
		this.learningRate = learningRate;
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
	
	
	
}
