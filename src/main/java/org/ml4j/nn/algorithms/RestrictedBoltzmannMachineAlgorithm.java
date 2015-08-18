package org.ml4j.nn.algorithms;

import org.ml4j.DoubleMatrix;
import org.ml4j.nn.RestrictedBoltzmannMachine;

public class RestrictedBoltzmannMachineAlgorithm {

	private RestrictedBoltzmannMachine rbm;
	private int miniBatchSize;
	
	public RestrictedBoltzmannMachineAlgorithm(RestrictedBoltzmannMachine rbm,int miniBatchSize) {
		this.rbm = rbm;
		this.miniBatchSize = miniBatchSize;
	}
	
	public RestrictedBoltzmannMachineHypothesisFunction getHypothesisFunction(double[][] trainingDataMatrix,
			RestrictedBoltzmannMachineAlgorithmTrainingContext context) {

		rbm.train(new DoubleMatrix(trainingDataMatrix),
				context.getMaxIterations(),miniBatchSize,context.getLearningRate());
		
		
		return new RestrictedBoltzmannMachineHypothesisFunction(rbm);
	
	}

}
