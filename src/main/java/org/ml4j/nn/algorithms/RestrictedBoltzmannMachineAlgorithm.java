package org.ml4j.nn.algorithms;

import org.jblas.DoubleMatrix;
import org.ml4j.nn.RestrictedBoltzmannMachine;

public class RestrictedBoltzmannMachineAlgorithm {

	private RestrictedBoltzmannMachine rbm;
	
	public RestrictedBoltzmannMachineAlgorithm(RestrictedBoltzmannMachine rbm) {
		this.rbm = rbm;
	}
	
	public RestrictedBoltzmannMachineHypothesisFunction getHypothesisFunction(double[][] trainingDataMatrix,
			NeuralNetworkAlgorithmTrainingContext context) {

		rbm.train(new DoubleMatrix(trainingDataMatrix),
				context.getMaxIterations());
		
		
		return new RestrictedBoltzmannMachineHypothesisFunction(rbm);
	
	}

}
