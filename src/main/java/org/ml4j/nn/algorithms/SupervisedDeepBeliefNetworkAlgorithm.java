package org.ml4j.nn.algorithms;

import org.ml4j.cuda.DoubleMatrix;
import org.ml4j.nn.SupervisedDeepBeliefNetwork;

public class SupervisedDeepBeliefNetworkAlgorithm  {

	private SupervisedDeepBeliefNetwork dbn;
	
	public SupervisedDeepBeliefNetworkAlgorithm(SupervisedDeepBeliefNetwork dbn) {
		this.dbn = dbn;
	}
	
	public SupervisedDeepBeliefNetworkHypothesisFunction getHypothesisFunction(double[][] trainingDataMatrix,double[][] trainingLabelsMatrix,
			RestrictedBoltzmannMachineAlgorithmTrainingContext context) {

		dbn.trainGreedilyLayerwise(new DoubleMatrix(trainingDataMatrix),new DoubleMatrix(trainingLabelsMatrix),
				context.getMaxIterations(),context.getBatchSize(),context.getLearningRate());
		return new SupervisedDeepBeliefNetworkHypothesisFunction(dbn);
	
	}

}
