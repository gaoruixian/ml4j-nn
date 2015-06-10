package org.ml4j.nn.algorithms;

import org.jblas.DoubleMatrix;
import org.ml4j.nn.SupervisedDeepBeliefNetwork;

public class SupervisedDeepBeliefNetworkAlgorithm  {

	private SupervisedDeepBeliefNetwork dbn;
	private int miniBatchSize;
	
	public SupervisedDeepBeliefNetworkAlgorithm(SupervisedDeepBeliefNetwork dbn,int miniBatchSize) {
		this.dbn = dbn;
		this.miniBatchSize = miniBatchSize;
	}
	
	public SupervisedDeepBeliefNetworkHypothesisFunction getHypothesisFunction(double[][] trainingDataMatrix,double[][] trainingLabelsMatrix,
			RestrictedBoltzmannMachineAlgorithmTrainingContext context) {

		dbn.trainGreedilyLayerwise(new DoubleMatrix(trainingDataMatrix),new DoubleMatrix(trainingLabelsMatrix),
				context.getMaxIterations(),miniBatchSize,context.getLearningRate(),context.getGibbsSamples());
		return new SupervisedDeepBeliefNetworkHypothesisFunction(dbn);
	
	}

}
