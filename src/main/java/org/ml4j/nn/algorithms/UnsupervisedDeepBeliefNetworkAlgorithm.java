package org.ml4j.nn.algorithms;

import org.jblas.DoubleMatrix;
import org.ml4j.nn.UnsupervisedDeepBeliefNetwork;

public class UnsupervisedDeepBeliefNetworkAlgorithm  {

	private UnsupervisedDeepBeliefNetwork dbn;
	private int miniBatchSize;
	
	public UnsupervisedDeepBeliefNetworkAlgorithm(UnsupervisedDeepBeliefNetwork dbn,int miniBatchSize) {
		this.dbn = dbn;
		this.miniBatchSize = miniBatchSize;
	}
	
	public UnsupervisedDeepBeliefNetworkHypothesisFunction getHypothesisFunction(double[][] trainingDataMatrix,
			RestrictedBoltzmannMachineAlgorithmTrainingContext context) {

		dbn.trainGreedilyLayerwise(new DoubleMatrix(trainingDataMatrix),
				context.getMaxIterations(),miniBatchSize,context.getLearningRate(),context.getGibbsSamples());
		return new UnsupervisedDeepBeliefNetworkHypothesisFunction(dbn,context.getGibbsSamples());
	
	}

}
