package org.ml4j.nn.algorithms;

import org.ml4j.cuda.DoubleMatrix;
import org.ml4j.nn.UnsupervisedDeepBeliefNetwork;

public class UnsupervisedDeepBeliefNetworkAlgorithm  {

	private UnsupervisedDeepBeliefNetwork dbn;
	private int reconstructionGibbsSamples;
	
	public UnsupervisedDeepBeliefNetworkAlgorithm(UnsupervisedDeepBeliefNetwork dbn,int reconstructionGibbsSamples) {
		this.dbn = dbn;
		this.reconstructionGibbsSamples = reconstructionGibbsSamples;
	}
	
	public UnsupervisedDeepBeliefNetworkHypothesisFunction getHypothesisFunction(double[][] trainingDataMatrix,
			RestrictedBoltzmannMachineAlgorithmTrainingContext context) {

		dbn.trainGreedilyLayerwise(new DoubleMatrix(trainingDataMatrix),
				context.getMaxIterations(),context.getBatchSize(),context.getLearningRate());
		return new UnsupervisedDeepBeliefNetworkHypothesisFunction(dbn,reconstructionGibbsSamples);
	
	}

}
