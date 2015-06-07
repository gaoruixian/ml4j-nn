package org.ml4j.nn.algorithms;

import org.jblas.DoubleMatrix;
import org.ml4j.nn.DeepBeliefNetwork;

public class DeepBeliefNetworkAlgorithm  {

	private DeepBeliefNetwork dbn;
	private int miniBatchSize;
	
	public DeepBeliefNetworkAlgorithm(DeepBeliefNetwork dbn,int miniBatchSize) {
		this.dbn = dbn;
		this.miniBatchSize = miniBatchSize;
	}
	
	public DeepBeliefNetworkHypothesisFunction getHypothesisFunction(double[][] trainingDataMatrix,double[][] trainingLabelsMatrix,
			RestrictedBoltzmannMachineAlgorithmTrainingContext context) {

		dbn.trainGreedilyLayerwise(new DoubleMatrix(trainingDataMatrix),new DoubleMatrix(trainingLabelsMatrix),
				context.getMaxIterations(),miniBatchSize,context.getLearningRate());
		return new DeepBeliefNetworkHypothesisFunction(dbn);
	
	}

}
