package org.ml4j.nn.algorithms;

import org.ml4j.algorithms.HypothesisFunction;
import org.ml4j.nn.UnsupervisedDeepBeliefNetwork;

public class UnsupervisedDeepBeliefNetworkHypothesisFunction implements HypothesisFunction<double[],double[]> {

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;
	private UnsupervisedDeepBeliefNetwork dbn;
	
	public UnsupervisedDeepBeliefNetworkHypothesisFunction(UnsupervisedDeepBeliefNetwork dbn) {
		this.dbn = dbn;
	}

	@Override
	public double[] predict(double[] inputToReconstruct) {

		return dbn.generateVisibleProbabilities(inputToReconstruct);
	}


}
