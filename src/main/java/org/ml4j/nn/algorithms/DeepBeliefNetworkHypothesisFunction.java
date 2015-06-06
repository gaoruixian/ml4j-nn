package org.ml4j.nn.algorithms;

import org.ml4j.algorithms.HypothesisFunction;
import org.ml4j.mapping.LabeledData;
import org.ml4j.nn.DeepBeliefNetwork;

public class DeepBeliefNetworkHypothesisFunction implements HypothesisFunction<LabeledData<double[],double[]>,double[]> {

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;
	private DeepBeliefNetwork dbn;
	
	public DeepBeliefNetworkHypothesisFunction(DeepBeliefNetwork dbn) {
		this.dbn = dbn;
	}

	@Override
	public double[] predict(LabeledData<double[],double[]> labeledInputToReconstruct) {

		return dbn.generateVisibleProbabilities(labeledInputToReconstruct.getData(), labeledInputToReconstruct.getLabel());
	}


}
