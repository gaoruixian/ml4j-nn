package org.ml4j.nn.algorithms;

import org.ml4j.algorithms.HypothesisFunction;
import org.ml4j.cuda.DoubleMatrix;
import org.ml4j.nn.RestrictedBoltzmannMachine;

public class RestrictedBoltzmannMachineHypothesisFunction implements HypothesisFunction<double[],double[]> {

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;
	private RestrictedBoltzmannMachine rbm;
	
	public RestrictedBoltzmannMachineHypothesisFunction(RestrictedBoltzmannMachine rbm) {
		this.rbm = rbm;
	}

	public double[] sampleHiddenFromVisible(double[] visibleUnits) {
		return rbm.encodeToBinary(visibleUnits);
	}
	
	public double[][] sampleHiddenFromVisible(double[][] visibleUnits) {
		return rbm.encodeToBinary(visibleUnits);
	}

	public double[] sampleVisibleUnitsFromHidden(double[] hiddenUnits) {
		return rbm.decodeToBinary(hiddenUnits);
	}
	
	public double[] getVisibleProbabilitiesFromHidden(double[] hiddenUnits) {
		return rbm.decodeToProbabilities(new DoubleMatrix(new double[][] {hiddenUnits})).toArray();
	}
	
	
	@Override
	public double[] predict(double[] inputToReconstruct) {

		double[] predictions = sampleVisibleUnitsFromHidden(sampleHiddenFromVisible(inputToReconstruct));
		return predictions;
	}


}
