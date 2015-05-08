package org.ml4j.nn.algorithms;

import org.jblas.DoubleMatrix;
import org.ml4j.algorithms.HypothesisFunction;
import org.ml4j.nn.AutoEncoder;

public class AutoEncoderHypothesisFunction implements HypothesisFunction<double[], double[]> {

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;
	
	private AutoEncoder autoEncoder;

	public AutoEncoderHypothesisFunction(AutoEncoder autoEncoder) {
		this.autoEncoder = autoEncoder;
	}
	
	public double[][] encodeToLayer(double[][] numericFeaturesMatrix,int toLayerIndex) {
		return autoEncoder.encodeToLayer(numericFeaturesMatrix, toLayerIndex);
	}
	
	public double[][] decodeFromLayer(double[][] numericFeaturesMatrix,int fromLayerIndex) {
		return autoEncoder.decodeFromLayer(numericFeaturesMatrix, fromLayerIndex);
	}
	
	public double[] decodeFromLayer(double[] encodedFeatures,int fromLayerIndex) {
		return autoEncoder.decodeFromLayer(encodedFeatures, fromLayerIndex);

	}
	public double[] encodeToLayer(double[] numericFeatures,int toLayer) {
		return autoEncoder.encodeToLayer(numericFeatures, toLayer);
	}
	
	@Override
	public double[] predict(double[] inputToReconstruct) {

		DoubleMatrix inputs = new DoubleMatrix(inputToReconstruct).transpose();
		double[] predictions = autoEncoder.forwardPropagate(inputs).getOutputs().toArray();
		return predictions;
	}

	


}
