package org.ml4j.nn;

import org.jblas.DoubleMatrix;

public class NeuralNetworkLayerErrorGradient {

	private NeuralNetworkLayer layer;
	private DoubleMatrix delta;
	private int m;
	private double lambda;
	private DoubleMatrix inputActivations;
	private DoubleMatrix thetas;

	public NeuralNetworkLayerErrorGradient(NeuralNetworkLayer layer, DoubleMatrix thetas, DoubleMatrix delta, int m,
			double lambda, DoubleMatrix inputActivations) {
		this.layer = layer;
		this.m = m;
		this.delta = delta;
		this.lambda = lambda;
		this.thetas = thetas;
		this.inputActivations = inputActivations;
	}
	
	public NeuralNetworkLayer getLayer() {
		return layer;
	}



	public DoubleMatrix getDELTA() {
		return delta.mmul(inputActivations);
	}

	public DoubleMatrix getErrorGradient() {
		DoubleMatrix currentTheta = thetas;
		DoubleMatrix modTheta = new DoubleMatrix().copy(currentTheta);
		modTheta.putColumn(0, DoubleMatrix.zeros(currentTheta.getRows(), 1));

		return getDELTA().div(m).add(modTheta.mul(lambda / m));
	}
}
