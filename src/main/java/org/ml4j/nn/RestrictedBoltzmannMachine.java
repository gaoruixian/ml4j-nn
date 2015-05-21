package org.ml4j.nn;

import java.io.Serializable;

import org.jblas.DoubleMatrix;

public class RestrictedBoltzmannMachine implements Serializable {

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;
	private RestrictedBoltzmannLayer layer;

	private DoubleMatrix currentHiddenStates;
	private DoubleMatrix currentVisibleStates;

	public RestrictedBoltzmannMachine(RestrictedBoltzmannLayer layer) {
		this.layer = layer;
	}

	public double[] encodeToProbabilities(double[] visibleUnits) {
		return layer.getHiddenUnitProbabilities(visibleUnits);
	}

	public double[] decode(double[] hiddenUnits) {
		return layer.getVisibleUnitProbabilities(hiddenUnits);
	}

	public DoubleMatrix encodeToProbabilities(DoubleMatrix visibleUnits) {
		return layer.getHiddenUnitProbabilities(visibleUnits);
	}

	public DoubleMatrix decodeToProbabilities(DoubleMatrix hiddenUnits) {
		return layer.getVisibleUnitProbabilities(hiddenUnits);
	}

	public double[] generateVisibleBinaries() {
		return getBinarySample(generateVisibleProbabilities());
	}

	private double[] getBinarySample(double[] probs) {
		DoubleMatrix rand = DoubleMatrix.rand(1, probs.length);
		DoubleMatrix res = new DoubleMatrix(new double[][] { probs }).sub(rand);
		DoubleMatrix result = new DoubleMatrix(1, res.getColumns());
		for (int i = 0; i < result.getColumns(); i++) {
			if (res.get(0, i) > 0) {
				result.put(0, i, 1);
			}
		}
		return result.toArray();
	}

	public double[] generateVisibleProbabilities() {
		double[] randomVisibleUnits = new double[layer.getVisibleNeuronCount()];
		double[] probs = null;
		int cdn = 20;
		for (int i = 0; i < randomVisibleUnits.length; i++) {
			randomVisibleUnits[i] = Math.random();
		}
		DoubleMatrix visibleUnitsMatrix = new DoubleMatrix(new double[][] { randomVisibleUnits });
		this.currentVisibleStates = null;
		this.currentHiddenStates = null;
		for (int i = 0; i < cdn; i++) {
			DoubleMatrix recWithIntercept = pushData(visibleUnitsMatrix);
			pushReconstruction(recWithIntercept);
			visibleUnitsMatrix = layer.removeInterceptColumn(currentVisibleStates);
			probs = layer.getVisibleUnitProbabilities(layer.removeInterceptColumn(currentHiddenStates).toArray());
		}
		return probs;
	}

	public void train(DoubleMatrix doubleMatrix, int maxIterations) {

		double learningRate = 0.01;
		for (int l = 0; l < maxIterations; l++) {
			for (int i = 0; i < doubleMatrix.getRows(); i++) {

				DoubleMatrix reconstructionWithIntercept = pushData(doubleMatrix.getRow(i));
				DoubleMatrix positiveStatistics = getPairwiseVectorProduct(currentVisibleStates, currentHiddenStates);

				pushReconstruction(reconstructionWithIntercept);
				DoubleMatrix negativeStatistics = getPairwiseVectorProduct(currentVisibleStates, currentHiddenStates);

				DoubleMatrix delta = (positiveStatistics.sub(negativeStatistics)).mul(learningRate);

				layer.updateWithDelta(delta);
			}
		}

	}

	public double[] encodeToBinary(double[] visibleUnits) {
		return layer.getHiddenUnitSample(visibleUnits);
	}

	public double[] decodeToBinary(double[] hiddenUnits) {
		return layer.getVisibleUnitSample(hiddenUnits);
	}

	protected DoubleMatrix pushData(DoubleMatrix data) {
		this.currentVisibleStates = layer.addInterceptColumn(data);
		this.currentHiddenStates = layer.getHSampleGivenV(currentVisibleStates);
		return layer.getProbVGivenH(currentHiddenStates);
	}

	protected void pushReconstruction(DoubleMatrix reconstructionWithIntercept) {
		this.currentVisibleStates = reconstructionWithIntercept;
		this.currentHiddenStates = layer.getProbHGivenV(reconstructionWithIntercept);
	}

	public DoubleMatrix getPairwiseVectorProduct(DoubleMatrix vector1, DoubleMatrix vector2) {
		DoubleMatrix result = new DoubleMatrix(vector1.getColumns(), vector2.getColumns());
		for (int i = 0; i < vector1.getColumns(); i++) {
			for (int j = 0; j < vector2.getColumns(); j++) {
				result.put(i, j, vector1.get(0, i) * vector2.get(0, j));
			}
		}
		return result;
	}

}
