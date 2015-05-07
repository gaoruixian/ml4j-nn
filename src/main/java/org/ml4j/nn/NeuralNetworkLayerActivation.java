package org.ml4j.nn;

import org.jblas.DoubleMatrix;
import org.jblas.MatrixFunctions;

public class NeuralNetworkLayerActivation {

	private DoubleMatrix inputActivations;
	private DoubleMatrix outputActivations;
	private NeuralNetworkLayer layer;
	private DoubleMatrix Z;
	private DoubleMatrix thetas;

	// private DoubleMatrix delta;

	public double getRegularisationCost(int m, double lambda) {
		DoubleMatrix currentTheta = thetas;

		// int m = X.getRows();

		int[] rows = new int[currentTheta.getRows()];
		int[] cols = new int[currentTheta.getColumns() - 1];
		for (int j = 0; j < currentTheta.getRows(); j++) {
			rows[j] = j;
		}
		for (int j = 1; j < currentTheta.getColumns(); j++) {
			cols[j - 1] = j;
		}
		double ThetaReg = MatrixFunctions.pow(currentTheta.get(rows, cols), 2).sum();
		return ((lambda) * ThetaReg) / (2 * m); // Add the non regularization
												// and regularization cost
												// together

	}

	protected DoubleMatrix backPropagate(NeuralNetworkLayerActivation outerActivation, DoubleMatrix outerDeltas) {

		DoubleMatrix sigable = getZ();

		sigable = DoubleMatrix.concatHorizontally(DoubleMatrix.ones(sigable.getRows()), sigable);

		DoubleMatrix deltas = outerActivation.getThetas().transpose().mmul(outerDeltas)
				.mul(this.getLayer().getActivationFunction().activationGradient(sigable.transpose())).transpose();

		int[] rows = new int[deltas.getRows()];
		int[] cols = new int[deltas.getColumns() - 1];
		for (int j = 0; j < deltas.getRows(); j++) {
			rows[j] = j;
		}
		for (int j = 1; j < deltas.getColumns(); j++) {
			cols[j - 1] = j;
		}

		deltas = deltas.get(rows, cols);

		return deltas.transpose();
	}

	public DoubleMatrix getZ() {
		return Z;
	}

	protected NeuralNetworkLayerErrorGradient getErrorGradient(DoubleMatrix D, double lambda, int m) {
		// DoubleMatrix D = delta.get(i);
		DoubleMatrix inputActivations = getInputActivations();
		NeuralNetworkLayerErrorGradient grad = new NeuralNetworkLayerErrorGradient(getLayer(), thetas, D, m, lambda,
				inputActivations);
		return grad;
	}

	public NeuralNetworkLayerActivation(NeuralNetworkLayer layer, DoubleMatrix inputActivations, DoubleMatrix Z,
			DoubleMatrix outputActivations) {
		this.inputActivations = inputActivations;
		this.Z = Z;
		this.outputActivations = outputActivations;
		this.layer = layer;
		this.thetas = layer.getClonedThetas();
	}

	private DoubleMatrix getThetas() {
		return thetas;
	}

	public DoubleMatrix getInputActivations() {
		return inputActivations;
	}

	public DoubleMatrix getOutputActivations() {
		return outputActivations;
	}

	public NeuralNetworkLayer getLayer() {
		return layer;
	}

}
