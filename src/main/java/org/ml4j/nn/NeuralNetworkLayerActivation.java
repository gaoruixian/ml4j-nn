/*
 * Copyright 2015 the original author or authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.ml4j.nn;

import org.jblas.DoubleMatrix;
import org.jblas.MatrixFunctions;

public class NeuralNetworkLayerActivation {

	private DoubleMatrix inputActivations;
	private DoubleMatrix outputActivations;
	private FeedForwardLayer layer;
	private DoubleMatrix Z;
	private DoubleMatrix thetas;


	public double getRegularisationCost(int m, double lambda) {
		DoubleMatrix currentTheta = thetas;
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

		if (outerActivation.layer.hasBiasUnit)
		{
		sigable = DoubleMatrix.concatHorizontally(DoubleMatrix.ones(sigable.getRows()), sigable);
		}
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
		if (outerActivation.layer.hasBiasUnit)
		{
		deltas = deltas.get(rows, cols);
		}
		return deltas.transpose();
	}

	public DoubleMatrix getZ() {
		return Z;
	}

	protected NeuralNetworkLayerErrorGradient getErrorGradient(DoubleMatrix D, double lambda, int m) {
		DoubleMatrix inputActivations = getInputActivations();
		NeuralNetworkLayerErrorGradient grad = new NeuralNetworkLayerErrorGradient(getLayer(), thetas, D, m, lambda,
				inputActivations);
		return grad;
	}

	public NeuralNetworkLayerActivation(FeedForwardLayer layer, DoubleMatrix inputActivations, DoubleMatrix Z,
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

	public FeedForwardLayer getLayer() {
		return layer;
	}

}
