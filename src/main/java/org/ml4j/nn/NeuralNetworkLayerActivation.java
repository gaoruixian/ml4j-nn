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

import org.ml4j.DoubleMatrix;
import org.ml4j.MatrixFunctions;

public class NeuralNetworkLayerActivation<L extends DirectedLayer<?>> {

	private DoubleMatrix inputActivations;
	private DoubleMatrix outputActivations;
	private L layer;
	private DoubleMatrix Z;
	private DoubleMatrix thetas;
	private DoubleMatrix thetasMask;
	private DoubleMatrix dropoutMask;

	

	public DoubleMatrix getDropoutMask() {
		return dropoutMask;
	}

	public double getRegularisationCost(int m, double lambda) {
		DoubleMatrix currentTheta = thetas;
		int[] rows = new int[currentTheta.getRows() - 1];
		int[] cols = new int[currentTheta.getColumns()];
		for (int j = 0; j < currentTheta.getColumns(); j++) {
			cols[j] = j;
		}
		for (int j = 1; j < currentTheta.getRows(); j++) {
			rows[j - 1] = j;
		}
		double ThetaReg = MatrixFunctions.pow(currentTheta.get(rows, cols), 2).sum();
		return ((lambda) * ThetaReg) / (2 * m); // Add the non regularization
												// and regularization cost
												// together

	}

	protected DoubleMatrix backPropagate(NeuralNetworkLayerActivation<?> outerActivation, DoubleMatrix outerDeltas) {

		DoubleMatrix sigable = getZ();

		if (outerActivation.layer.hasBiasUnit)
		{
			sigable = DoubleMatrix.concatHorizontally(DoubleMatrix.ones(sigable.getRows()), sigable);
		}
		DoubleMatrix deltas = outerActivation.getThetas().mmul(outerDeltas);
		
		if (outerActivation.getLayer().inputDropout != 1)
		{
			deltas = deltas.mul(outerActivation.getDropoutMask());
		}
		
		
		deltas = deltas.mul(this.getLayer().getActivationFunction().activationGradient(sigable.transpose()));
	
		
		if (outerActivation.layer.hasBiasUnit)
		{
			int[] cols = new int[deltas.getColumns()];
			int[] rows = new int[deltas.getRows() - 1];
			for (int j = 0; j < deltas.getColumns(); j++) {
				cols[j] = j;
			}
			for (int j = 1; j < deltas.getRows(); j++) {
				rows[j - 1] = j;
			}
			deltas = deltas.get(rows, cols);
		}
		return deltas;
	}

	public DoubleMatrix getZ() {
		return Z;
	}

	protected NeuralNetworkLayerErrorGradient getErrorGradient(DoubleMatrix D, double lambda, int m) {
		DoubleMatrix inputActivations = getInputActivations();
		NeuralNetworkLayerErrorGradient grad = new NeuralNetworkLayerErrorGradient(getLayer(), thetas,thetasMask, D, m, lambda,
				inputActivations);
		return grad;
	}

	public NeuralNetworkLayerActivation(L layer, DoubleMatrix inputActivations, DoubleMatrix Z,
			DoubleMatrix outputActivations,DoubleMatrix thetasMask,DoubleMatrix dropoutMask) {
		this.inputActivations = inputActivations;
		this.Z = Z;
		this.outputActivations = outputActivations;
		this.layer = layer;
		this.thetas = layer.getClonedThetas();
		this.thetasMask = thetasMask;
		this.dropoutMask = dropoutMask;
	}
	
	public NeuralNetworkLayerActivation(L layer, DoubleMatrix inputActivations, DoubleMatrix Z,
			DoubleMatrix outputActivations) {
		this.inputActivations = inputActivations;
		this.Z = Z;
		this.outputActivations = outputActivations;
		this.layer = layer;
		this.thetas = layer.getClonedThetas();
		this.thetasMask = DoubleMatrix.ones(thetas.getRows(),thetas.getColumns());
		this.dropoutMask = DoubleMatrix.ones(inputActivations.getRows(),inputActivations.getColumns());
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

	public L getLayer() {
		return layer;
	}

}
