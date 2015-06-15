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

public class NeuralNetworkLayerErrorGradient {

	private DirectedLayer<?> layer;
	private DoubleMatrix delta;
	private int m;
	private double lambda;
	private DoubleMatrix inputActivations;
	private DoubleMatrix thetas;

	public NeuralNetworkLayerErrorGradient(DirectedLayer<?> layer, DoubleMatrix thetas, DoubleMatrix delta, int m,
			double lambda, DoubleMatrix inputActivations) {
		this.layer = layer;
		this.m = m;
		this.delta = delta;
		this.lambda = lambda;
		this.thetas = thetas;
		this.inputActivations = inputActivations;
	}
	
	public DirectedLayer<?> getLayer() {
		return layer;
	}



	public DoubleMatrix getDELTA() {
		return delta.mmul(inputActivations);
	}

	public DoubleMatrix getErrorGradient() {
		DoubleMatrix currentTheta = thetas;
		DoubleMatrix modTheta = new DoubleMatrix().copy(currentTheta);
		if (layer.hasBiasUnit)
		{
		modTheta.putColumn(0, DoubleMatrix.zeros(currentTheta.getRows(), 1));
		}
		return getDELTA().div(m).add(modTheta.mul(lambda / m));
	}
}
