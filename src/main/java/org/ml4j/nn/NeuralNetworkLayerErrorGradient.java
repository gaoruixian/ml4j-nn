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

public class NeuralNetworkLayerErrorGradient {

	private DirectedLayer<?> layer;
	private DoubleMatrix delta;
	private int m;
	private double lambda;
	private DoubleMatrix inputActivations;
	private DoubleMatrix thetas;
	private DoubleMatrix thetasMask;

	public NeuralNetworkLayerErrorGradient(DirectedLayer<?> layer, DoubleMatrix thetas,DoubleMatrix thetasMask, DoubleMatrix delta, int m,
			double lambda, DoubleMatrix inputActivations) {
		this.layer = layer;
		this.m = m;
		this.delta = delta;
		this.lambda = lambda;
		this.thetas = thetas;
		this.inputActivations = inputActivations;
		this.thetasMask = thetasMask;
	}
	
	public DirectedLayer<?> getLayer() {
		return layer;
	}

	public DoubleMatrix getDELTA() {
	//	System.out.println(delta.getRows() + ":" + delta.getColumns());
		//System.out.println(inputActivations.getRows() + ":" + inputActivations.getColumns());

		return delta.mmul(inputActivations).transpose();
	}

	public DoubleMatrix getErrorGradient() {
		DoubleMatrix currentTheta = thetas;
		
		DoubleMatrix grad =  getDELTA().div(m);
		
		if (lambda != 0)
		{
			DoubleMatrix modTheta = currentTheta;
			
			//modTheta.putRow(0, DoubleMatrix.zeros( 1,currentTheta.getColumns()));
			
			DoubleMatrix firstRow = grad.getRow(0);

			grad = grad.add(currentTheta.mul(lambda / m));
			if (layer.hasBiasUnit)
			{

				grad.putRow(0, firstRow);
			}
		}
		
		
	
		grad.muli(thetasMask);
		
		layer.applyGradientWeightConstraints(grad);
		
		return grad;
	}
}
