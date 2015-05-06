/*
 * Copyright 2014 the original author or authors.
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
package org.ml4j.nn.algorithms;

import java.io.Serializable;

import org.ml4j.nn.costfunctions.CostFunction;

public class NeuralNetworkAlgorithmTrainingContext implements Serializable {

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;

	private int maxIterations;

	private double regularizationLambda;

	private CostFunction costFunction;

	public NeuralNetworkAlgorithmTrainingContext(CostFunction costFunction, int maxIterations) {
		super();
		this.maxIterations = maxIterations;
		this.costFunction = costFunction;
	}

	public NeuralNetworkAlgorithmTrainingContext(int maxIterations) {
		super();
		this.maxIterations = maxIterations;
	}

	public CostFunction getCostFunction() {
		return costFunction;
	}

	public int getMaxIterations() {
		return maxIterations;
	}

	public double getRegularizationLambda() {
		return regularizationLambda;
	}

	public void setRegularizationLambda(double regularizationLambda) {
		this.regularizationLambda = regularizationLambda;
	}

}
