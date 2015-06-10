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

import java.util.ArrayList;
import java.util.List;
import java.util.Vector;

import org.jblas.DoubleMatrix;

public class BackPropagation {

	private List<NeuralNetworkLayerErrorGradient> gradients;

	public BackPropagation(ForwardPropagation forwardPropagation, Vector<DoubleMatrix> deltas, double[] lambdas, int m) {

		this.gradients = getRetrainableLayerGradients(lambdas, forwardPropagation, deltas, m);
	}

	/*
	 * public BackPropagation(List<NeuralNetworkLayerErrorGradient> gradients) {
	 * this.gradients = gradients;
	 * 
	 * }
	 */

	public List<NeuralNetworkLayerErrorGradient> getGradientsForRetrainableLayers() {
		return gradients;
	}
	
	

	private List<NeuralNetworkLayerErrorGradient> getRetrainableLayerGradients(double[] lambdas,
			ForwardPropagation forwardPropagation, Vector<DoubleMatrix> retrainableDeltas, int m) {

		List<NeuralNetworkLayerErrorGradient> layerGrads = new ArrayList<NeuralNetworkLayerErrorGradient>();
		// Calculate the gradients of each weight matrix
		int i = 0;
		for (NeuralNetworkLayerActivation layerActivation : forwardPropagation.getActivations()) {
			if (layerActivation.getLayer().isRetrainable()) {
				DoubleMatrix D = retrainableDeltas.get(i);
				double lambda = lambdas[i];
				layerGrads.add(layerActivation.getErrorGradient(D, lambda, m));

			}
			i++;

		}
		return layerGrads;
	}

}
