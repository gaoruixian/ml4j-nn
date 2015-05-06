package org.ml4j.nn;

import java.util.ArrayList;
import java.util.List;
import java.util.Vector;

import org.jblas.DoubleMatrix;

public class BackPropagation {

	private List<NeuralNetworkLayerErrorGradient> gradients;

	public BackPropagation(ForwardPropagation forwardPropagation, Vector<DoubleMatrix> deltas, double[] lambdas, int m) {

		this.gradients = getLayerGradients(lambdas, forwardPropagation, deltas, m);
	}

	public BackPropagation(List<NeuralNetworkLayerErrorGradient> gradients) {
		this.gradients = gradients;

	}

	public List<NeuralNetworkLayerErrorGradient> getGradients() {
		return gradients;
	}

	private List<NeuralNetworkLayerErrorGradient> getLayerGradients(double[] lambdas,
			ForwardPropagation forwardPropagation, Vector<DoubleMatrix> delta, int m) {

		List<NeuralNetworkLayerErrorGradient> layerGrads = new ArrayList<NeuralNetworkLayerErrorGradient>();
		// Calculate the gradients of each weight matrix
		int i = 0;
		for (NeuralNetworkLayerActivation layerActivation : forwardPropagation.getActivations()) {
			DoubleMatrix D = delta.get(i);
			double lambda = lambdas[i];
			layerGrads.add(layerActivation.getErrorGradient(D, lambda, m));
			i++;

		}
		return layerGrads;
	}

}
