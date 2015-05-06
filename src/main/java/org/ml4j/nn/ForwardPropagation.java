package org.ml4j.nn;

import java.util.List;

import org.jblas.DoubleMatrix;
import org.ml4j.nn.costfunctions.CostFunction;

public class ForwardPropagation {

	private DoubleMatrix outputs;
	private List<NeuralNetworkLayerActivation> activations;

	public ForwardPropagation(DoubleMatrix outputs, List<NeuralNetworkLayerActivation> activations) {
		this.outputs = outputs;
		this.activations = activations;
	}

	public List<NeuralNetworkLayerActivation> getActivations() {
		return activations;
	}

	public DoubleMatrix getOutputs() {
		return outputs;
	}

	public double getCost(DoubleMatrix desiredOutputs, double[] lambda, CostFunction cf) {

		DoubleMatrix X = activations.get(0).getInputActivations();

		int m = X.getRows();

		DoubleMatrix H = getOutputs();
		double J = cf.getCost(desiredOutputs, H);

		// Calculate regularization part of cost.
		int i = 0;
		for (NeuralNetworkLayerActivation layerActivation : getActivations()) {
			J = J + layerActivation.getRegularisationCost(m, lambda[i]);
			i++;
		}

		return J;
	}

}
