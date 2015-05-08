package org.ml4j.nn;

import java.io.Serializable;

import org.jblas.DoubleMatrix;
import org.ml4j.nn.activationfunctions.ActivationFunction;

public class NeuralNetworkLayer implements Serializable {

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;

	private DoubleMatrix thetas;

	private int inputNeuronCount;
	private int outputNeuronCount;
	private boolean retrainable;

	private ActivationFunction activationFunction;

	public ActivationFunction getActivationFunction() {
		return activationFunction;
	}

	public boolean isRetrainable() {
		return retrainable;
	}

	public void setRetrainable(boolean retrainable) {
		this.retrainable = retrainable;
	}

	public NeuralNetworkLayer dup(boolean retrainable) {
		NeuralNetworkLayer dup = new NeuralNetworkLayer(inputNeuronCount, outputNeuronCount, this.getClonedThetas(),
				activationFunction, retrainable);
		return dup;
	}

	public NeuralNetworkLayerActivation forwardPropagate(DoubleMatrix layerInputs) {
		DoubleMatrix Z = layerInputs.mmul(thetas.transpose());

		DoubleMatrix acts = activationFunction.activate(Z);
		NeuralNetworkLayerActivation activation = new NeuralNetworkLayerActivation(this, layerInputs, Z, acts);

		return activation;
	}

	public DoubleMatrix getClonedThetas() {

		DoubleMatrix ret = thetas.dup();
		return ret;
	}

	public int getInputNeuronCount() {
		return inputNeuronCount;
	}

	public int getOutputNeuronCount() {
		return outputNeuronCount;
	}

	public NeuralNetworkLayer(int inputNeuronCount, int outputNeuronCount, ActivationFunction activationFunction) {
		if (activationFunction == null) throw new IllegalArgumentException("Activation function passed to layer cannot be null");
		this.inputNeuronCount = inputNeuronCount;
		this.outputNeuronCount = outputNeuronCount;
		this.activationFunction = activationFunction;
		this.thetas = generateInitialThetas(getOutputNeuronCount(), getInputNeuronCount() + 1);
		this.retrainable = true;
	}
	

	public NeuralNetworkLayer(int inputNeuronCount, int outputNeuronCount, DoubleMatrix thetas,
			ActivationFunction activationFunction, boolean retrainable) {
		if (activationFunction == null) throw new IllegalArgumentException("Activation function passed to layer cannot be null");
		if (thetas == null) throw new IllegalArgumentException("Thetas passed to layer cannot be null");
		if (thetas.getRows() != outputNeuronCount || thetas.getColumns() != (inputNeuronCount + 1)) throw new IllegalArgumentException("Thetas matrix must be of dimensions " + outputNeuronCount +  ":" + (inputNeuronCount + 1));
		this.inputNeuronCount = inputNeuronCount;
		this.outputNeuronCount = outputNeuronCount;
		this.activationFunction = activationFunction;
		this.thetas = thetas;
		this.retrainable = retrainable;

	}

	protected void updateThetas(DoubleMatrix thetas, int layerInNetwork, boolean permitFurtherRetrains) {

		if (!retrainable) {
			throw new IllegalStateException("Layer " + layerInNetwork
					+ " has already been trained and has not been set to retrainable");
		}
		this.thetas = thetas;
		if (!permitFurtherRetrains) {
			this.retrainable = false;
		}
	}

	public DoubleMatrix generateInitialThetas(int r, int c) {
		DoubleMatrix initial = DoubleMatrix.randn(r, c);
		return initial;
	}
	
	public double[] getNeuronActivationMaximisingInputFeatures(int hiddenUnitIndex) {
		int jCount = thetas.getColumns() - 1;
		double[] maximisingInputFeatures = new double[jCount];
		for (int j = 0; j < jCount; j++) {
			double wij = getWij(hiddenUnitIndex, j);
			double sum = 0;

			for (int j2 = 0; j2 < jCount; j2++) {
				sum = sum + Math.pow(getWij(hiddenUnitIndex, j2), 2);
			}
			sum = Math.sqrt(sum);
			maximisingInputFeatures[j] = wij / sum;
		}
		return maximisingInputFeatures;
	}
	
	private double getWij(int i, int j) {
		DoubleMatrix weights = thetas;
		int jInd = j + 1;
		return weights.get(i, jInd);
	}
	

}
