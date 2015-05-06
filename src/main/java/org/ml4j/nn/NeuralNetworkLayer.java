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
	private int layerNum;
	private ActivationFunction activationFunction;

	public ActivationFunction getActivationFunction() {
		return activationFunction;
	}
	
	
	

	protected DoubleMatrix backPropagate(DoubleMatrix layerInputs, DoubleMatrix outerThetas, DoubleMatrix outerDeltas) {
		DoubleMatrix sigable = layerInputs.mmul(thetas.transpose());

		sigable = DoubleMatrix.concatHorizontally(DoubleMatrix.ones(sigable.getRows()), sigable);

		DoubleMatrix previousD = outerThetas.transpose().mmul(outerDeltas)
				.mul(activationFunction.activationGradient(sigable.transpose())).transpose();

		int[] rows = new int[previousD.getRows()];
		int[] cols = new int[previousD.getColumns() - 1];
		for (int j = 0; j < previousD.getRows(); j++) {
			rows[j] = j;
		}
		for (int j = 1; j < previousD.getColumns(); j++) {
			cols[j - 1] = j;
		}

		previousD = previousD.get(rows, cols);

		return previousD.transpose();
	}

	public DoubleMatrix forwardPropagate(DoubleMatrix layerInputs) {

		return activationFunction.activate(layerInputs.mmul(thetas.transpose()));

	}

	public DoubleMatrix getThetas() {
		return thetas;
	}

	public int getInputNeuronCount() {
		return inputNeuronCount;
	}

	public int getOutputNeuronCount() {
		return outputNeuronCount;
	}

	public NeuralNetworkLayer(int inputNeuronCount, int outputNeuronCount, int layerNum, boolean initialiseThetas,
			ActivationFunction activationFunction) {
		this.inputNeuronCount = inputNeuronCount;
		this.outputNeuronCount = outputNeuronCount;
		this.layerNum = layerNum;
		this.activationFunction = activationFunction;
		if (initialiseThetas) {
			setThetas(generateInitialThetas(getOutputNeuronCount(), getInputNeuronCount() + 1));
		}
	}

	public int getLayerNum() {
		return layerNum;
	}

	public void setThetas(DoubleMatrix thetas) {

		this.thetas = thetas;
	}

	public DoubleMatrix generateInitialThetas(int r, int c) {
		DoubleMatrix initial = DoubleMatrix.randn(r, c);
		return initial;
	}

}
