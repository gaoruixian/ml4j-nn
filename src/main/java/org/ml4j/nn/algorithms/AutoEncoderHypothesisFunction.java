package org.ml4j.nn.algorithms;

import java.util.Vector;

import org.jblas.DoubleMatrix;
import org.ml4j.nn.NeuralNetwork;
import org.ml4j.nn.activationfunctions.ActivationFunction;

public class AutoEncoderHypothesisFunction extends NeuralNetworkHypothesisFunction {

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;

	public AutoEncoderHypothesisFunction(NeuralNetwork neuralNetwork) {
		super(neuralNetwork);
	}
	

	public double[][] encode(double[][] numericFeaturesMatrix) {
		double[][] encodedDataSet = new double[numericFeaturesMatrix.length][];
		int i = 0;
		for (double[] numericFeatures : numericFeaturesMatrix )
		{
			encodedDataSet[i++] = encode(numericFeatures);
		}
		return encodedDataSet;
	}

	public double[] decode(double[] encodedFeatures) {
		double[][] d = new double[][] { encodedFeatures };
		DoubleMatrix X = new DoubleMatrix(d);
		Vector<DoubleMatrix> Theta = neuralNetwork.getThetas();
		int m = X.getRows();
		Vector<DoubleMatrix> activations = new Vector<DoubleMatrix>(Theta.size() + 1);
		DoubleMatrix firstActivation = new DoubleMatrix(m, Theta.firstElement().getColumns());
		firstActivation = DoubleMatrix.concatHorizontally(DoubleMatrix.ones(m, 1), X);
		activations.add(firstActivation);

		DoubleMatrix hypothesis = new DoubleMatrix(m, Theta.lastElement().getColumns());
		ActivationFunction activation = neuralNetwork.getLayers().get(1).getActivationFunction();
		hypothesis = activation.activate(activations.lastElement().mmul(Theta.lastElement().transpose()));
		return hypothesis.toArray();
	}

	
	public double[] encode(double[] numericFeatures) {
		double[][] d = new double[][] { numericFeatures };
		DoubleMatrix X = new DoubleMatrix(d);
		Vector<DoubleMatrix> Theta = neuralNetwork.getThetas();
		int m = X.getRows();
		Vector<DoubleMatrix> activations = new Vector<DoubleMatrix>(Theta.size() + 1);
		DoubleMatrix firstActivation = new DoubleMatrix(m, Theta.firstElement().getColumns());
		firstActivation = DoubleMatrix.concatHorizontally(DoubleMatrix.ones(m, 1), X);
		activations.add(firstActivation);

		DoubleMatrix hypothesis = new DoubleMatrix(m, Theta.lastElement().getColumns());
		ActivationFunction activation = neuralNetwork.getLayers().get(0).getActivationFunction();
		hypothesis = activation.activate(activations.lastElement().mmul(Theta.firstElement().transpose()));
		return hypothesis.toArray();
	}

	public double[] getHiddenNeuronActivationMaximisingInputFeatures(int hiddenUnitIndex) {
		int jCount = neuralNetwork.getThetas().get(0).getColumns() - 1;
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
		DoubleMatrix weights = neuralNetwork.getThetas().get(0);
		int jInd = j + 1;
		return weights.get(i, jInd);
	}
	
	
	

}
