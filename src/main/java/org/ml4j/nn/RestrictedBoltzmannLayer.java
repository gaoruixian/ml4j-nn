package org.ml4j.nn;

import org.jblas.DoubleMatrix;
import org.ml4j.nn.util.NeuralNetworkUtils;

public class RestrictedBoltzmannLayer extends BaseLayer<RestrictedBoltzmannLayer> {

	private int visibleNeuronCount;
	private int hiddenNeuronCount;

	private DoubleMatrix thetas;

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;

	public int getVisibleNeuronCount() {
		return visibleNeuronCount;
	}

	public int getHiddenNeuronCount() {
		return hiddenNeuronCount;
	}

	/**
	 * 
	 * See https://www.cs.toronto.edu/~hinton/absps/guideTR.pdf
	 * 
	 * @param data
	 * @param hiddenNeuronCount
	 * @param learningRate
	 * @return
	 */
	public static DoubleMatrix generateInitialThetas(double[][] data, int hiddenNeuronCount,double learningRate) {
		int visibleNeuronCount = data[0].length;
		int initialHiddenUnitBiasWeight = -4;
		DoubleMatrix thetas = DoubleMatrix.randn(visibleNeuronCount + 1, hiddenNeuronCount + 1).mul(learningRate);
		for (int i = 1; i < thetas.getColumns(); i++) {
			thetas.put(0, i, initialHiddenUnitBiasWeight);
		}
		for (int i = 1; i < thetas.getRows(); i++) {
			double[] proportionsOfOnUnits = getProportionsOfOnUnits(data);
			double proportionOfTimeUnitActivated = proportionsOfOnUnits[i - 1];
			// Needed to add the following to limit p here, otherwise the log blows up
			proportionOfTimeUnitActivated = Math.max(proportionOfTimeUnitActivated, 0.001);
			double initialVisibleUnitBiasWeight = Math.log(proportionOfTimeUnitActivated / (1 - proportionOfTimeUnitActivated));
			thetas.put(i, 0, initialVisibleUnitBiasWeight);
		}
		thetas.put(0, 0, 0);
		return thetas;
	}

	public RestrictedBoltzmannLayer(int visibleNeuronCount, int hiddenNeuronCount, DoubleMatrix thetas,
			boolean retrainable) {
		super(retrainable);
		this.visibleNeuronCount = visibleNeuronCount;
		this.hiddenNeuronCount = hiddenNeuronCount;
		this.thetas = thetas;
	}

	@Override
	public RestrictedBoltzmannLayer dup(boolean retrainable) {
		return new RestrictedBoltzmannLayer(visibleNeuronCount, hiddenNeuronCount, getClonedThetas(), retrainable);
	}

	public DoubleMatrix getClonedThetas() {

		DoubleMatrix ret = thetas.dup();
		return ret;
	}

	public double[] getNeuronActivationProbabilitiesForHiddenUnit(int j) {

		double[] hiddenUnitActivations = new double[this.hiddenNeuronCount];
		hiddenUnitActivations[j] = 1;

		return getVisibleUnitProbabilities(hiddenUnitActivations);

	}

	public DoubleMatrix getHiddenUnitProbabilities(double[][] visibleUnits) {
		return removeInterceptColumn(getProbHGivenV(addInterceptColumn(new DoubleMatrix(visibleUnits))));
	}

	public DoubleMatrix getVisibleUnitProbabilities(double[][] hiddenUnits) {
		return removeInterceptColumn(getProbVGivenH(addInterceptColumn(new DoubleMatrix(hiddenUnits))));
	}

	public double[] getHiddenUnitProbabilities(double[] visibleUnits) {
		return getHiddenUnitProbabilities(new DoubleMatrix(new double[][] { visibleUnits })).toArray();
	}

	public double[] getVisibleUnitProbabilities(double[] hiddenUnits) {
		return getVisibleUnitProbabilities(new DoubleMatrix(new double[][] { hiddenUnits })).toArray();
	}

	public DoubleMatrix getHiddenUnitProbabilities(DoubleMatrix visibleUnits) {
		return removeInterceptColumn(getProbHGivenV(addInterceptColumn(visibleUnits)));
	}

	public DoubleMatrix getVisibleUnitProbabilities(DoubleMatrix hiddenUnits) {
		return removeInterceptColumn(getProbVGivenH(addInterceptColumn(hiddenUnits)));
	}

	public double[] getHiddenUnitSample(double[] visibleUnits) {
		double[] probs = getHiddenUnitProbabilities(visibleUnits);
		return getBinarySample(probs);
	}

	private double[] getBinarySample(double[] probs) {
		DoubleMatrix rand = DoubleMatrix.rand(1, probs.length);
		DoubleMatrix res = new DoubleMatrix(new double[][] { probs }).sub(rand);
		DoubleMatrix result = new DoubleMatrix(1, res.getColumns());
		for (int i = 0; i < result.getColumns(); i++) {
			if (res.get(0, i) > 0) {
				result.put(0, i, 1);
			}
		}
		return result.toArray();
	}

	public double[] getVisibleUnitSample(double[] hiddenUnits) {
		double[] probs = getVisibleUnitProbabilities(hiddenUnits);

		return getBinarySample(probs);
	}

	protected DoubleMatrix getProbHGivenV(DoubleMatrix v) {

		DoubleMatrix result = v.mmul(thetas);
		result = NeuralNetworkUtils.sigmoid(result);
		for (int i = 0; i < v.getRows(); i++) {
			result.put(i, 0, 1);
		}

		return result;
	}

	protected DoubleMatrix getHSampleGivenV(DoubleMatrix v) {
		DoubleMatrix hProbs = getProbHGivenV(v);
		return getBinarySample(hProbs);
	}

	DoubleMatrix getVSampleGivenH(DoubleMatrix h) {
		DoubleMatrix vProbs = getProbVGivenH(h);
		return getBinarySample(vProbs);
	}

	private DoubleMatrix getBinarySample(DoubleMatrix hProbs) {
		DoubleMatrix rand = DoubleMatrix.rand(hProbs.getRows(), hProbs.getColumns());
		DoubleMatrix res = hProbs.sub(rand);
		DoubleMatrix result = new DoubleMatrix(res.getRows(), res.getColumns());
		for (int r = 0; r < result.getRows(); r++) {
			for (int i = 0; i < result.getColumns(); i++) {
				if (res.get(r, i) > 0) {
					result.put(r, i, 1);
				}
			}
		}
		return result;
	}

	protected DoubleMatrix getProbVGivenH(DoubleMatrix h) {
		DoubleMatrix result = h.mmul(thetas.transpose());
		result = NeuralNetworkUtils.sigmoid(result);
		for (int i = 0; i < h.getRows(); i++) {
			result.put(i, 0, 1);
		}
		return result;
	}

	protected DoubleMatrix removeInterceptColumn(DoubleMatrix in) {
		DoubleMatrix result = new DoubleMatrix(in.getRows(), in.getColumns() - 1);
		for (int i = 0; i < result.getRows(); i++) {
			for (int j = 0; j < result.getColumns() - 1; j++) {
				result.put(i, j, in.get(i, j + 1));
			}
		}
		return result;
	}

	protected DoubleMatrix addInterceptColumn(DoubleMatrix in) {
		return DoubleMatrix.concatHorizontally(DoubleMatrix.ones(in.getRows()), in);
	}

	public void updateWithDelta(DoubleMatrix delta) {

		this.thetas = this.thetas.add(delta);
	}

	private static double[] getProportionsOfOnUnits(double[][] data) {
		int[] counts = new int[data[0].length];
		for (double[] d : data) {
			for (int i = 0; i < counts.length; i++) {
				if (d[i] == 1) {
					counts[i]++;
				}
			}
		}
		double[] props = new double[counts.length];
		for (int i = 0; i < props.length; i++) {
			props[i] = counts[i] / data.length;
		}
		return props;
	}

}
