package org.ml4j.nn;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import org.jblas.DoubleMatrix;

public class RestrictedBoltzmannMachine implements Serializable {

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;
	private RestrictedBoltzmannLayer layer;

	private DoubleMatrix currentHiddenStates;
	private DoubleMatrix currentVisibleStates;

	
	
	
	protected DoubleMatrix getCurrentVisibleStates() {
		return currentVisibleStates;
	}

	protected DoubleMatrix getCurrentHiddenStates() {
		return currentHiddenStates;
	}

	public RestrictedBoltzmannLayer getLayer() {
		return layer;
	}

	public RestrictedBoltzmannMachine(RestrictedBoltzmannLayer layer) {
		this.layer = layer;
	}

	public double[] encodeToProbabilities(double[] visibleUnits) {
		return layer.getHiddenUnitProbabilities(visibleUnits);
	}

	public double[] decode(double[] hiddenUnits) {
		return layer.getVisibleUnitProbabilities(hiddenUnits);
	}

	public DoubleMatrix encodeToProbabilities(DoubleMatrix visibleUnits) {
		return layer.getHiddenUnitProbabilities(visibleUnits);
	}

	public DoubleMatrix decodeToProbabilities(DoubleMatrix hiddenUnits) {
		return layer.getVisibleUnitProbabilities(hiddenUnits);
	}

	public double[] generateVisibleBinaries() {
		return getBinarySample(generateVisibleProbabilities());
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

	public double[] generateVisibleProbabilities() {
		double[] randomVisibleUnits = new double[layer.getVisibleNeuronCount()];
		double[] probs = null;
		int cdn = 20;
		for (int i = 0; i < randomVisibleUnits.length; i++) {
			randomVisibleUnits[i] = Math.random();
		}
		DoubleMatrix visibleUnitsMatrix = new DoubleMatrix(new double[][] { randomVisibleUnits });
		this.currentVisibleStates = null;
		this.currentHiddenStates = null;
		for (int i = 0; i < cdn; i++) {
			DoubleMatrix recWithIntercept = pushData(visibleUnitsMatrix);
			pushReconstruction(recWithIntercept);
			probs = layer.getVisibleUnitProbabilities(RestrictedBoltzmannLayer.removeInterceptColumn(currentHiddenStates).toArray());
		}
		return probs;
	}
	
	public double getAverageEnergy()
	{
		return getAverageEnergy(currentVisibleStates,currentHiddenStates);
	}
	
	public double getAverageEnergy(DoubleMatrix vs,DoubleMatrix hs)
	{

		//vs = DoubleMatrix.concatHorizontally(DoubleMatrix.ones(vs.getRows()), vs);
		//hs = DoubleMatrix.concatHorizontally(DoubleMatrix.ones(vs.getRows()), hs);

		return -vs.mmul(layer.getClonedThetas()).mmul(hs.transpose()).sum()/vs.getRows();

	}
	
	private List<DoubleMatrix> getBatches(DoubleMatrix doubleMatrix,int batchSize)
	{
		List<DoubleMatrix> batches = new ArrayList<DoubleMatrix>();
		int rowNum = 0;
		int numCompleteBatches = doubleMatrix.getRows()/batchSize;
		while (rowNum < doubleMatrix.getRows())
		{
			int batchCount = batches.size() <  numCompleteBatches ? batchSize : (doubleMatrix.getRows() - rowNum);
			
			int[] rows = new int[batchSize]; 
			for (int i = 0; i < batchCount; i++)
			{
				rows[i] = rowNum++;
			}
			batches.add(doubleMatrix.getRows(rows));
		}
		Collections.shuffle(batches);
		return batches;
	}
	
	
	public void train(DoubleMatrix matrix, int maxIterations,int miniBatchSize,double learningRate) {

		layer.setThetas(RestrictedBoltzmannLayer.generateInitialThetas(new double[matrix.getRows()][layer.getVisibleNeuronCount()], layer.getHiddenNeuronCount(),learningRate));
		for (int l = 0; l < maxIterations; l++) {
			for (DoubleMatrix doubleMatrix : getBatches(matrix,miniBatchSize))
			{
					DoubleMatrix reconstructionWithIntercept = pushData(doubleMatrix);
					DoubleMatrix positiveStatistics = getAveragePairwiseRowProducts(currentVisibleStates, currentHiddenStates);
	
					pushReconstruction(reconstructionWithIntercept);
					DoubleMatrix negativeStatistics = getAveragePairwiseRowProducts(currentVisibleStates, currentHiddenStates);
	
					DoubleMatrix delta = (positiveStatistics.sub(negativeStatistics)).mul(learningRate);

					layer.updateWithDelta(delta);
			}
			DoubleMatrix reconstructions = getReconstruction(matrix);
			System.out.println(getAverageReconstructionError(matrix,reconstructions));

		}

	}
	
	public double getAverageReconstructionError(DoubleMatrix data,DoubleMatrix reconstruction)
	{
		DoubleMatrix m =  data.sub(reconstruction);
		return m.mul(m).sum()/data.getRows();
		
	}

	public double[] encodeToBinary(double[] visibleUnits) {
		return layer.getHiddenUnitSample(visibleUnits);
	}
	
	public double[][] encodeToBinary(double[][] visibleUnits) {
		return layer.getHiddenUnitSample(visibleUnits);
	}

	public double[] decodeToBinary(double[] hiddenUnits) {
		return layer.getVisibleUnitSample(hiddenUnits);
	}

	protected DoubleMatrix pushData(DoubleMatrix data) {
		this.currentVisibleStates = layer.addInterceptColumn(data);
		this.currentHiddenStates = layer.getHSampleGivenV(currentVisibleStates);
		return layer.getProbVGivenH(currentHiddenStates);
	}
	
	
	protected DoubleMatrix getReconstruction(DoubleMatrix data) {
		DoubleMatrix currentVisibleStates = layer.addInterceptColumn(data);
		DoubleMatrix currentHiddenStates = layer.getHSampleGivenV(currentVisibleStates);
		return RestrictedBoltzmannLayer.removeInterceptColumn(layer.getProbVGivenH(currentHiddenStates));
	}

	protected void pushReconstruction(DoubleMatrix reconstructionWithIntercept) {
		this.currentVisibleStates = reconstructionWithIntercept;
		this.currentHiddenStates = layer.getProbHGivenV(reconstructionWithIntercept);
	}
	
	public DoubleMatrix getAveragePairwiseRowProducts(DoubleMatrix matrix1, DoubleMatrix matrix2) {
		DoubleMatrix result = new DoubleMatrix(matrix1.getColumns(), matrix2.getColumns());
		
		for (int i = 0; i < matrix1.getRows(); i++)
		{
			DoubleMatrix vector1 = matrix1.getRow(i);
			DoubleMatrix vector2 = matrix2.getRow(i);

			result.addi(getPairwiseVectorProduct(vector1,vector2));
		}
		return result.div(matrix1.getRows());
		
	}

	public DoubleMatrix getPairwiseVectorProduct(DoubleMatrix vector1, DoubleMatrix vector2) {
		DoubleMatrix result = new DoubleMatrix(vector1.getColumns(), vector2.getColumns());
		for (int i = 0; i < vector1.getColumns(); i++) {
			for (int j = 0; j < vector2.getColumns(); j++) {
				result.put(i, j, vector1.get(0, i) * vector2.get(0, j));
			}
		}
		return result;
	}

}
