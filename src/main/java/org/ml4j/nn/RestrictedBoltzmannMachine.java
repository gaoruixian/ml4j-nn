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

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import org.ml4j.DoubleMatrix;
import org.ml4j.nn.util.NeuralNetworkUtils;

public class RestrictedBoltzmannMachine extends SymmetricallyConnectedNeuralNetwork<RestrictedBoltzmannLayer,RestrictedBoltzmannMachine> implements Serializable {

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;

	private DoubleMatrix currentHiddenStates;
	private DoubleMatrix currentVisibleStates;
	
	private boolean thetasInitialised = false;

	
	protected DoubleMatrix getCurrentVisibleStates() {
		return currentVisibleStates;
	}

	protected DoubleMatrix getCurrentHiddenStates() {
		return currentHiddenStates;
	}

	public RestrictedBoltzmannLayer getLayer() {
		return getLayers().get(0);
	}

	public RestrictedBoltzmannMachine(RestrictedBoltzmannLayer layer) {
		super(new RestrictedBoltzmannLayer[] {layer});
	}

	public double[] encodeToProbabilities(double[] visibleUnits) {
		return getLayer().getHiddenUnitProbabilities(visibleUnits);
	}

	public double[] decode(double[] hiddenUnits) {
		return getLayer().getVisibleUnitProbabilities(hiddenUnits);
	}

	public DoubleMatrix encodeToProbabilities(DoubleMatrix visibleUnits) {
		return getLayer().getHiddenUnitProbabilities(visibleUnits);
	}

	public DoubleMatrix decodeToProbabilities(DoubleMatrix hiddenUnits) {
		return getLayer().getVisibleUnitProbabilities(hiddenUnits);
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
		double[] randomVisibleUnits = new double[getLayer().getVisibleNeuronCount()];
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
			probs = getLayer().getVisibleUnitProbabilities(NeuralNetworkUtils.removeInterceptColumn(currentHiddenStates).toArray());
		}
		return probs;
	}
	
	public double getAverageEnergy()
	{
		return getAverageEnergy(currentVisibleStates,currentHiddenStates);
	}
	
	public double getAverageEnergy(DoubleMatrix vs,DoubleMatrix hs)
	{
		return -vs.mmul(getLayer().getClonedThetas()).mmul(hs.transpose()).sum()/vs.getRows();

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

		if (!thetasInitialised)
		{
			getLayer().setThetas(RestrictedBoltzmannLayer.generateInitialThetas(new double[matrix.getRows()][getLayer().getVisibleNeuronCount()], getLayer().getHiddenNeuronCount(),learningRate));
			this.thetasInitialised = true;
		}
		for (int l = 0; l < maxIterations; l++) {
			for (DoubleMatrix doubleMatrix : getBatches(matrix,miniBatchSize))
			{
					DoubleMatrix reconstructionWithIntercept = pushData(doubleMatrix);
					DoubleMatrix positiveStatistics = getAveragePairwiseRowProducts(currentVisibleStates, currentHiddenStates);

					pushReconstruction(reconstructionWithIntercept);
					DoubleMatrix negativeStatistics = getAveragePairwiseRowProducts(currentVisibleStates, currentHiddenStates);
	
					DoubleMatrix delta = (positiveStatistics.sub(negativeStatistics)).mul(learningRate);

					getLayer().updateWithDelta(delta);
			}
			DoubleMatrix reconstructions = getReconstruction(matrix);
		
			System.out.print("Iteration " + (l + 1) + " | Average Reconstruction Error: " + getAverageReconstructionError(matrix,reconstructions) + "\r");
		}

	}
	
	public double getAverageReconstructionError(DoubleMatrix data,DoubleMatrix reconstruction)
	{
		DoubleMatrix m =  data.sub(reconstruction);
		return m.mul(m).sum()/data.getRows();
		
	}

	public double[] encodeToBinary(double[] visibleUnits) {
		return getLayer().getHiddenUnitSample(visibleUnits);
	}
	
	public double[][] encodeToBinary(double[][] visibleUnits) {
		return getLayer().getHiddenUnitSample(visibleUnits);
	}

	public double[] decodeToBinary(double[] hiddenUnits) {
		return getLayer().getVisibleUnitSample(hiddenUnits);
	}

	protected DoubleMatrix pushData(DoubleMatrix data) {
		this.currentVisibleStates = getLayer().addInterceptColumn(data);
		this.currentHiddenStates = getLayer().getHSampleGivenV(currentVisibleStates);
		return getLayer().getProbVGivenH(currentHiddenStates);
	}
	
	
	protected DoubleMatrix getReconstruction(DoubleMatrix data) {
		DoubleMatrix currentVisibleStates = getLayer().addInterceptColumn(data);
		DoubleMatrix currentHiddenStates = getLayer().getHSampleGivenV(currentVisibleStates);
		return NeuralNetworkUtils.removeInterceptColumn(getLayer().getProbVGivenH(currentHiddenStates));
	}

	protected void pushReconstruction(DoubleMatrix reconstructionWithIntercept) {
		this.currentVisibleStates = reconstructionWithIntercept;
		this.currentHiddenStates = getLayer().getProbHGivenV(reconstructionWithIntercept);
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
