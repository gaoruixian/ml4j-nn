package org.ml4j.nn.demo;

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

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import org.ml4j.DoubleMatrix;
import org.ml4j.imaging.targets.ImageDisplay;
import org.ml4j.nn.RestrictedBoltzmannLayer;
import org.ml4j.nn.RestrictedBoltzmannMachine;
import org.ml4j.nn.algorithms.RestrictedBoltzmannMachineAlgorithm;
import org.ml4j.nn.algorithms.RestrictedBoltzmannMachineAlgorithmTrainingContext;
import org.ml4j.nn.algorithms.RestrictedBoltzmannMachineHypothesisFunction;
import org.ml4j.nn.util.MnistUtils;
import org.ml4j.nn.util.PixelFeaturesMatrixCsvDataExtractor;
import org.ml4j.nn.util.SingleDigitLabelsMatrixCsvDataExtractor;
import org.ml4j.util.DoubleArrayMatrixLoader;

/**
 * Demo of Handwritten Digit Feature Extraction using an RBM with the
 * Mnist dataset
 * 
 * @author Michael Lavelle
 *
 */
public class RestrictedBoltzmannMachineHandwrittenDigitFeatureExtractionDemo {

	public static void main(String[] args) throws IOException, InterruptedException {

		DoubleArrayMatrixLoader loader = new DoubleArrayMatrixLoader(
				RestrictedBoltzmannMachineHandwrittenDigitFeatureExtractionDemo.class.getClassLoader());

		// Load Mnist data into double[][] matrices
		double[][] trainingDataMatrix = loader.loadDoubleMatrixFromCsv("mnist2500_X_custom.csv",
				new PixelFeaturesMatrixCsvDataExtractor(), 0, 2000);
		double[][] testSetDataMatrix = loader.loadDoubleMatrixFromCsv("mnist2500_X_custom.csv",
				new PixelFeaturesMatrixCsvDataExtractor(), 2000, 2500);

		double[][] trainingLabelsMatrix = loader.loadDoubleMatrixFromCsv("mnist2500_labels_custom.csv",
				new SingleDigitLabelsMatrixCsvDataExtractor(), 0, 2000);

		double[][] testSetLabelsMatrix = loader.loadDoubleMatrixFromCsv("mnist2500_labels_custom.csv",
				new SingleDigitLabelsMatrixCsvDataExtractor(), 2000, 2500);

		trainingDataMatrix = getOnlyDigit2s(trainingDataMatrix, trainingLabelsMatrix);
		testSetDataMatrix = getOnlyDigit2s(testSetDataMatrix, testSetLabelsMatrix);

		// Configure an RBM

		// Hidden Neuron Topology
		int hiddenNeuronsCount = 50;
		

		// Training Context
		int batchSize = 10;
		int iterations = 500;
		double learningRate = 0.05;

		RestrictedBoltzmannMachineAlgorithmTrainingContext context = new RestrictedBoltzmannMachineAlgorithmTrainingContext(batchSize,iterations,learningRate);
	
		RestrictedBoltzmannLayer firstLayer = new RestrictedBoltzmannLayer(784, hiddenNeuronsCount);

		ImageDisplay<Long> display = new ImageDisplay<Long>(280, 280);
		
		RestrictedBoltzmannMachine rbm = new RestrictedBoltzmannMachine(firstLayer);
		RestrictedBoltzmannMachineAlgorithm alg = new RestrictedBoltzmannMachineAlgorithm(rbm,batchSize);

		// Obtain an encoding hypothesis function from the RestrictedBoltzmannMachine, so we
		// can extract features
		System.out.println("Training");

		RestrictedBoltzmannMachineHypothesisFunction hyp1 = alg.getHypothesisFunction(trainingDataMatrix, context);

		// Obtain average activation for each feature on our training set

		// DoubleMatrix activations =
		// AutoEncoder.forwardPropPredictFirstActivationOnly(hyp1.getAutoEncoder().getTheta(),
		// new DoubleMatrix(trainingDataMatrix));
		DoubleMatrix activations = firstLayer.getHiddenUnitProbabilities(trainingDataMatrix);
		double totalFeatureActivation = 0d;
		for (int j = 0; j < activations.getColumns(); j++) {
			DoubleMatrix featureActivations = activations.getColumn(j);
			double averageFeatureActivation = featureActivations.sum() / activations.getRows();
			totalFeatureActivation = totalFeatureActivation + averageFeatureActivation;
			System.out.println("Average activation of hidden neuron for feature:" + j + " on training data is:"
					+ averageFeatureActivation);
		}
		double averageFeatureActivation = totalFeatureActivation / activations.getColumns();

		System.out.println("Average activation of a hidden neuron accross all features is:" + averageFeatureActivation);

		System.out.println("Drawing visualisations of patterns sought by the hidden neurons");
		for (int j = 0; j < hiddenNeuronsCount; j++) {
			double[] neuronActivationMaximisingFeatures = firstLayer.getNeuronActivationProbabilitiesForHiddenUnit(j);
			double[] intensities = new double[neuronActivationMaximisingFeatures.length];
			for (int i = 0; i < intensities.length; i++) {
				double val = neuronActivationMaximisingFeatures[i];
				double scaleFactor = 200;
				intensities[i] = val *scaleFactor;
			}
			MnistUtils.draw(intensities, display);
			Thread.sleep(500);
		}
		
		
		RestrictedBoltzmannMachine clonedRestrictedBolzmannMachine = new RestrictedBoltzmannMachine(firstLayer.dup(false));
		
		
		
		// Use a cloned rbm to generate new data

		System.out.println("Generating new data");
		for (int i = 0; i < 100; i++)
		{
			double[] probs = clonedRestrictedBolzmannMachine.generateVisibleProbabilities();
			System.out.println("Generated new probability map from random visible input");
			MnistUtils.draw(probs, display);
			Thread.sleep(20);
		}

		// TODO Add sparsity constraints

		// TODO Check that sparsity contraints affect the average activation
		// values appropriately

		// Visualise the reconstructions of the input data
		System.out.println("Visualising reconstructed probability image maps");
		for (int i = 0; i < testSetDataMatrix.length; i++) {

			// For each element in our test set, obtain the compressed encoded
			// features
			double[] encodedFeatures = hyp1.sampleHiddenFromVisible(testSetDataMatrix[i]);

			// Now reconstruct the features again
			double[] reconstructedFeatures = hyp1.getVisibleProbabilitiesFromHidden(encodedFeatures);

			// Display the reconstructed input image
			MnistUtils.draw(reconstructedFeatures, display);
			Thread.sleep(500);

		}

	}

	private static double[][] getOnlyDigit2s(double[][] data, double[][] labels) {

		List<double[]> twosData = new ArrayList<double[]>();
		int r = 0;
		for (double[] dataPoint : data) {
			int digit = 0;
			for (int l = 0; l < labels[0].length; l++) {
				if (labels[r][l] == 1)
					digit = l;
			}
			if (digit == 2) {
				twosData.add(dataPoint);
			}
			r++;
		}

		double[][] twosMatrix = new double[twosData.size()][];
		for (int i = 0; i < twosMatrix.length; i++) {
			twosMatrix[i] = twosData.get(i);
		}

		return twosMatrix;
	}
	
	
}
