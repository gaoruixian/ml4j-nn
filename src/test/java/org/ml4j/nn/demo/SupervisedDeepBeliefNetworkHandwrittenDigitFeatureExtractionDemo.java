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

import org.ml4j.imaging.targets.ImageDisplay;
import org.ml4j.mapping.LabeledData;
import org.ml4j.nn.FeedForwardNeuralNetwork;
import org.ml4j.nn.RestrictedBoltzmannLayer;
import org.ml4j.nn.RestrictedBoltzmannMachine;
import org.ml4j.nn.RestrictedBoltzmannMachineStack;
import org.ml4j.nn.SupervisedDeepBeliefNetwork;
import org.ml4j.nn.activationfunctions.ActivationFunction;
import org.ml4j.nn.activationfunctions.BinarySoftmaxActivationFunction;
import org.ml4j.nn.activationfunctions.SegmentedActivationFunction;
import org.ml4j.nn.activationfunctions.SigmoidActivationFunction;
import org.ml4j.nn.activationfunctions.SoftmaxActivationFunction;
import org.ml4j.nn.algorithms.NeuralNetworkAlgorithm;
import org.ml4j.nn.algorithms.NeuralNetworkAlgorithmTrainingContext;
import org.ml4j.nn.algorithms.NeuralNetworkHypothesisFunction;
import org.ml4j.nn.algorithms.RestrictedBoltzmannMachineAlgorithmTrainingContext;
import org.ml4j.nn.algorithms.SupervisedDeepBeliefNetworkAlgorithm;
import org.ml4j.nn.algorithms.SupervisedDeepBeliefNetworkHypothesisFunction;
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
public class SupervisedDeepBeliefNetworkHandwrittenDigitFeatureExtractionDemo {

	
	public static void main(String[] args) throws IOException, InterruptedException {

		DoubleArrayMatrixLoader loader = new DoubleArrayMatrixLoader(
				SupervisedDeepBeliefNetworkHandwrittenDigitFeatureExtractionDemo.class.getClassLoader());

		// Load Mnist data into double[][] matrices
		double[][] trainingDataMatrix = loader.loadDoubleMatrixFromCsv("mnist2500_X_custom.csv",
				new PixelFeaturesMatrixCsvDataExtractor(), 0, 500);
		double[][] testSetDataMatrix = loader.loadDoubleMatrixFromCsv("mnist2500_X_custom.csv",
				new PixelFeaturesMatrixCsvDataExtractor(), 500, 1000);

		double[][] trainingLabelsMatrix = loader.loadDoubleMatrixFromCsv("mnist2500_labels_custom.csv",
				new SingleDigitLabelsMatrixCsvDataExtractor(), 0, 500);

		double[][] testSetLabelsMatrix = loader.loadDoubleMatrixFromCsv("mnist2500_labels_custom.csv",
				new SingleDigitLabelsMatrixCsvDataExtractor(), 500, 1000);

		double[][] onlyTwoLabels = getOnlyDigit2s(testSetLabelsMatrix,testSetLabelsMatrix);
		double[][] onlyTwosExamples = getOnlyDigit2s(testSetDataMatrix,testSetLabelsMatrix);

		double[] twoLabel = onlyTwoLabels[0];
		double[] twoExample = onlyTwosExamples[0];
		
		// Configure a Deep Belief Network		

		// Training Context
		int batchSize = 10;
		int iterations = 500;
		double learningRate = 0.05;
		int gibbsSamples = 100;

		RestrictedBoltzmannMachineAlgorithmTrainingContext context = new RestrictedBoltzmannMachineAlgorithmTrainingContext(batchSize,iterations,learningRate,gibbsSamples);
	
		RestrictedBoltzmannLayer firstLayer = new RestrictedBoltzmannLayer(784, 500);

		RestrictedBoltzmannLayer secondLayer = new RestrictedBoltzmannLayer(500, 500);

		ActivationFunction sigmoid = new SigmoidActivationFunction();
		ActivationFunction binarySoftmax = new BinarySoftmaxActivationFunction();
		int[][] ranges = new int[][] {new int[] {0,1},new int[]{1,11},new int[] {11,511}};
		ActivationFunction[] acts = new ActivationFunction[] {sigmoid,binarySoftmax,sigmoid};
		SegmentedActivationFunction act = new SegmentedActivationFunction(acts,ranges);
		
		RestrictedBoltzmannLayer thirdLayer = new RestrictedBoltzmannLayer(510, 2000,act,sigmoid);

		//thirdLayer.setLabeled(true);
		
		ImageDisplay<Long> display = new ImageDisplay<Long>(280, 280);
		
		SupervisedDeepBeliefNetwork dbn = new SupervisedDeepBeliefNetwork(new RestrictedBoltzmannMachineStack(new RestrictedBoltzmannMachine(firstLayer),new RestrictedBoltzmannMachine(secondLayer)),new RestrictedBoltzmannMachine(thirdLayer));
		SupervisedDeepBeliefNetworkAlgorithm alg = new SupervisedDeepBeliefNetworkAlgorithm(dbn,batchSize);

		// Obtain an generating hypothesis function from the Deep Belief Network, so we
		// can generate new data from a single training example
		System.out.println("Training");

		SupervisedDeepBeliefNetworkHypothesisFunction hyp1 = alg.getHypothesisFunction(trainingDataMatrix,trainingLabelsMatrix, context);
	
		System.out.println("Generating new data");
		for (int i = 0; i < 20; i++)
		{
			double[] probs = hyp1.predict(new LabeledData<double[],double[]>(twoExample,twoLabel));
			System.out.println("Generated new probability map from given an fixed example of digit two");
			MnistUtils.draw(probs, display);
			Thread.sleep(100);
		}
		
		System.out.println("Creating FeedForward Neural Network initialised from DBN");
		
		FeedForwardNeuralNetwork feedForwardNeuralNetwork = dbn.createFeedForwardNeuralNetwork(new SoftmaxActivationFunction());
		
		NeuralNetworkAlgorithm neuralNetworkAlgorithm = new NeuralNetworkAlgorithm(feedForwardNeuralNetwork);

		System.out.println("Training FeedForward Neural Network using back prop");

		// Training Context
		NeuralNetworkAlgorithmTrainingContext neuralNetworkContext = new NeuralNetworkAlgorithmTrainingContext(100);
		neuralNetworkContext.setRegularizationLambda(0.05d);

				// Obtain a prediction hypothesis function from the Neural Network, so
				// we can predict output classes given training examples
		NeuralNetworkHypothesisFunction neuralNetworkHyp = neuralNetworkAlgorithm.getHypothesisFunction(trainingDataMatrix, trainingLabelsMatrix,
		neuralNetworkContext);

		// Training Set accuracy
		System.out.println("Accuracy on training set:" + neuralNetworkHyp.getAccuracy(trainingDataMatrix, trainingLabelsMatrix));

		// Test Set accuracy
		System.out.println("Accuracy on test set:" + neuralNetworkHyp.getAccuracy(testSetDataMatrix, testSetLabelsMatrix));


				for (int i = 0; i < 100; i++) {

					// For each element in our test set, obtain the predicted and actual
					// classification
					double[] predictions = neuralNetworkHyp.predict(testSetDataMatrix[i]);

					int predicted = getArgMaxIndex(predictions);
					int actual = getArgMaxIndex(testSetLabelsMatrix[i]);

					// Output prediction
					System.out.println("Predicted:" + predicted + ",Actual:" + actual);

					// Display the actual input image
					MnistUtils.draw(testSetDataMatrix[i], display);
					Thread.sleep(1000);

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
	
	private static int getArgMaxIndex(double[] predictionNeuronValues) {
		Double max = null;
		Integer maxInt = null;
		int ind = 0;
		for (double d : predictionNeuronValues) {
			if (max == null || d > max.doubleValue()) {
				max = d;
				maxInt = ind;
			}
			ind++;
		}
		return maxInt;
	}


}
