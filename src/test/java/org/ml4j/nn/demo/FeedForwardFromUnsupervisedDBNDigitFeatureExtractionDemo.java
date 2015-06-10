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

import org.ml4j.imaging.targets.ImageDisplay;
import org.ml4j.nn.FeedForwardLayer;
import org.ml4j.nn.FeedForwardNeuralNetwork;
import org.ml4j.nn.RestrictedBoltzmannLayer;
import org.ml4j.nn.UnsupervisedDeepBeliefNetwork;
import org.ml4j.nn.activationfunctions.SoftmaxActivationFunction;
import org.ml4j.nn.algorithms.NeuralNetworkAlgorithm;
import org.ml4j.nn.algorithms.NeuralNetworkAlgorithmTrainingContext;
import org.ml4j.nn.algorithms.NeuralNetworkHypothesisFunction;
import org.ml4j.nn.algorithms.RestrictedBoltzmannMachineAlgorithmTrainingContext;
import org.ml4j.nn.algorithms.UnsupervisedDeepBeliefNetworkAlgorithm;
import org.ml4j.nn.algorithms.UnsupervisedDeepBeliefNetworkHypothesisFunction;
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
public class FeedForwardFromUnsupervisedDBNDigitFeatureExtractionDemo {

	
	public static void main(String[] args) throws IOException, InterruptedException {

		DoubleArrayMatrixLoader loader = new DoubleArrayMatrixLoader(
				FeedForwardFromUnsupervisedDBNDigitFeatureExtractionDemo.class.getClassLoader());

		// Load Mnist data into double[][] matrices
		double[][] trainingDataMatrix = loader.loadDoubleMatrixFromCsv("mnist2500_X_custom.csv",
				new PixelFeaturesMatrixCsvDataExtractor(), 0, 500);
		double[][] testSetDataMatrix = loader.loadDoubleMatrixFromCsv("mnist2500_X_custom.csv",
				new PixelFeaturesMatrixCsvDataExtractor(), 500, 1000);

		double[][] trainingLabelsMatrix = loader.loadDoubleMatrixFromCsv("mnist2500_labels_custom.csv",
				new SingleDigitLabelsMatrixCsvDataExtractor(), 0, 500);

		double[][] testSetLabelsMatrix = loader.loadDoubleMatrixFromCsv("mnist2500_labels_custom.csv",
				new SingleDigitLabelsMatrixCsvDataExtractor(), 500, 1000);

		// Configure a Deep Belief Network		

		// Training Context
		int batchSize = 10;
		int iterations = 10;
		double learningRate = 0.05;
		int gibbsSamples = 100;

		RestrictedBoltzmannMachineAlgorithmTrainingContext context = new RestrictedBoltzmannMachineAlgorithmTrainingContext(batchSize,iterations,learningRate,gibbsSamples);
	
		RestrictedBoltzmannLayer firstLayer = new RestrictedBoltzmannLayer(784, 500);

		RestrictedBoltzmannLayer secondLayer = new RestrictedBoltzmannLayer(500, 500);

		RestrictedBoltzmannLayer thirdLayer = new RestrictedBoltzmannLayer(500, 2000);
			
		UnsupervisedDeepBeliefNetwork dbn = new UnsupervisedDeepBeliefNetwork(firstLayer,secondLayer,thirdLayer);
		
		UnsupervisedDeepBeliefNetworkAlgorithm alg = new UnsupervisedDeepBeliefNetworkAlgorithm(dbn,batchSize);

		// Obtain an generating hypothesis function from the Deep Belief Network, so we
		// can generate new data from a single training example
		System.out.println("Training");

		@SuppressWarnings("unused")
		UnsupervisedDeepBeliefNetworkHypothesisFunction hyp1 = alg.getHypothesisFunction(trainingDataMatrix, context);
	
		System.out.println("Creating FeedForward Neural Network initialised from DBN");
		
		FeedForwardLayer supervisedLayer = new FeedForwardLayer(2000,10,new SoftmaxActivationFunction(),true);
		FeedForwardNeuralNetwork feedForwardNeuralNetwork = dbn.createFeedForwardNeuralNetwork(supervisedLayer);
		
		NeuralNetworkAlgorithm neuralNetworkAlgorithm = new NeuralNetworkAlgorithm(feedForwardNeuralNetwork);

		System.out.println("Training FeedForward Neural Network to classify using back prop");

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


		ImageDisplay<Long> display = new ImageDisplay<Long>(280, 280);

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
