/*
` * Copyright 2015 the original author or authors.
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
package org.ml4j.nn.demo;

import java.io.IOException;

import org.ml4j.imaging.targets.ImageDisplay;
import org.ml4j.nn.FeedForwardLayer;
import org.ml4j.nn.FeedForwardNeuralNetwork;
import org.ml4j.nn.activationfunctions.SigmoidActivationFunction;
import org.ml4j.nn.activationfunctions.SoftmaxActivationFunction;
import org.ml4j.nn.algorithms.NeuralNetworkAlgorithm;
import org.ml4j.nn.algorithms.NeuralNetworkAlgorithmTrainingContext;
import org.ml4j.nn.algorithms.NeuralNetworkHypothesisFunction;
import org.ml4j.nn.util.MnistUtils;
import org.ml4j.nn.util.PixelFeaturesMatrixCsvDataExtractor;
import org.ml4j.nn.util.SingleDigitLabelsMatrixCsvDataExtractor;
import org.ml4j.util.DoubleArrayMatrixLoader;

/**
 * Demo of Handwritten Digit Classification using the Mnist dataset
 * 
 * @author Michael Lavelle
 *
 */
public class FeedForwardNeuralNetworkHandwrittenDigitClassificationDemo {

	public static void main(String[] args) throws IOException, InterruptedException {

		// Load Mnist data into double[][] matrices

		DoubleArrayMatrixLoader loader = new DoubleArrayMatrixLoader(
				FeedForwardNeuralNetworkHandwrittenDigitClassificationDemo.class.getClassLoader());

		double[][] trainingDataMatrix = loader.loadDoubleMatrixFromCsv("mnist2500_X_custom.csv",
				new PixelFeaturesMatrixCsvDataExtractor(), 0, 100);
		double[][] testSetDataMatrix = loader.loadDoubleMatrixFromCsv("mnist2500_X_custom.csv",
				new PixelFeaturesMatrixCsvDataExtractor(), 100, 200);
		double[][] trainingLabelsMatrix = loader.loadDoubleMatrixFromCsv("mnist2500_labels_custom.csv",
				new SingleDigitLabelsMatrixCsvDataExtractor(), 0, 100);
		double[][] testSetLabelsMatrix = loader.loadDoubleMatrixFromCsv("mnist2500_labels_custom.csv",
				new SingleDigitLabelsMatrixCsvDataExtractor(), 100, 200);

		// Configure a Neural Network, with configurable hidden neuron topology,
		// and classification output neurons corresponding to the 10 numbers to
		// be predicted.

		FeedForwardLayer firstLayer = new FeedForwardLayer(784, 10, new SigmoidActivationFunction());
		FeedForwardLayer secondLayer = new FeedForwardLayer(10, 10, new SoftmaxActivationFunction());

		FeedForwardNeuralNetwork neuralNetwork = new FeedForwardNeuralNetwork(firstLayer, secondLayer);

		NeuralNetworkAlgorithm alg = new NeuralNetworkAlgorithm(neuralNetwork);

		// Training Context
		NeuralNetworkAlgorithmTrainingContext context = new NeuralNetworkAlgorithmTrainingContext(500);
		context.setRegularizationLambda(0.05d);

		// Obtain a prediction hypothesis function from the Neural Network, so
		// we can predict output classes given training examples
		NeuralNetworkHypothesisFunction hyp1 = alg.getHypothesisFunction(trainingDataMatrix, trainingLabelsMatrix,
				context);

		// Training Set accuracy
		System.out.println("Accuracy on training set:" + hyp1.getAccuracy(trainingDataMatrix, trainingLabelsMatrix));

		// Test Set accuracy
		System.out.println("Accuracy on test set:" + hyp1.getAccuracy(testSetDataMatrix, testSetLabelsMatrix));

		ImageDisplay<Long> display = new ImageDisplay<Long>(280, 280);

		for (int i = 0; i < 100; i++) {

			// For each element in our test set, obtain the predicted and actual
			// classification
			double[] predictions = hyp1.predict(testSetDataMatrix[i]);

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
