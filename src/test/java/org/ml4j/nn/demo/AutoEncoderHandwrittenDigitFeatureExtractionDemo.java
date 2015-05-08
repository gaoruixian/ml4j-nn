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

import org.jblas.DoubleMatrix;
import org.ml4j.imaging.targets.ImageDisplay;
import org.ml4j.nn.AutoEncoder;
import org.ml4j.nn.NeuralNetworkLayer;
import org.ml4j.nn.activationfunctions.SigmoidActivationFunction;
import org.ml4j.nn.algorithms.AutoEncoderAlgorithm;
import org.ml4j.nn.algorithms.AutoEncoderHypothesisFunction;
import org.ml4j.nn.algorithms.NeuralNetworkAlgorithmTrainingContext;
import org.ml4j.nn.util.MnistUtils;
import org.ml4j.nn.util.PixelFeaturesMatrixCsvDataExtractor;
import org.ml4j.util.DoubleArrayMatrixLoader;

/**
 * Demo of Handwritten Digit Feature Extraction using an AutoEncoder with the Mnist dataset
 * 
 * @author Michael Lavelle
 *
 */
public class AutoEncoderHandwrittenDigitFeatureExtractionDemo {

	public static void main(String[] args) throws IOException, InterruptedException {

		DoubleArrayMatrixLoader loader=  new DoubleArrayMatrixLoader(AutoEncoderHandwrittenDigitFeatureExtractionDemo.class.getClassLoader());

		
		// Load Mnist data into double[][] matrices
		double[][] trainingDataMatrix = loader.loadDoubleMatrixFromCsv("mnist2500_X_custom.csv", new PixelFeaturesMatrixCsvDataExtractor(),0,100);
		double[][] testSetDataMatrix = loader.loadDoubleMatrixFromCsv("mnist2500_X_custom.csv",  new PixelFeaturesMatrixCsvDataExtractor(),100,200);

		// Configure an AutoEncoder, with single hidden layer of configurable number of neurons
		
		// Hidden Neuron Topology
		int hiddenNeuronsCount = 100;

		// Training Context
		NeuralNetworkAlgorithmTrainingContext context = new NeuralNetworkAlgorithmTrainingContext(500);
		context.setRegularizationLambda(1);
		//context.setSparsityBeta(0.01);
		//context.setSparsityParameter(0.004);

		NeuralNetworkLayer firstLayer = new NeuralNetworkLayer(784, 100, new SigmoidActivationFunction());
		NeuralNetworkLayer secondLayer = new NeuralNetworkLayer(100, 784, new SigmoidActivationFunction());

		AutoEncoder autoEncoder = new AutoEncoder(firstLayer, secondLayer);
		
		AutoEncoderAlgorithm alg = new AutoEncoderAlgorithm(autoEncoder);

		// Obtain an encoding hypothesis function from the AutoEncoder, so we can extract compressed features
		AutoEncoderHypothesisFunction hyp1 = alg.getHypothesisFunction(trainingDataMatrix, context);

		// Obtain average activation for each feature on our training set
		
		
		//DoubleMatrix activations = AutoEncoder.forwardPropPredictFirstActivationOnly(hyp1.getAutoEncoder().getTheta(), new DoubleMatrix(trainingDataMatrix));
		DoubleMatrix activations = firstLayer.forwardPropagate(DoubleMatrix.concatHorizontally(DoubleMatrix.ones(trainingDataMatrix.length,1),new DoubleMatrix(trainingDataMatrix))).getOutputActivations();
		double totalFeatureActivation = 0d;
		for (int j = 0; j < activations.getColumns(); j++)
		{
			DoubleMatrix featureActivations = activations.getColumn(j);
			double averageFeatureActivation = featureActivations.sum() / activations.rows ;
			totalFeatureActivation = totalFeatureActivation + averageFeatureActivation;
			System.out.println("Average activation of hidden neuron for feature:" + j + " on training data is:" + averageFeatureActivation);
		}
		double averageFeatureActivation = totalFeatureActivation/activations.getColumns();
		
		System.out.println("Average activation of a hidden neuron accross all features is:" + averageFeatureActivation);
	
		// TODO Visualise the encoded features
        ImageDisplay<Long> display = new ImageDisplay<Long>(280,280);
        
        System.out.println("Drawing visualisations of patterns sought by the hidden neurons");
		for (int j = 0; j < hiddenNeuronsCount; j++)
		{
			double[] neuronActivationMaximisingFeatures = firstLayer.getNeuronActivationMaximisingInputFeatures(j);
			double[] intensities = new double[neuronActivationMaximisingFeatures.length];
			for (int i = 0 ; i < intensities.length; i++)
			{
				double val = neuronActivationMaximisingFeatures[i];
				double boundary = 0.02;
				intensities[i] =  val < -boundary ? 0 : val > boundary ? 1 : 0.5;
			}
			MnistUtils.draw(intensities,display);
			Thread.sleep(500);
		}
		
		// TODO Add sparsity constraints
		
		// TODO Check that sparsity contraints affect the average activation values appropriately
		
		// Visualise the reconstructions of the input data
        System.out.println("Visualising reconstructed data");
		for (int i = 0; i < 100; i++) {
			
			// For each element in our test set, obtain the compressed encoded features
			double[] encodedFeatures = hyp1.encodeFirstLayer(testSetDataMatrix[i]);
			
			// Now reconstruct the features again 
			double[] reconstructedFeatures = hyp1.decodeFirstLayer(encodedFeatures);
			
			// Display the reconstructed input image
			MnistUtils.draw(reconstructedFeatures,display);
			Thread.sleep(500);

		}

	}

	

}
