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
import org.ml4j.nn.FeedForwardLayer;
import org.ml4j.nn.RestrictedBoltzmannLayer;
import org.ml4j.nn.RestrictedBoltzmannMachine;
import org.ml4j.nn.RestrictedBoltzmannMachineStack;
import org.ml4j.nn.StackedAutoEncoder;
import org.ml4j.nn.UnsupervisedDeepBeliefNetwork;
import org.ml4j.nn.algorithms.AutoEncoderAlgorithm;
import org.ml4j.nn.algorithms.AutoEncoderHypothesisFunction;
import org.ml4j.nn.algorithms.NeuralNetworkAlgorithmTrainingContext;
import org.ml4j.nn.algorithms.RestrictedBoltzmannMachineAlgorithmTrainingContext;
import org.ml4j.nn.algorithms.UnsupervisedDeepBeliefNetworkAlgorithm;
import org.ml4j.nn.algorithms.UnsupervisedDeepBeliefNetworkHypothesisFunction;
import org.ml4j.nn.util.MnistUtils;
import org.ml4j.nn.util.PixelFeaturesMatrixCsvDataExtractor;
import org.ml4j.util.DoubleArrayMatrixLoader;

/**
 * Demo of Handwritten Digit Feature Extraction using an AutoEncoder with the Mnist dataset
 * 
 * @author Michael Lavelle
 *
 */
public class StackedAutoEncoderHandwrittenDigitFeatureExtractionDemo {

	public static void main(String[] args) throws IOException, InterruptedException {

		DoubleArrayMatrixLoader loader=  new DoubleArrayMatrixLoader(StackedAutoEncoderHandwrittenDigitFeatureExtractionDemo.class.getClassLoader());

		
		// Load Mnist data into double[][] matrices
		double[][] trainingDataMatrix = loader.loadDoubleMatrixFromCsv("mnist2500_X_custom.csv", new PixelFeaturesMatrixCsvDataExtractor(),0,100);
		double[][] testSetDataMatrix = loader.loadDoubleMatrixFromCsv("mnist2500_X_custom.csv",  new PixelFeaturesMatrixCsvDataExtractor(),100,200);
		
	
		int batchSize =10;
		int iterations = 10;
		double learningRate = 0.1;
		int gibbsSamples = 100;
		
		RestrictedBoltzmannMachineAlgorithmTrainingContext context = new RestrictedBoltzmannMachineAlgorithmTrainingContext(batchSize,iterations,learningRate,gibbsSamples);
		
		RestrictedBoltzmannLayer firstLayer = new RestrictedBoltzmannLayer(784, 500);

		
		RestrictedBoltzmannLayer secondLayer = new RestrictedBoltzmannLayer(500, 500);

		RestrictedBoltzmannLayer thirdLayer = new RestrictedBoltzmannLayer(500, 2000);
		
		ImageDisplay<Long> display = new ImageDisplay<Long>(280, 280);
		
		
		UnsupervisedDeepBeliefNetwork dbn = new UnsupervisedDeepBeliefNetwork(new RestrictedBoltzmannMachineStack(new RestrictedBoltzmannMachine(firstLayer),new RestrictedBoltzmannMachine(secondLayer),new RestrictedBoltzmannMachine(thirdLayer)));
		UnsupervisedDeepBeliefNetworkAlgorithm alg = new UnsupervisedDeepBeliefNetworkAlgorithm(dbn,batchSize);

		// Obtain an generating hypothesis function from the Deep Belief Network, so we
		// can generate new data from a single training example
		System.out.println("Training");

		@SuppressWarnings("unused")
		UnsupervisedDeepBeliefNetworkHypothesisFunction hyp1 = alg.getHypothesisFunction(trainingDataMatrix, context);
	
		System.out.println("Creating stacked autoencoder initialised from DBN");
	
		
		StackedAutoEncoder stackedAutoEncoder = dbn.createStackedAutoEncoder();
		

		// Training Context
		NeuralNetworkAlgorithmTrainingContext aeContext = new NeuralNetworkAlgorithmTrainingContext(100);
		aeContext.setRegularizationLambda(1);
		//context.setSparsityBeta(0.01);
		//context.setSparsityParameter(0.004);

		AutoEncoderAlgorithm aeAlg = new AutoEncoderAlgorithm(stackedAutoEncoder);

		// Obtain an encoding hypothesis function from the AutoEncoder, so we can extract compressed features
		AutoEncoderHypothesisFunction hyp2 = aeAlg.getHypothesisFunction(trainingDataMatrix, aeContext);

		// Obtain average activation for each feature on our training set
		
		FeedForwardLayer firstEncoderLayer = stackedAutoEncoder.getFirstLayer();

		//DoubleMatrix activations = AutoEncoder.forwardPropPredictFirstActivationOnly(hyp1.getAutoEncoder().getTheta(), new DoubleMatrix(trainingDataMatrix));
		DoubleMatrix activations = firstEncoderLayer.activate(trainingDataMatrix);
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
		
		// TODO Add sparsity constraints
		
		// TODO Check that sparsity contraints affect the average activation values appropriately
		
		// Visualise the reconstructions of the input data
        System.out.println("Visualising reconstructed data");
		for (int i = 0; i < 100; i++) {
			
			// For each element in our test set, obtain the compressed encoded features
			double[] encodedFeatures = hyp2.encodeToLayer(testSetDataMatrix[i],2);
			
			// Now reconstruct the features again 
			double[] reconstructedFeatures = hyp2.decodeFromLayer(encodedFeatures,3);
			
			// Display the reconstructed input image
			MnistUtils.draw(reconstructedFeatures,display);
			Thread.sleep(500);

		}

	}

	

}
