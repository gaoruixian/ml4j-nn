package org.ml4j.nn.demo;

import org.ml4j.nn.FeedForwardLayer;
import org.ml4j.nn.RecurrentLayer;
import org.ml4j.nn.RecurrentNeuralNetwork;
import org.ml4j.nn.activationfunctions.LinearActivationFunction;
import org.ml4j.nn.activationfunctions.SigmoidActivationFunction;
import org.ml4j.nn.sequences.DoubleSequence;
import org.ml4j.nn.sequences.SupervisedSequence;
import org.ml4j.nn.sequences.SupervisedSequences;

public class RecurrentNeuralNetworkDemo {

	public static void main(String[] args)
	{
		
		// Create a recurrent neural network, with 1 input neuron, 1 output neuron and 3 hidden neurons
		RecurrentLayer firstLayer = new RecurrentLayer(1, 3, new SigmoidActivationFunction(),true);
		FeedForwardLayer secondLayer = new FeedForwardLayer(3, 1, new LinearActivationFunction(),true);
		// Set max sequence length to be 5, and set learning rate to be 0.1
		RecurrentNeuralNetwork neuralNetwork = new RecurrentNeuralNetwork(firstLayer, secondLayer,5,0.1);
		
		// Create an input sequence
		DoubleSequence inputSequence =  new DoubleSequence(new double[] {1,2,3,2,1,2,3,2,1,2,3,2,1,2,3,2,1});
		
		// Create supervised sequence from this sequence, where the target sequence element is the next element in the input sequence
		SupervisedSequence supervisedSequence = new SupervisedSequence(inputSequence);
		
		// Create all supervised subsequences from this sequence
		SupervisedSequences supervisedSequences = supervisedSequence.createSubsequences();

		// Train on supervised subsequences
		
		// No regularisation
		double[] lambda = new double[2];
		
		// Max iterations = 500
		int maxIter = 500;
		neuralNetwork.trainOnSequences(supervisedSequences, lambda, maxIter);
		System.out.println("\nCompleted training on increasing then decreasing sequences\n");

		
		
		// Ensure that the neural network is cleared of context
		System.out.println("Clearing network of context\n");
		neuralNetwork.clearContext();
		
		System.out.println("Inputing increasing sequence\n");
		
		// Push 1
		System.out.println("Input 1\n");
		neuralNetwork.forwardPropagate(new double[] {1});
		

		// Push 2, and get prediction
		double prediction1 = neuralNetwork.forwardPropagate(new double[] {2}).getOutputs().toArray()[0];
		System.out.println("Input 2\n");

		System.out.println("Next element prediction is " +prediction1 +  "\n");
		
		// Ensure that the neural network is cleared of context
		System.out.println("Clearing network of context\n");
		
		System.out.println("Inputing decreasing sequence\n");

		
		neuralNetwork.clearContext();

		// Push 3
		System.out.println("Input 3\n");

		neuralNetwork.forwardPropagate(new double[] {3});
		
		
		// Push 2 and get prediction
		System.out.println("Input 2\n");
		double prediction2 = neuralNetwork.forwardPropagate(new double[] {2}).getOutputs().toArray()[0];
		System.out.println("Next element prediction is " + prediction2 + "\n");
		
		
		System.out.println("------\n");
		
		System.out.println("Clearing network of context\n");
		neuralNetwork.clearContext();
		
		System.out.println("Input 1\n");
		double nextElement = neuralNetwork.forwardPropagate(new double[] {1}).getOutputs().get(0);
		
		
		System.out.println("Input 2\n");
		nextElement = neuralNetwork.forwardPropagate(new double[] {2}).getOutputs().get(0);
		
		nextElement = Math.round(nextElement);
		System.out.println("Generating sequence, rounding the predictions to nearest integer\n");
		
		for (int i = 0; i < 100; i++)
		{
			System.out.println("Next element prediction is:" + nextElement + "\n");
			nextElement = neuralNetwork.forwardPropagate(new double[] {nextElement}).getOutputs().get(0);
			nextElement = Math.round(nextElement);
		}
	
	}
	
}
