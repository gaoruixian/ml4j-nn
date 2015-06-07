# ml4j-nn

```

// Create NeuralNetwork
	
FeedForwardLayer firstLayer = new FeedForwardLayer(784, 10, new SigmoidActivationFunction());
FeedForwardLayer secondLayer = new FeedForwardLayer(10, 10, new SoftmaxActivationFunction());	
FeedForwardNeuralNetwork neuralNetwork = new FeedForwardNeuralNetwork(firstLayer, secondLayer);

// Train NeuralNetwork
	
double regularisationLambda = 0.1;
int maxIterations = 500;
neuralNetwork.train(trainingDataMatrix, trainingLabelsMatrix, regularisationLambda, maxIterations);

// Training Set accuracy
System.out.println("Accuracy on training set:" + neuralNetwork.getAccuracy(trainingDataMatrix,trainingLabelsMatrix));

// Test Set accuracy
System.out.println("Accuracy on test set:" + neuralNetwork.getAccuracy(testSetDataMatrix, testSetLabelsMatrix));



```
