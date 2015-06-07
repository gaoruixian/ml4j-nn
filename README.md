# ml4j-nn

```

// Create NeuralNetwork
	
// Layers configured with input units, hidden units, an activation function and whether a bias unit is required	
FeedForwardLayer firstLayer = new FeedForwardLayer(784, 10, new SigmoidActivationFunction(),true);
FeedForwardLayer secondLayer = new FeedForwardLayer(10, 10, new SoftmaxActivationFunction(),true);	
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
