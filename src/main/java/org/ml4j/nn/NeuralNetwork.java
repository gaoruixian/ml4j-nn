package org.ml4j.nn;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Vector;

import org.jblas.DoubleMatrix;
import org.ml4j.nn.costfunctions.CostFunction;
import org.ml4j.nn.optimisation.CostFunctionMinimiser;
import org.ml4j.nn.optimisation.MinimisableCostAndGradientFunction;
import org.ml4j.nn.optimisation.NeuralNetworkUpdatingCostFunction;
import org.ml4j.nn.optimisation.Tuple;
import org.ml4j.nn.util.NeuralNetworkUtils;

public class NeuralNetwork implements Serializable {

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;
	private List<NeuralNetworkLayer> layers;
	private int[] topology;

	
	public NeuralNetwork(NeuralNetworkLayer... layers) {
		this.layers = new ArrayList<NeuralNetworkLayer>();
		this.layers.addAll(Arrays.asList(layers));
		this.topology = getCalculatedTopology();
	}
	
	private int[] getCalculatedTopology()
	{
		int[] topology = new int[layers.size() + 1];
		topology[0] = layers.get(0).getInputNeuronCount();
		int ind = 1;
		for (NeuralNetworkLayer layer : layers)
		{
			topology[ind++] = layer.getOutputNeuronCount();
		}
		return topology;
	}
	
	public ForwardPropagation forwardPropagate(double[][] inputs) 
	{
		return forwardPropagate(new DoubleMatrix(inputs));
	}
	
	public ForwardPropagation forwardPropagate(double[] inputs) 
	{
		return forwardPropagate(new DoubleMatrix(new double[][] {inputs}));
	}
		
	public ForwardPropagation forwardPropagate(DoubleMatrix inputs) {
		DoubleMatrix inputActivations = inputs;
		List<NeuralNetworkLayerActivation> layerActivations = new ArrayList<NeuralNetworkLayerActivation>();
		for (NeuralNetworkLayer layer : layers) {
			inputActivations = DoubleMatrix.concatHorizontally(DoubleMatrix.ones(inputActivations.getRows(), 1),
					inputActivations);
			DoubleMatrix acts = layer.forwardPropagate(inputActivations);

			NeuralNetworkLayerActivation activation = new NeuralNetworkLayerActivation(layer, inputActivations, acts);
			layerActivations.add(activation);
			inputActivations = activation.getOutputActivations();
		}
		return new ForwardPropagation(inputActivations, layerActivations);
	}

	public BackPropagation backPropagate(ForwardPropagation forwardPropatation, DoubleMatrix desiredOutputs,
			double[] lambdas) {
		// Back propagate the deltas

		// Setup empty delta vector, and outer most deltas
		Vector<DoubleMatrix> deltasV = new Vector<DoubleMatrix>();
		DoubleMatrix deltas = forwardPropatation.getOutputs().sub(desiredOutputs).transpose();

		// Iterate thrrough activations in reverse order
		List<NeuralNetworkLayerActivation> activations = forwardPropatation.getActivations();
		List<NeuralNetworkLayerActivation> activationsReversed = new ArrayList<NeuralNetworkLayerActivation>();
		activationsReversed.addAll(activations);
		Collections.reverse(activationsReversed);
		NeuralNetworkLayerActivation previousActivation = null;
		for (NeuralNetworkLayerActivation activation : activationsReversed) {
			// For outer most layer, we just use the outer-most deltas

			// For other layers, we perform back propagation to obtain deltas,
			// passing in layer's input activations and the previous layer's
			// deltas and thetas

			if (previousActivation != null) {
				DoubleMatrix inputActivations = activation.getInputActivations();
				DoubleMatrix newDeltas = activation.getLayer().backPropagate(inputActivations,
						previousActivation.getLayer().getThetas(), deltas);
				deltasV.add(newDeltas);
				deltas = newDeltas;
			} else {
				deltasV.add(deltas);
			}
			previousActivation = activation;

		}
		// Ensure collected deltas are in correct order
		Collections.reverse(deltasV);

		// Create new BackPropagtion wrapper containing the deltas,the forward
		// propagation, the lambdas

		// This wrapper uses the data provided to calculate gradients which are
		// used by optimisation algs
		return new BackPropagation(forwardPropatation, deltasV, lambdas, desiredOutputs.getRows());
	}
	
	public void train(DoubleMatrix inputs, DoubleMatrix desiredOutputs, double[] lambdas, 
			int max_iter) {

			train(inputs,desiredOutputs,lambdas,getDefaultCostFunction(),max_iter);
	}
	
	public void train(DoubleMatrix inputs, DoubleMatrix desiredOutputs, double lambda, 
			int max_iter) {

			train(inputs,desiredOutputs,createLayerRegularisations(lambda),getDefaultCostFunction(),max_iter);
	}
	
	public void train(DoubleMatrix inputs, DoubleMatrix desiredOutputs, double lambda, CostFunction costFunction,
			int max_iter) {
		train(inputs,desiredOutputs,createLayerRegularisations(lambda),costFunction,max_iter);

	}

	public void train(DoubleMatrix inputs, DoubleMatrix desiredOutputs, double[] lambdas, CostFunction costFunction,
			int max_iter) {

		DoubleMatrix newThetas = getMinimisingThetas(inputs, desiredOutputs, getThetas(), lambdas, costFunction,
				max_iter);

		updateThetas(newThetas);
	}

	public List<NeuralNetworkLayer> getLayers() {
		return layers;
	}

	public Tuple<Double, DoubleMatrix> calculateCostAndGradients(DoubleMatrix X, DoubleMatrix Y, double[] lambda,
			CostFunction costFunction) {

		// ----------------|START FORWARD PROP AND FIND COST |-------------

		ForwardPropagation forwardPropagation = forwardPropagate(X);

		BackPropagation backPropagation = backPropagate(forwardPropagation, Y, lambda);

		// Get the cost from forward prop, taking account of thetas of existing
		// network

		// Get default cost function from outer-most activation function

		double J = forwardPropagation.getCost(Y, lambda, costFunction);

		// Get the gradients from back prop
		List<NeuralNetworkLayerErrorGradient> layerGradients = backPropagation.getGradients();

		// Convert to deired format
		Vector<DoubleMatrix> gradList = new Vector<DoubleMatrix>();

		for (NeuralNetworkLayerErrorGradient grad : layerGradients) {
			gradList.add(grad.getErrorGradient());

		}
		DoubleMatrix gradients = new DoubleMatrix().copy(NeuralNetworkUtils.reshapeToVector(gradList));

		return new Tuple<Double, DoubleMatrix>(new Double(J), gradients);
	}

	private DoubleMatrix getMinimisingThetas(DoubleMatrix inputs, DoubleMatrix desiredOutputs,
			Vector<DoubleMatrix> initialThetas, double[] lambdas, CostFunction costFunction, int max_iter) {

		MinimisableCostAndGradientFunction minimisableCostFunction = new NeuralNetworkUpdatingCostFunction(inputs,
				desiredOutputs, topology, lambdas, this, costFunction);
		DoubleMatrix pInput = NeuralNetworkUtils.reshapeToVector(initialThetas);
		return CostFunctionMinimiser.fmincg(minimisableCostFunction, pInput, max_iter, true);
	}

	public Vector<DoubleMatrix> getThetas() {
		Vector<DoubleMatrix> allThetasVec = new Vector<DoubleMatrix>();
		for (NeuralNetworkLayer layer : layers) {
			allThetasVec.add(layer.getThetas());
		}
		return allThetasVec;

	}
	
	public void setThetas(Vector<DoubleMatrix> thetasVec)
	{
		int i = 0;
		for (NeuralNetworkLayer layer : layers) {
			layer.setThetas(thetasVec.get(i++));
		}
	}

	private void updateThetas(DoubleMatrix newThetas) {

		Vector<DoubleMatrix> newThetasList = NeuralNetworkUtils.reshapeToList(newThetas, topology);
		int ind = 0;
		for (NeuralNetworkLayer layer : layers) {
			layer.setThetas(newThetasList.get(ind++));
		}
	}
	
	/**
	 * Helper function to compute the accuracy of predictions give said
	 * predictions and correct output matrix
	 */
	

	public String getAccuracy(DoubleMatrix trainingDataMatrix, DoubleMatrix trainingLabelsMatrix) {

		DoubleMatrix predictions = forwardPropagate(trainingDataMatrix).getOutputs();
		return computeAccuracy(predictions, trainingLabelsMatrix) + "";

	}
	
	protected double computeAccuracy(DoubleMatrix predictions, DoubleMatrix Y) {
		return ((predictions.mul(Y)).sum()) * 100 / Y.getRows();
	}
	
	
	public CostFunction getDefaultCostFunction() {
			List<NeuralNetworkLayer> layers = getLayers();
			NeuralNetworkLayer outerLayer = layers.get(layers.size() - 1);
			return outerLayer.getActivationFunction().getDefaultCostFunction();
	}
	
	public double[] createLayerRegularisations(double regularisationLamdba) {
		double[] layerRegularisations = new double[getLayers().size()];
		for (int i = 0; i < layerRegularisations.length; i++) {
			layerRegularisations[i] = regularisationLamdba;
		}
		return layerRegularisations;
	}

}
