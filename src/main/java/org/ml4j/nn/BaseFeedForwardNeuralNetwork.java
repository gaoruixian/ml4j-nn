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

public abstract class BaseFeedForwardNeuralNetwork<N extends BaseFeedForwardNeuralNetwork<N>> extends DirectedNeuralNetwork<FeedForwardLayer,N> implements Serializable {

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;
	private int[] topology;

	public BaseFeedForwardNeuralNetwork(FeedForwardLayer... layers) {
		super(layers);
		this.topology = getCalculatedTopology();
	}
	
	public BaseFeedForwardNeuralNetwork(List<FeedForwardLayer> layers) {
		super(layers);
		this.topology = getCalculatedTopology();
	}
	
	public BaseFeedForwardNeuralNetwork(BaseFeedForwardNeuralNetwork<?> nn)
	{
		super(nn.layers);
		this.topology = getCalculatedTopology();
	}

	public void setAllLayersRetrainable() {
		for (FeedForwardLayer layer : layers) {
			layer.setRetrainable(true);
		}
	}
	
	public FeedForwardNeuralNetwork cloneAndAddLayers(FeedForwardLayer... layers) {
		List<FeedForwardLayer> clonedLayers = new ArrayList<FeedForwardLayer>();
		clonedLayers.addAll(this.layers);
		clonedLayers.addAll(Arrays.asList(layers));
		return new FeedForwardNeuralNetwork(clonedLayers.toArray(new FeedForwardLayer[clonedLayers.size()]));
	}	
	
	public boolean isSymmetricTopology()
	{
		if (getNumberOfLayers() % 2 != 0)
		{
			return false;
		}
		int half = getNumberOfLayers() /2;
		for (int i = 0; i < half; i++)
		{
			if ((layers.get(i).getInputNeuronCount() != layers.get(half * 2 - i - 1).getOutputNeuronCount())
				|| (layers.get(i).getOutputNeuronCount() != layers.get(half *2- i - 1).getInputNeuronCount()))
				{
					return false;
				}
		}
		return true;
	}

	public N dup(boolean allLayersRetrainable) {
		FeedForwardLayer[] dupLayers = new FeedForwardLayer[layers.size()];
		for (int i = 0; i < dupLayers.length; i++) {
			dupLayers[i] = layers.get(i).dup(allLayersRetrainable || layers.get(i).isRetrainable());
		}
		return createFromLayers(dupLayers);
	}
	
	protected abstract N createFromLayers(FeedForwardLayer[] layers);

	private int[] getCalculatedTopology() {
		int[] topology = new int[layers.size() + 1];
		topology[0] = layers.get(0).getInputNeuronCount();
		int ind = 1;
		Integer previousOutputNeuronCount = null;
		int layerNumber = 1;
		for (FeedForwardLayer layer : layers) {
			if (previousOutputNeuronCount != null) {
				if (previousOutputNeuronCount.intValue() != layer.getInputNeuronCount()) {
					throw new IllegalArgumentException("Input neuron count of layer " + layerNumber + " is "
							+ layer.getInputNeuronCount() + " but previous layer has "
							+ previousOutputNeuronCount.intValue() + " output neurons");
				}
			}
			topology[ind++] = layer.getOutputNeuronCount();
			previousOutputNeuronCount = layer.getOutputNeuronCount();
			layerNumber++;
		}
		return topology;
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

				DoubleMatrix newDeltas = activation.backPropagate(previousActivation, deltas);

				// DoubleMatrix newDeltas =
				// activation.getLayer().backPropagate(activation,
				// previousActivation.getThetas(), deltas);
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

	protected void train(DoubleMatrix inputs, DoubleMatrix desiredOutputs, double[] lambdas, int max_iter) {

		train(inputs, desiredOutputs, lambdas, getDefaultCostFunction(), max_iter);
	}

	protected void train(DoubleMatrix inputs, DoubleMatrix desiredOutputs, double lambda, int max_iter) {

		train(inputs, desiredOutputs, createLayerRegularisations(lambda), getDefaultCostFunction(), max_iter);
	}

	protected void train(DoubleMatrix inputs, DoubleMatrix desiredOutputs, double lambda, CostFunction costFunction,
			int max_iter) {
		train(inputs, desiredOutputs, createLayerRegularisations(lambda), costFunction, max_iter);

	}

	protected void train(DoubleMatrix inputs, DoubleMatrix desiredOutputs, double[] lambdas, CostFunction costFunction,
			int max_iter) {

		if (!isContainingRetrainableLayers()) {
			throw new IllegalStateException(
					"NeuralNetwork must contain at least one (re)trainable layer before calling train method");
		}

		// This clones the NeuralNetwork, minimises the thetas, and returns
		// optimal thetas
		DoubleMatrix newThetas = getMinimisingThetasForRetrainableLayers(inputs, desiredOutputs,
				getClonedRetrainableThetas(), lambdas, costFunction, max_iter);
		updateThetasForRetrainableLayers(newThetas, false);
	}

	public List<FeedForwardLayer> getLayers() {
		return layers;
	}

	
	public int getNumberOfLayers()
	{
		return layers.size();
	}
	
	public FeedForwardLayer getOuterLayer()
	{
		return layers.get(getNumberOfLayers() -1);
	}
	
	public FeedForwardLayer getFirstLayer()
	{
		return layers.get(0);
	}

	public Tuple<Double, DoubleMatrix> calculateCostAndGradientsForRetrainableLayers(DoubleMatrix X, DoubleMatrix Y,
			double[] lambda, CostFunction costFunction) {

		// ----------------|START FORWARD PROP AND FIND COST |-------------

		ForwardPropagation forwardPropagation = forwardPropagate(X);

		BackPropagation backPropagation = backPropagate(forwardPropagation, Y, lambda);

		// Get the cost from forward prop, taking account of thetas of existing
		// network

		// Get default cost function from outer-most activation function

		double J = forwardPropagation.getCostWithRetrainableLayerRegularisation(Y, lambda, costFunction);

		// Get the gradients from back prop
		List<NeuralNetworkLayerErrorGradient> layerGradients = backPropagation.getGradientsForRetrainableLayers();

		// Convert to deired format
		Vector<DoubleMatrix> gradList = new Vector<DoubleMatrix>();

		for (NeuralNetworkLayerErrorGradient grad : layerGradients) {
			gradList.add(grad.getErrorGradient());

		}
		DoubleMatrix gradients = new DoubleMatrix().copy(NeuralNetworkUtils.reshapeToVector(gradList));

		return new Tuple<Double, DoubleMatrix>(new Double(J), gradients);
	}

	private DoubleMatrix getMinimisingThetasForRetrainableLayers(DoubleMatrix inputs, DoubleMatrix desiredOutputs,
			Vector<DoubleMatrix> initialRetrainableThetas, double[] retrainableLambdas, CostFunction costFunction,
			int max_iter) {

		// Duplicate neural network
		N duplicateNeuralNetwork = dup(false);
		MinimisableCostAndGradientFunction minimisableCostFunction = new NeuralNetworkUpdatingCostFunction(inputs,
				desiredOutputs, topology, retrainableLambdas, duplicateNeuralNetwork, costFunction);

		DoubleMatrix pInput = NeuralNetworkUtils.reshapeToVector(initialRetrainableThetas);
		return CostFunctionMinimiser.fmincg(minimisableCostFunction, pInput, max_iter, true);
	}

	public Vector<DoubleMatrix> getClonedThetas() {
		Vector<DoubleMatrix> allThetasVec = new Vector<DoubleMatrix>();
		for (FeedForwardLayer layer : layers) {
			allThetasVec.add(layer.getClonedThetas());
		}
		return allThetasVec;

	}

	public Vector<DoubleMatrix> getClonedRetrainableThetas() {
		Vector<DoubleMatrix> allThetasVec = new Vector<DoubleMatrix>();
		for (FeedForwardLayer layer : layers) {
			if (layer.isRetrainable()) {
				allThetasVec.add(layer.getClonedThetas());
			}
		}
		return allThetasVec;

	}

	public boolean isContainingRetrainableLayers() {
		for (FeedForwardLayer layer : layers) {
			if (layer.isRetrainable()) {
				return true;
			}
		}
		return false;
	}

	public void updateThetas(Vector<DoubleMatrix> thetas, boolean permitFurtherRetrains) {

		int i = 0;
		for (FeedForwardLayer layer : layers) {
			int layerIndex = i;
			layer.updateThetas(thetas.get(i++), layerIndex, permitFurtherRetrains);
			System.out.println("Updated layer:" + layerIndex +  ":" + this.toString());
		}
	}

	public void updateThetasForRetrainableLayers(Vector<DoubleMatrix> retrainableThetas, boolean permitFurtherRetrains) {

		int i = 0;
		int layerIndex = 0;
		for (FeedForwardLayer layer : layers) {
			if (layer.isRetrainable()) {
				layer.updateThetas(retrainableThetas.get(i), layerIndex, permitFurtherRetrains);
				i++;
			}
			layerIndex++;

		}
	}

	public void updateThetasForRetrainableLayers(DoubleMatrix retrainableThetas, boolean permitFutherRetrains) {

		Vector<DoubleMatrix> ts = NeuralNetworkUtils.reshapeToList(retrainableThetas, getRetrainableTopologies());

		updateThetasForRetrainableLayers(ts, permitFutherRetrains);
	}

	private int[][] getRetrainableTopologies() {
		int count = 0;
		for (FeedForwardLayer layer : layers) {
			if (layer.isRetrainable())
				count++;
		}
		int[][] topologies = new int[count][2];
		int ind = 0;
		for (FeedForwardLayer layer : layers) {
			if (layer.isRetrainable()) {
				topologies[ind] = new int[] { layer.getOutputNeuronCount(), layer.getInputNeuronCount() + (layer.hasBiasUnit() ? 1 : 0) };
				ind++;

			}

		}

		return topologies;
	}

	public void updateThetasForAllLayers(DoubleMatrix thetas, boolean permitFutherRetrains) {

		updateThetas(NeuralNetworkUtils.reshapeToList(thetas, topology), permitFutherRetrains);
	}

	

	public CostFunction getDefaultCostFunction() {
		FeedForwardLayer outerLayer = getOuterLayer();
		return outerLayer.getActivationFunction().getDefaultCostFunction();
	}

	public double[] createLayerRegularisations(double regularisationLamdba) {
		double[] layerRegularisations = new double[getNumberOfLayers()];
		for (int i = 0; i < layerRegularisations.length; i++) {
			layerRegularisations[i] = regularisationLamdba;
		}
		return layerRegularisations;
	}

}
