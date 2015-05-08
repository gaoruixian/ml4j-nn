package org.ml4j.nn;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import org.jblas.DoubleMatrix;
import org.ml4j.algorithms.HypothesisFunction;
import org.ml4j.nn.activationfunctions.ActivationFunction;
import org.ml4j.nn.algorithms.AutoEncoderAlgorithm;
import org.ml4j.nn.algorithms.AutoEncoderHypothesisFunction;
import org.ml4j.nn.algorithms.NeuralNetworkAlgorithmTrainingContext;
import org.ml4j.nn.algorithms.NeuralNetworkHypothesisFunction;
import org.ml4j.nn.costfunctions.CostFunction;

public class StackedAutoEncoder extends AutoEncoder {

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;
	
	private List<AutoEncoder> autoEncoderStack;
	
	private List<AutoEncoder> getAutoEncoderStack()
	{
		return autoEncoderStack;
	}
	
	public StackedAutoEncoder(AutoEncoder...autoEncoders) {
		super(getStackedLayers(autoEncoders));
		this.autoEncoderStack = Arrays.asList(autoEncoders);
	}
	
	private static NeuralNetworkLayer[] getStackedLayers(AutoEncoder...autoEncoders)
	{
		List<NeuralNetworkLayer> encoders = new ArrayList<NeuralNetworkLayer>();
		List<NeuralNetworkLayer> decoders = new ArrayList<NeuralNetworkLayer>();

		for (AutoEncoder autoEncoder : autoEncoders)
		{
			if (!autoEncoder.isSymmetricTopology())
			{
				throw new IllegalArgumentException("Can only stack symmetric autoencoders at this time");
			}
			List<NeuralNetworkLayer> allLayers = autoEncoder.getLayers();
			int encoderCount = allLayers.size() /2;
			int ind = 0;
			for (int i = 0; i < encoderCount;i++)
			{
				encoders.add(allLayers.get(ind++));
			}
			for (int i = 0; i < encoderCount;i++)
			{
				decoders.add(allLayers.get(ind++));
			}
		}
		Collections.reverse(decoders);
		NeuralNetworkLayer[] stackedLayers = new NeuralNetworkLayer[encoders.size() + decoders.size()];
		int ind = 0;
		for (int i = 0; i < encoders.size(); i++)
		{
			stackedLayers[ind++] = encoders.get(i);
		}
		for (int i = 0; i < encoders.size(); i++)
		{
			stackedLayers[ind++] = decoders.get(i);
		}
		return stackedLayers;
	}

	public StackedAutoEncoder(NeuralNetworkLayer... layers) {
		super(layers);
	}
	
	
	public void trainGreedilyLayerwise(DoubleMatrix inputs, double[] lambdas, int max_iter) {
		for (AutoEncoder encoder : getAutoEncoderStack())
		{
			encoder.train(inputs, lambdas,max_iter);
		}		}

	public void trainGreedilyLayerwise(DoubleMatrix inputs, double lambda, int max_iter) {
		DoubleMatrix currentInputs = inputs;
		for (AutoEncoder encoder : getAutoEncoderStack())
		{
			AutoEncoderAlgorithm alg = new AutoEncoderAlgorithm(encoder);
			NeuralNetworkAlgorithmTrainingContext context = new NeuralNetworkAlgorithmTrainingContext(max_iter);
			context.setRegularizationLambda(lambda);
			AutoEncoderHypothesisFunction hyp = alg.getHypothesisFunction(currentInputs.toArray2(), context);
			currentInputs= new DoubleMatrix(hyp.encode(currentInputs.toArray2()));
		}	
	}

	public void trainGreedilyLayerwise(DoubleMatrix inputs, double lambda, CostFunction costFunction,
			int max_iter) {
		for (AutoEncoder encoder : getAutoEncoderStack())
		{
			encoder.train(inputs, lambda, costFunction,max_iter);
		}
	}

	public void trainGreedilyLayerwise(DoubleMatrix inputs, double[] lambdas, CostFunction costFunction,
			int max_iter) {
		
		for (AutoEncoder encoder : getAutoEncoderStack())
		{
			encoder.train(inputs, lambdas, costFunction,max_iter);
		}
		
	}

	
	
}
