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

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import org.ml4j.DoubleMatrix;
import org.ml4j.nn.algorithms.AutoEncoderAlgorithm;
import org.ml4j.nn.algorithms.AutoEncoderHypothesisFunction;
import org.ml4j.nn.algorithms.NeuralNetworkAlgorithmTrainingContext;

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
	
	
	
	private static FeedForwardLayer[] getStackedLayers(AutoEncoder...autoEncoders)
	{
		List<FeedForwardLayer> encoders = new ArrayList<FeedForwardLayer>();
		List<FeedForwardLayer> decoders = new ArrayList<FeedForwardLayer>();

		for (AutoEncoder autoEncoder : autoEncoders)
		{
			if (!autoEncoder.isSymmetricTopology())
			{
				throw new IllegalArgumentException("Can only stack symmetric autoencoders at this time");
			}
			List<FeedForwardLayer> allLayers = autoEncoder.getLayers();
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
		FeedForwardLayer[] stackedLayers = new FeedForwardLayer[encoders.size() + decoders.size()];
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

	public StackedAutoEncoder(FeedForwardLayer... layers) {
		super(layers);
	}
	
	public void trainGreedilyLayerwise(DoubleMatrix inputs, double lambda, int max_iter) {
		DoubleMatrix currentInputs = inputs;
		for (AutoEncoder encoder : getAutoEncoderStack())
		{
			AutoEncoderAlgorithm alg = new AutoEncoderAlgorithm(encoder);
			NeuralNetworkAlgorithmTrainingContext context = new NeuralNetworkAlgorithmTrainingContext(max_iter);
			context.setRegularizationLambda(lambda);
			AutoEncoderHypothesisFunction hyp = alg.getHypothesisFunction(currentInputs.toArray2(), context);
			currentInputs= new DoubleMatrix(hyp.encodeToLayer(currentInputs.toArray2(),0));
		}	
	}

}
