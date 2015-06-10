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
import java.util.List;

public abstract class DeepBeliefNetwork<N extends DeepBeliefNetwork<N>> extends BaseNeuralNetwork<RestrictedBoltzmannLayer,N> implements Serializable {

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;
	protected RestrictedBoltzmannMachineStack  unsupervisedRbmStack;

	protected DeepBeliefNetwork(RestrictedBoltzmannMachineStack unsupervisedRbmStack)
	{
		super(unsupervisedRbmStack.getLayers());
		this.unsupervisedRbmStack = unsupervisedRbmStack;
	}
	
	
	
	protected DeepBeliefNetwork(RestrictedBoltzmannMachineStack unsupervisedRbmStack,RestrictedBoltzmannMachine supervisedLayer)
	{
		super(getLayers(unsupervisedRbmStack,supervisedLayer));
		this.unsupervisedRbmStack = unsupervisedRbmStack;
	}
	
		
	private static List<RestrictedBoltzmannLayer> getLayers(RestrictedBoltzmannMachineStack unsupervisedStack,RestrictedBoltzmannMachine supervisedRbm)
	{
		List<RestrictedBoltzmannLayer> layers = new ArrayList<RestrictedBoltzmannLayer>();
		layers.addAll(unsupervisedStack.getLayers());
		layers.add(supervisedRbm.getLayer());
		return layers;
	}
}
