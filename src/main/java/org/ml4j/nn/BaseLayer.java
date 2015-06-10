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

/**
 * Represents a Layer of a NeuralNetwork
 * 
 * @author Michael Lavelle
 *
 * @param <L> The type of BaseLayer<?> this instance represents
 */
public abstract class BaseLayer<L extends BaseLayer<?>> implements Serializable {
	
	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;

	
	private boolean retrainable;


	public boolean isRetrainable() {
		return retrainable;
	}


	public void setRetrainable(boolean retrainable) {
		this.retrainable = retrainable;
	}
	
	/**
	 * BaseLayer Constructor
	 * 
	 * @param retrainable Specifies whether the layer is allowed to be trained
	 */
	public BaseLayer(boolean retrainable)
	{
		this.retrainable = retrainable;
	}

	/**
	 * Duplicates the layer, copying the values of parameters, not
	 * sharing them.
	 * 
	 * @param retrainable Specifies whether the layer is allowed to be (re)trained further
	 * @return The duplicated layer
	 */
	public abstract L dup(boolean retrainable) ;
}
