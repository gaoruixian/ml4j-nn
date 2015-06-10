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

/**
 * Represents an Undirected Layer of a NeuralNetwork - a layer through which information propagates
 * in both directions using symmetric connection weights.
 * 
 * @author Michael Lavelle
 *
 * @param <L> The type of DirectedLayer<L> this instance represents
 */
public abstract class UndirectedLayer<L extends UndirectedLayer<L>> extends BaseLayer<L>{

	
	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;

	/**
	 * UndirectedLayer constructor
	 * 
	 * @param retrainable Whether this layer can be (re)trained further.
	 */
	public UndirectedLayer(boolean retrainable) {
		super(retrainable);
	}

	

}
