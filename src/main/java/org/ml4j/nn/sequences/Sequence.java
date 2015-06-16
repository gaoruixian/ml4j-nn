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
package org.ml4j.nn.sequences;

import org.jblas.DoubleMatrix;

/**
 * Sequential Data for use with Recurrent Neural Networks
 * 
 * @author Michael Lavelle
 *
 */
public interface Sequence {

	/**
	 * 
	 * @return the number of elements in this sequence (eg. timesteps)
	 */
	int getSequenceLength();

	/**
	 * 
	 * @return The sequence data as a DoubleMatrix - one row 
	 * per element
	 * 
	 */
	DoubleMatrix getSequenceData();
		
	/**
	 * 
	 * @return the number of attributes within each element
	 */
	int getElementLength();
	
}
