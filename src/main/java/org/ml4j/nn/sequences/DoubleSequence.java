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

import org.ml4j.cuda.DoubleMatrix;

public class DoubleSequence implements Sequence {

	private DoubleMatrix data;
	
	public DoubleSequence(double[] sequence)
	{
		this.data = new DoubleMatrix(sequence);
	}
	
	/**
	 * 
	 * @return the number of elements in this sequence (eg. timesteps)
	 */
	@Override
	public int getSequenceLength() {
		return data.getRows();
	}

	/**
	 * 
	 * @return The sequence data as a DoubleMatrix - one row 
	 * per element, each row with a single attribute, ie. this
	 * is a column
	 * 
	 */
	@Override
	public DoubleMatrix getSequenceData() {
		return data;
	}

	/**
	 * 
	 * @return the number of attributes within each element
	 */
	@Override
	public int getElementLength() {
		return data.getColumns();
	}

}
