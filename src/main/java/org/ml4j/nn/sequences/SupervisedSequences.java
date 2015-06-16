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

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.List;

/**
 * Container for SupervisedSequences
 * 
 * @author Michael Lavelle
 *
 */
public class SupervisedSequences {
	
	private Collection<SupervisedSequence> sequences;
	
	public SupervisedSequences(Collection<SupervisedSequence> sequences)
	{
		this.sequences = sequences;
	}
	
	public SupervisedSequences(SupervisedSequence... sequences)
	{
		this.sequences = Arrays.asList(sequences);
	}
	
	public void addSequence(SupervisedSequence sequence)
	{
		this.sequences.add(sequence);
	}
	
	public int getInputElementLength()
	{
		return sequences.iterator().next().getInputElementLength();
	}
	
	
	public int getOutputElementLength()
	{
		return sequences.iterator().next().getOutputElementLength();
	}

	public Collection<SupervisedSequence> filterBySequenceLength(int i) {
		
		List<SupervisedSequence> filtered = new ArrayList<SupervisedSequence>();
		for (SupervisedSequence sequence : sequences)
		{
			if (sequence.getSequenceLength() == i)
			{
				filtered.add(sequence);
			}
		}
		
		return filtered;
	}

}
