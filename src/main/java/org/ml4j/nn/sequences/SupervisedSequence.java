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
import java.util.Collection;

import org.ml4j.cuda.DoubleMatrix;

/**
 * Maps an input sequence to an output sequence
 * 
 * @author Michael Lavelle
 *
 */
public class SupervisedSequence  {

	private DoubleMatrix inputSequence;
	private DoubleMatrix outputSequence;
	
	/**
	 * Creates a supervised sequence from an unsupervised sequence, setting the target sequence elements to be the next element
	 * of the unsupervised sequence
	 * 
	 * @param sequence
	 */
	public SupervisedSequence(Sequence sequence)
	{
		double[][] s = generateUnsupervisedInputSequence(sequence.getSequenceData().toArray2());
		this.inputSequence = new DoubleMatrix(s);
		this.outputSequence = new DoubleMatrix(generateUnsupervisedOutputSequence(sequence.getSequenceData().toArray2()));
	}
	
	/**
	 * Creates a supervised sequence from an unsupervised double[] sequence, setting the target sequence elements to be the specified column of the next element
	 * of the unsupervised sequence
	 * 
	 * @param sequence
	 */
	public SupervisedSequence(DoubleArraySequence sequence,int outputColumn)
	{
		this.inputSequence = new DoubleMatrix(generateUnsupervisedInputSequence(sequence.getSequenceData().toArray2()));
		this.outputSequence = new DoubleMatrix(generateUnsupervisedOutputSequence(sequence.getSequenceData().getColumn(outputColumn).toArray2()));
	}
	
	/**
	 * Creates a supervised sequence from an input sequence and an output sequence
	 * 
	 * @param inputSequence
	 * @param outputSequence
	 */
	public SupervisedSequence(Sequence inputSequence,Sequence outputSequence)
	{
		if (inputSequence.getSequenceLength() != outputSequence.getSequenceLength())
		{
			throw new IllegalArgumentException("Input sequence must be the same length as output sequence");
		}
		this.inputSequence = inputSequence.getSequenceData();
		System.out.println("Input sequ:" + inputSequence.getSequenceData());
		this.outputSequence = outputSequence.getSequenceData();
	}
	
	public void print()
	{
		for (int r = 0; r< inputSequence.getRows(); r++)
		{
			DoubleMatrix input = inputSequence.getRow(r);
			DoubleMatrix output = outputSequence.getRow(r);
			System.out.println(input + ":" + output);
		}
	}
	
	/**
	 * Creates a supervised sequence from an unsupervised  sequence, setting the target sequence elements to be the specified column of the next element
	 * of the unsupervised sequence
	 * 
	 * @param sequence
	 */
	public SupervisedSequence(Sequence sequence,int predictionColumn)
	{
		this.inputSequence = new DoubleMatrix(generateUnsupervisedInputSequence(sequence.getSequenceData().toArray2()));
		this.outputSequence = new DoubleMatrix(generateUnsupervisedOutputSequence(sequence.getSequenceData().toArray2())).getColumn(predictionColumn);
	}
	
	public SupervisedSequence(Sequence sequence,int[] predictionColumns)
	{
		this.inputSequence = new DoubleMatrix(generateUnsupervisedInputSequence(sequence.getSequenceData().toArray2()));
		this.outputSequence = new DoubleMatrix(generateUnsupervisedOutputSequence(sequence.getSequenceData().toArray2())).getColumns(predictionColumns);
	}
	
	private double[][] generateUnsupervisedInputSequence(double[][] inputSequence) {
		double[][] unsupervisedSequence = new double[inputSequence.length - 1][inputSequence[0].length];
		for (int i = 0; i < unsupervisedSequence.length - 1; i++)
		{

			unsupervisedSequence[i] = inputSequence[i];

		}
		return unsupervisedSequence;
	}


	private double[][] generateUnsupervisedOutputSequence(double[][] inputSequence) {

		double[][] supervisedSequence = new double[inputSequence.length - 1][inputSequence[0].length];
		for (int i = 0; i < supervisedSequence.length - 1; i++)
		{
			supervisedSequence[i] = inputSequence[i + 1];
		}
		return supervisedSequence;
	
	}
	
	public int getSequenceLength() {
		return inputSequence.getRows();
	}

	public DoubleMatrix getInputElement(int r) {
		return inputSequence.getRow(r);
	}

	public DoubleMatrix getOutputElement(int r) {
		return outputSequence.getRow(r);
	}

	public SupervisedSequences createSubsequences() {
		Collection<SupervisedSequence> subsequences = new ArrayList<SupervisedSequence>();
		for (int i = 1; i <= getSequenceLength();i++)
		{
			for (int start = 0; start < getSequenceLength() - i; start++)
			{
				subsequences.add(new SupervisedSequence(getSubsequence(inputSequence,i,start),getSubsequence(outputSequence,i,start)));
			}
		}
		return new SupervisedSequences(subsequences);
	}
	
	protected SupervisedSequence(DoubleMatrix inputSequence,DoubleMatrix outputSequence)
	{
		this.inputSequence = inputSequence;
		this.outputSequence = outputSequence;
	}
	
	protected DoubleMatrix getSubsequence(DoubleMatrix input,int length,int start)
	{
			int[] rows = new int[length];
			for (int i = 0 ; i< rows.length;i++)
			{
				rows[i] = i + start;
			}
			return input.getRows(rows);
	}

	public int getInputElementLength() {
		return inputSequence.getRow(0).getColumns();
	}

	
	public int getOutputElementLength() {
		return outputSequence.getRow(0).getColumns();
	}

}
