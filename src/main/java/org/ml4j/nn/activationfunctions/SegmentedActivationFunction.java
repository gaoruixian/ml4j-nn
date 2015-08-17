package org.ml4j.nn.activationfunctions;

import org.ml4j.DoubleMatrix;

public class SegmentedActivationFunction implements ActivationFunction {

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;
	
	private int[][] ranges;
	private ActivationFunction[] activationFunctions;
	
	
	
	public ActivationFunction[] getActivationFunctions() {
		return activationFunctions;
	}

	public SegmentedActivationFunction(ActivationFunction[] activationFunctions,int[][] ranges)
	{
		if (ranges.length != activationFunctions.length)
		{
			throw new IllegalArgumentException("Number of ranges must equal number of activation functions");
		}
		this.ranges = ranges;
		this.activationFunctions = activationFunctions;
	}

	@Override
	public DoubleMatrix activate(DoubleMatrix input) {
		DoubleMatrix result = null;
		int rangeNumber = 0;
		for (int[] range : ranges)
		{
			int startIndex = range[0];
			int endIndex = range[1];
			int[] cols = new int[endIndex - startIndex];
			for (int i = 0; i < cols.length; i++)
			{
				cols[i] = startIndex + i;
			}
			
			DoubleMatrix r = activationFunctions[rangeNumber++].activate(input.getColumns(cols));
			if (result == null)
			{
				result = r;
			}
			else
			{
				result = DoubleMatrix.concatHorizontally(result, r);
			}
		}
		
		
		return result;
	}

}
