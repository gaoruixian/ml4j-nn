package org.ml4j.nn.activationfunctions;

import org.jblas.DoubleMatrix;

public class BinarySoftmaxActivationFunction implements ActivationFunction {

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;
	
	private SoftmaxActivationFunction softmaxActivationFunction;
	
	public BinarySoftmaxActivationFunction()
	{
		this.softmaxActivationFunction = new SoftmaxActivationFunction();
	}
	
	@Override
	public DoubleMatrix activate(DoubleMatrix input) {
		DoubleMatrix result =  softmaxActivationFunction.activate(input);
		
		DoubleMatrix result2 = new DoubleMatrix(result.getRows(),result.getColumns());
		for (int r = 0; r < result.getRows();r++)
		{
			DoubleMatrix row = result.getRow(r);
			int index = row.argmax();
			for (int c = 0; c < row.getColumns(); c++)
			{
				if (c == index)
				{
					result2.put(r, c,1);
				}
				else
				{
					result2.put(r, c,0);

				}
			}
		}
		return result;
	}

}
