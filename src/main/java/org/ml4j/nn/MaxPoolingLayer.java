package org.ml4j.nn;

import org.ml4j.DoubleMatrix;
import org.ml4j.nn.activationfunctions.LinearActivationFunction;

/**
 * Initial Prototype for max pooling layer - needs tidy up
 * 
 * @author Michael Lavelle
 *
 */
public class MaxPoolingLayer extends FeedForwardLayer {

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;

	private int depth;

	protected MaxPoolingLayer(int inputNeuronCount, int outputNeuronCount,
			DoubleMatrix thetas, DoubleMatrix thetasMask, int depth) {
		super(inputNeuronCount, outputNeuronCount, thetas, thetasMask,
				new LinearActivationFunction(), false, false,1);
		this.depth = depth;
	}

	public int getDepth() {
		return depth;
	}

	public MaxPoolingLayer(int inputNeuronCount, int outputNeuronCount,
			int depth) {
		super(inputNeuronCount, outputNeuronCount, createThetas(
				inputNeuronCount,
				outputNeuronCount,
				depth,
				createThetasMask(depth, inputNeuronCount, outputNeuronCount
					)), createThetasMask(depth,
				inputNeuronCount, outputNeuronCount),
				new LinearActivationFunction(), false, false,1);
		this.depth = depth;
	}
	
	

	@Override
	public double createDropoutScaling(boolean training) {
		return 1;
	}

	@Override
	public DoubleMatrix createDropoutMask(DoubleMatrix inputs, boolean training) {
		DoubleMatrix dropoutMask = DoubleMatrix.zeros(inputs.getRows(),inputs.getColumns());
		
		int[][] inputMasks = createInputMasks();
		for (int[] inputMask : inputMasks)
		{
			
			for (int r = 0; r < inputs.getRows(); r++)
			{
				Double maxVal = null;
				Integer maxInd = null;
				for (int i = 0; i < inputMask.length ; i++)
				{
					double val = inputs.get(r,inputMask[i]);
					if (maxVal == null || val > maxVal.doubleValue())
					{
						maxInd = inputMask[i];
						maxVal = val;
					}
				}
				dropoutMask.put(r, maxInd,1);
				
			}
		}
		return dropoutMask;
	}

	private static DoubleMatrix createThetas(int inputNeuronCount,
			int outputNeuronCount, int depth, DoubleMatrix thetasMask) {

		int outputDim = (int) Math.sqrt(outputNeuronCount / depth);
		int inputDim = (int) Math.sqrt(inputNeuronCount / depth);

		int scale = inputDim / outputDim;
		
		if (inputDim * inputDim * depth != inputNeuronCount)
		{
			throw new IllegalArgumentException("Incorrect dimensions configuration of average pooling layer");
		}
		if (outputDim * outputDim * depth != outputNeuronCount)
		{
			throw new IllegalArgumentException("Incorrect dimensions configuration of average pooling layer");
		}
		if (outputDim * scale * outputDim * scale * depth != inputNeuronCount)
		{
			throw new IllegalArgumentException("Incorrect dimensions configuration of average pooling layer");
		}
		
		DoubleMatrix initialThetas = DoubleMatrix.ones(
				inputNeuronCount,outputNeuronCount);

		return initialThetas;
	}

	public FeedForwardLayer dup(boolean retrainable) {

		if (retrainable) {
			throw new IllegalArgumentException(
					"Max pooling layers are not retrainable");
		}

		FeedForwardLayer dup = new MaxPoolingLayer(inputNeuronCount,
				outputNeuronCount, this.getClonedThetas(), thetasMask, depth);
		return dup;
	}

	@Override
	public void applyGradientWeightConstraints(DoubleMatrix gradients) {

	}
	
	public String toString()
	{
		int inputDim = (int) Math.sqrt(inputNeuronCount / depth);
		int outputDim = (int) Math.sqrt(outputNeuronCount / depth);

		return "MaxPooling Layer mapping " + depth + " input images of dimension ("
 + inputDim + " * " + inputDim + " to " + depth + " images of dimension (" + outputDim + " * " + outputDim + ")";
	}
		
	public static DoubleMatrix createThetasMask(int depth,
			int inputNeuronCount, int outputNeuronCount) {

		DoubleMatrix thetasMask = new DoubleMatrix(
				inputNeuronCount,outputNeuronCount);
		
		int outputDim = (int) Math.sqrt(outputNeuronCount / depth);
		int inputDim = (int) Math.sqrt(inputNeuronCount / depth);

		int gridInputSize = inputNeuronCount / depth;
		int gridOutputSize = outputNeuronCount / depth;

		int scale = inputDim / outputDim;

		for (int grid = 0; grid < depth; grid++) {
			for (int i = 0; i < outputDim; i++) {
				for (int j = 0; j < outputDim; j++) {

					int startInputRow = i * scale;
					int startInputCol = j * scale;
					int outputInd = grid * gridOutputSize + (i * outputDim + j);
					for (int r = 0; r < scale; r++) {
						for (int c = 0; c < scale; c++) {
							int row = startInputRow + r;
							int col = startInputCol + c;
							int inputInd = grid * gridInputSize + row
									* inputDim + col;
							thetasMask.put( inputInd,outputInd, 1);

						}
					}
				}
			}
		}
		return thetasMask;
	}
	
	public int[][] createInputMasks() {

		int outputDim = (int) Math.sqrt(outputNeuronCount / depth);
		int inputDim = (int) Math.sqrt(inputNeuronCount / depth);

		int gridInputSize = inputNeuronCount / depth;
		int gridOutputSize = outputNeuronCount / depth;

		int scale = inputDim / outputDim;

		int[][] inputMasks = new int[outputDim * outputDim * depth][scale * scale];

		
		
		for (int grid = 0; grid < depth; grid++) {
			for (int i = 0; i < outputDim; i++) {
				for (int j = 0; j < outputDim; j++) {

					int startInputRow = i * scale;
					int startInputCol = j * scale;
					int outputInd = grid * gridOutputSize + (i * outputDim + j);
					int[] inputMask = new int[scale * scale];
					int ind = 0;
					for (int r = 0; r < scale; r++) {
						for (int c = 0; c < scale; c++) {
							int row = startInputRow + r;
							int col = startInputCol + c;
							int inputInd = grid * gridInputSize + row
									* inputDim + col;
							//thetasMask.put(outputInd, inputInd, 1);
							inputMask[ind++] = inputInd;

						}
					}
					inputMasks[outputInd] = inputMask; 
				}
			}
		}
		return inputMasks;
	}

}
