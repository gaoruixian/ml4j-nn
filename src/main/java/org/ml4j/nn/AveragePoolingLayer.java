package org.ml4j.nn;

import org.ml4j.cuda.DoubleMatrix;
import org.ml4j.nn.activationfunctions.LinearActivationFunction;

/**
 * Initial Prototype for average pooling layer - needs tidy up
 * 
 * @author Michael Lavelle
 *
 */
public class AveragePoolingLayer extends FeedForwardLayer {

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;

	private int depth;

	protected AveragePoolingLayer(int inputNeuronCount, int outputNeuronCount,
			DoubleMatrix thetas, DoubleMatrix thetasMask, int depth) {
		super(inputNeuronCount, outputNeuronCount, thetas, thetasMask,
				new LinearActivationFunction(), false, false,1);
		this.depth = depth;
	}

	public int getDepth() {
		return depth;
	}

	public AveragePoolingLayer(int inputNeuronCount, int outputNeuronCount,
			int depth) {
		super(inputNeuronCount, outputNeuronCount, createThetas(
				inputNeuronCount,
				outputNeuronCount,
				depth,
				createThetasMask(depth, inputNeuronCount, outputNeuronCount,
						false), false), createThetasMask(depth,
				inputNeuronCount, outputNeuronCount, false),
				new LinearActivationFunction(), false, false,1);
		this.depth = depth;
	}

	private static DoubleMatrix createThetas(int inputNeuronCount,
			int outputNeuronCount, int depth, DoubleMatrix thetasMask,
			boolean hasBiasUnit) {

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
		
		double scalingFactor = 1d / (scale * scale);

		DoubleMatrix initialThetas = DoubleMatrix.ones(outputNeuronCount,
				inputNeuronCount).mul(scalingFactor);

		return initialThetas;
	}

	public FeedForwardLayer dup(boolean retrainable) {

		if (retrainable) {
			throw new IllegalArgumentException(
					"Max pooling layers are not retriable");
		}

		FeedForwardLayer dup = new AveragePoolingLayer(inputNeuronCount,
				outputNeuronCount, this.getClonedThetas(), thetasMask, depth);
		return dup;
	}

	@Override
	public void applyGradientWeightConstraints(DoubleMatrix gradients) {

	}

	public static DoubleMatrix createThetasMask(int depth,
			int inputNeuronCount, int outputNeuronCount, boolean hasBiasUnit) {

		DoubleMatrix thetasMask = new DoubleMatrix(outputNeuronCount,
				inputNeuronCount);
		if (hasBiasUnit) {
			thetasMask = DoubleMatrix.concatHorizontally(
					DoubleMatrix.ones(thetasMask.getRows()), thetasMask);
		}

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
							thetasMask.put(outputInd, inputInd, 1);

						}
					}
				}
			}
		}
		return thetasMask;
	}

}
