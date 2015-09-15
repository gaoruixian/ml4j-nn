package org.ml4j.nn;

import org.ml4j.DoubleMatrix;
import org.ml4j.nn.activationfunctions.DifferentiableActivationFunction;
/**
 * Initial Prototype for convolutional layer - needs tidy up
 * 
 * @author Michael Lavelle
 *
 */
public class ConvolutionalLayer extends FeedForwardLayer {

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;

	private int filterCount;
	private int depth;
	private Integer stride;
	
	protected ConvolutionalLayer(int inputNeuronCount, int outputNeuronCount,
			DoubleMatrix thetas, DoubleMatrix thetasMask,
			DifferentiableActivationFunction activationFunction,
			boolean biasUnit, boolean retrainable, int filterCount, int depth,Integer stride,double inputDropout) {
		super(inputNeuronCount, outputNeuronCount, thetas, thetasMask,
				activationFunction, biasUnit, retrainable,inputDropout);
		this.filterCount = filterCount;
		this.depth = depth;
		this.stride = stride;
	}
	
	@Override
	public String toString() {
		int outputDim = (int) Math.sqrt(outputNeuronCount / filterCount);
		int inputDim = (int) Math.sqrt(inputNeuronCount / depth);

		int filterWidth = inputDim + (1 - outputDim) * (stride == null ? 1 : stride);
		int strideAmount = stride == null ? 1 : stride;

		return "Convolutional Layer accepting " + depth + " input images of dimension (" + inputDim + " * " + inputDim + ") and applying " + getFilterCount() + " convolutional (" + filterWidth + " * " + filterWidth + ") filters at stride " + strideAmount + ", producing " + getFilterCount() + " (" + outputDim + " * " + outputDim + ") output image(s)";
	}

	public int getDepth() {
		return depth;
	}

	public int getFilterCount() {
		return filterCount;
	}

	public ConvolutionalLayer(int inputNeuronCount, int outputNeuronCount,
			DifferentiableActivationFunction activationFunction,
			boolean biasUnit, int filterCount, int depth,Integer stride,double initialThetaScaling,double inputDropout) {
		super(inputNeuronCount, outputNeuronCount, createThetas(
				inputNeuronCount,
				outputNeuronCount,
				filterCount,
				depth,
				stride,
				createThetasMask(filterCount, depth,stride, inputNeuronCount,
						outputNeuronCount, biasUnit), biasUnit,initialThetaScaling),
				createThetasMask(filterCount, depth,stride, inputNeuronCount,
						outputNeuronCount, biasUnit), activationFunction,
				biasUnit, true,inputDropout);
		this.filterCount = filterCount;
		this.depth = depth;
		this.stride = stride;
	}
	
	
	public ConvolutionalLayer(int inputNeuronCount, int outputNeuronCount,
			DifferentiableActivationFunction activationFunction,
			boolean biasUnit, int filterCount, int depth,int stride,double initialThetaScaling) {
		this(inputNeuronCount,outputNeuronCount,activationFunction,biasUnit,filterCount,depth,stride,initialThetaScaling,1);
	}
	
	public ConvolutionalLayer(int inputNeuronCount, int outputNeuronCount,
			DifferentiableActivationFunction activationFunction,
			boolean biasUnit, int filterCount, int depth,int stride) {
		this(inputNeuronCount,outputNeuronCount,activationFunction,biasUnit,filterCount,depth,stride,0.05,1);
	}
	
	public ConvolutionalLayer(int inputNeuronCount, int outputNeuronCount,
			DifferentiableActivationFunction activationFunction,
			boolean biasUnit, int filterCount, int depth) {
		this(inputNeuronCount,outputNeuronCount,activationFunction,biasUnit,filterCount,depth,1,0.05,1);
	}

	private static DoubleMatrix createThetas(int inputNeuronCount,
			int outputNeuronCount, int filterCount, int depth,Integer stride,
			DoubleMatrix thetasMask, boolean hasBiasUnit,double initialThetaScaling) {
		DoubleMatrix initialThetas = DoubleMatrix.randn(
				inputNeuronCount + (hasBiasUnit ? 1 : 0),outputNeuronCount).mul(initialThetaScaling);

		int filterOutputSize = outputNeuronCount / filterCount;

		for (int grid = 0; grid < depth; grid++) {
			for (int f = 0; f < filterCount; f++) {
				int startColumnIndex = f * filterOutputSize;
				int inputWidth = (int) Math.sqrt(inputNeuronCount / depth);
				int outputWidth = (int) Math.sqrt(outputNeuronCount
						/ filterCount);
				int filterWidth = inputWidth - (outputWidth - 1) * (stride == null ? 1 : stride);
				int sharedValueCount = filterWidth * filterWidth;

				int[][] sharedValueIndexes = new int[filterOutputSize][sharedValueCount
						* depth + (hasBiasUnit ? 1 : 0)];
				double[] sharedValues = new double[sharedValueCount];
				for (int column = 0; column < filterOutputSize; column++) {
					int[] inds = thetasMask.getColumn(column + startColumnIndex).findIndices();
					sharedValueIndexes[column] = inds;
				}

				for (int i = 0; i < sharedValueCount; i++) {
					sharedValues[i] = initialThetas.get(
							sharedValueIndexes[0][i + filterWidth * grid
									+ (hasBiasUnit ? 1 : 0)],startColumnIndex);
				}

				for (int column = 0; column < filterOutputSize; column++) {
					for (int sharedValueIndex = 0; sharedValueIndex < sharedValueCount; sharedValueIndex++) {

						initialThetas.put(sharedValueIndexes[column][sharedValueIndex + sharedValueCount
								* grid + (hasBiasUnit ? 1 : 0)],column + startColumnIndex, 
								sharedValues[sharedValueIndex]);
					}
				}

			}
		}

		return initialThetas;
	}

	public FeedForwardLayer dup(boolean retrainable) {
		FeedForwardLayer dup = new ConvolutionalLayer(inputNeuronCount,
				outputNeuronCount, this.getClonedThetas(), thetasMask,
				activationFunction, hasBiasUnit(), retrainable, filterCount,
				depth,stride,inputDropout);
		return dup;
	}

	@Override
	public void applyGradientWeightConstraints(DoubleMatrix gradients) {

		int filterOutputSize = getOutputNeuronCount() / filterCount;

		for (int grid = 0; grid < depth; grid++) {
			for (int f = 0; f < filterCount; f++) {
				int startColumnIndex = f * filterOutputSize;
				int inputWidth = (int) Math.sqrt(getInputNeuronCount() / depth);
				int outputWidth = (int) Math.sqrt(getOutputNeuronCount()
						/ filterCount);
				int filterWidth = inputWidth + (1 - outputWidth) * (stride == null ? 1 : stride);
				int sharedValueCount = filterWidth * filterWidth;

				int[][] sharedValueIndexes = new int[filterOutputSize][sharedValueCount
						+ (hasBiasUnit() ? 1 : 0)];
				double[] averageValues = new double[sharedValueCount];
				for (int column = 0; column < filterOutputSize; column++) {
					int[] inds = thetasMask.getColumn(column + startColumnIndex).findIndices();
					sharedValueIndexes[column] = inds;
					for (int i = 0; i < averageValues.length; i++) {

						averageValues[i] = averageValues[i]
								+ gradients.get(inds[filterWidth * grid
										+ i + (hasBiasUnit() ? 1 : 0)],column  + startColumnIndex);
					}
				}

				for (int i = 0; i < averageValues.length; i++) {
					averageValues[i] = averageValues[i] / filterOutputSize;
				}

				for (int column = 0; column < filterOutputSize; column++) {
					for (int sharedValueIndex = 0; sharedValueIndex < sharedValueCount; sharedValueIndex++) {
						gradients.put(sharedValueIndexes[column][sharedValueIndex + sharedValueCount
								* grid + (hasBiasUnit() ? 1 : 0)],column + startColumnIndex, 
								averageValues[sharedValueIndex]);
					}
				}

			}
		}

	}

	public static DoubleMatrix createThetasMask(int filterCount, int depth,Integer stride,
			int inputNeuronCount, int outputNeuronCount, boolean hasBiasUnit) {

		DoubleMatrix thetasMask = new DoubleMatrix(
				inputNeuronCount,outputNeuronCount);
		if (hasBiasUnit) {
			thetasMask = DoubleMatrix.concatVertically(
					DoubleMatrix.ones(1,thetasMask.getColumns()), thetasMask);
		}

		int outputDim = (int) Math.sqrt(outputNeuronCount / filterCount);
		int inputDim = (int) Math.sqrt(inputNeuronCount / depth);

		int filterWidth = inputDim + (1 - outputDim) * (stride == null ? 1 : stride);
		int gridInputSize = inputNeuronCount / depth;
		int filterOutputSize = outputNeuronCount / filterCount;
		
		int strideAmount = stride == null ? 1 : stride;

		for (int grid = 0; grid < depth; grid++) {
			for (int f = 0; f < filterCount; f++) {
				for (int i = 0; i < outputDim; i++) {
					for (int j = 0; j < outputDim; j++) {
						for (int r = i * strideAmount; r < i * strideAmount + filterWidth; r++) {
							for (int c = j * strideAmount; c < j * strideAmount+ filterWidth; c++) {
								int outputInd = (filterOutputSize * f)
										+ (i * outputDim + j);
								int inputInd = grid * gridInputSize + r
										* inputDim + c + (hasBiasUnit ? 1 : 0);

								thetasMask.put( inputInd,outputInd, 1);
							}
						}
					}
				}
			}
		}
		return thetasMask;
	}

}
