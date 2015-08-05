package org.ml4j.nn;

import org.jblas.DoubleMatrix;
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

	protected ConvolutionalLayer(int inputNeuronCount, int outputNeuronCount,
			DoubleMatrix thetas, DoubleMatrix thetasMask,
			DifferentiableActivationFunction activationFunction,
			boolean biasUnit, boolean retrainable, int filterCount, int depth,double inputDropout) {
		super(inputNeuronCount, outputNeuronCount, thetas, thetasMask,
				activationFunction, biasUnit, retrainable,inputDropout);
		this.filterCount = filterCount;
		this.depth = depth;
	}
	
	

	public int getDepth() {
		return depth;
	}

	public int getFilterCount() {
		return filterCount;
	}

	public ConvolutionalLayer(int inputNeuronCount, int outputNeuronCount,
			DifferentiableActivationFunction activationFunction,
			boolean biasUnit, int filterCount, int depth,double inputDropout) {
		super(inputNeuronCount, outputNeuronCount, createThetas(
				inputNeuronCount,
				outputNeuronCount,
				filterCount,
				depth,
				createThetasMask(filterCount, depth, inputNeuronCount,
						outputNeuronCount, biasUnit), biasUnit),
				createThetasMask(filterCount, depth, inputNeuronCount,
						outputNeuronCount, biasUnit), activationFunction,
				biasUnit, true,inputDropout);
		this.filterCount = filterCount;
		this.depth = depth;
	}
	
	
	public ConvolutionalLayer(int inputNeuronCount, int outputNeuronCount,
			DifferentiableActivationFunction activationFunction,
			boolean biasUnit, int filterCount, int depth) {
		this(inputNeuronCount,outputNeuronCount,activationFunction,biasUnit,filterCount,depth,1);
	}

	private static DoubleMatrix createThetas(int inputNeuronCount,
			int outputNeuronCount, int filterCount, int depth,
			DoubleMatrix thetasMask, boolean hasBiasUnit) {
		DoubleMatrix initialThetas = DoubleMatrix.randn(outputNeuronCount,
				inputNeuronCount + (hasBiasUnit ? 1 : 0)).mul(0.05);

		int filterOutputSize = outputNeuronCount / filterCount;

		for (int grid = 0; grid < depth; grid++) {
			for (int f = 0; f < filterCount; f++) {
				int startRowIndex = f * filterOutputSize;
				int inputWidth = (int) Math.sqrt(inputNeuronCount / depth);
				int outputWidth = (int) Math.sqrt(outputNeuronCount
						/ filterCount);
				int filterWidth = inputWidth - outputWidth + 1;
				int sharedValueCount = filterWidth * filterWidth;

				int[][] sharedValueIndexes = new int[filterOutputSize][sharedValueCount
						* depth + (hasBiasUnit ? 1 : 0)];
				double[] sharedValues = new double[sharedValueCount];
				for (int row = startRowIndex; row < startRowIndex
						+ filterOutputSize; row++) {
					int[] inds = thetasMask.getRow(row).findIndices();
					sharedValueIndexes[row - startRowIndex] = inds;

				}

				for (int i = 0; i < sharedValueCount; i++) {
					sharedValues[i] = initialThetas.get(startRowIndex,
							sharedValueIndexes[0][i + filterWidth * grid
									+ (hasBiasUnit ? 1 : 0)]);
				}

				for (int row = startRowIndex; row < startRowIndex
						+ filterOutputSize; row++) {
					for (int sharedValueIndex = 0; sharedValueIndex < sharedValueCount; sharedValueIndex++) {
						initialThetas.put(row, sharedValueIndexes[row
								- startRowIndex][sharedValueIndex + filterWidth
								* grid + (hasBiasUnit ? 1 : 0)],
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
				depth,inputDropout);
		return dup;
	}

	@Override
	public void applyGradientWeightConstraints(DoubleMatrix gradients) {

		int filterOutputSize = getOutputNeuronCount() / filterCount;

		for (int grid = 0; grid < depth; grid++) {
			for (int f = 0; f < filterCount; f++) {
				int startRowIndex = f * filterOutputSize;
				int inputWidth = (int) Math.sqrt(getInputNeuronCount() / depth);
				int outputWidth = (int) Math.sqrt(getOutputNeuronCount()
						/ filterCount);
				int filterWidth = inputWidth - outputWidth + 1;
				int sharedValueCount = filterWidth * filterWidth;

				int[][] sharedValueIndexes = new int[filterOutputSize][sharedValueCount
						+ (hasBiasUnit() ? 1 : 0)];
				double[] averageValues = new double[sharedValueCount];
				for (int row = startRowIndex; row < startRowIndex
						+ filterOutputSize; row++) {
					int[] inds = thetasMask.getRow(row).findIndices();
					sharedValueIndexes[row - startRowIndex] = inds;
					for (int i = 0; i < averageValues.length; i++) {

						averageValues[i] = averageValues[i]
								+ gradients.get(row, inds[filterWidth * grid
										+ i + (hasBiasUnit() ? 1 : 0)]);
					}
				}

				for (int i = 0; i < averageValues.length; i++) {
					averageValues[i] = averageValues[i] / filterOutputSize;
				}

				for (int row = startRowIndex; row < startRowIndex
						+ filterOutputSize; row++) {
					for (int sharedValueIndex = 0; sharedValueIndex < sharedValueCount; sharedValueIndex++) {
						gradients.put(row, sharedValueIndexes[row
								- startRowIndex][sharedValueIndex + filterWidth
								* grid + (hasBiasUnit() ? 1 : 0)],
								averageValues[sharedValueIndex]);
					}
				}

			}
		}

	}

	public static DoubleMatrix createThetasMask(int filterCount, int depth,
			int inputNeuronCount, int outputNeuronCount, boolean hasBiasUnit) {

		DoubleMatrix thetasMask = new DoubleMatrix(outputNeuronCount,
				inputNeuronCount);
		if (hasBiasUnit) {
			thetasMask = DoubleMatrix.concatHorizontally(
					DoubleMatrix.ones(thetasMask.getRows()), thetasMask);
		}

		int outputDim = (int) Math.sqrt(outputNeuronCount / filterCount);
		int inputDim = (int) Math.sqrt(inputNeuronCount / depth);

		int filterWidth = inputDim - outputDim + 1;
		int gridInputSize = inputNeuronCount / depth;
		int filterOutputSize = outputNeuronCount / filterCount;

		for (int grid = 0; grid < depth; grid++) {
			for (int f = 0; f < filterCount; f++) {
				for (int i = 0; i < outputDim; i++) {
					for (int j = 0; j < outputDim; j++) {
						for (int r = i; r < i + filterWidth; r++) {
							for (int c = j; c < j + filterWidth; c++) {
								int outputInd = (filterOutputSize * f)
										+ (i * outputDim + j);
								int inputInd = grid * gridInputSize + r
										* inputDim + c + (hasBiasUnit ? 1 : 0);

								thetasMask.put(outputInd, inputInd, 1);
							}
						}
					}
				}
			}
		}
		return thetasMask;
	}

}
