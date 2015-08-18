package org.ml4j;

public interface MatrixAdapterFactory {

	MatrixAdapter createMatrix(int rows, int cols);

	MatrixAdapter createMatrix(int rows, int cols, double[] data);

	MatrixAdapter createMatrix();

	MatrixAdapter createMatrix(double[][] inputs);

	MatrixAdapter createMatrix(double[] inputToReconstruct);

	MatrixAdapter createOnes(int rows);

	MatrixAdapter createHorizontalConcatenation(MatrixAdapter matrix, MatrixAdapter matrix2);

	MatrixAdapter createVerticalConcatenation(MatrixAdapter matrix, MatrixAdapter matrix2);

	MatrixAdapter createOnes(int rows, int cols);

	MatrixAdapter createRandn(int outputNeuronCount, int i);

	MatrixAdapter createZeros(int rows, int cols);

	MatrixAdapter createRand(int i, int length);

}
