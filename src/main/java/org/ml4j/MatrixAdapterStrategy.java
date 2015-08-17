package org.ml4j;


public interface MatrixAdapterStrategy {

	MatrixAdapter createMatrix(int rows, int cols);

	MatrixAdapter createMatrix(int rows, int cols, double[] data);

	MatrixAdapter createMatrix();

	MatrixAdapter createMatrix(double[][] inputs);

	MatrixAdapter createMatrix(double[] inputToReconstruct);

	MatrixAdapter createOnes(int rows);

	MatrixAdapter concatHorizontally(MatrixAdapter matrix, MatrixAdapter matrix2);

	MatrixAdapter concatVertically(MatrixAdapter matrix, MatrixAdapter matrix2);

	MatrixAdapter createOnes(int rows, int cols);

	MatrixAdapter createRandn(int outputNeuronCount, int i);

	MatrixAdapter createZeros(int rows, int cols);

	MatrixAdapter createRand(int i, int length);

	MatrixAdapter mul(MatrixAdapter matrix, MatrixAdapter matrix2);

	MatrixAdapter pow(MatrixAdapter matrix, int i);

	MatrixAdapter log(MatrixAdapter matrix);

	void expi(MatrixAdapter matrix);

	void powi(MatrixAdapter matrix, int d);

	void logi(MatrixAdapter matrix);

	MatrixAdapter mmul(MatrixAdapter matrix, MatrixAdapter matrix2);

	void putRow(MatrixAdapter matrix, int i, MatrixAdapter matrix2);

	void diviColumnVector(MatrixAdapter matrix, MatrixAdapter matrix2);

	void put(MatrixAdapter matrix, int[] indicies, int inputInd, DoubleMatrix x);

	void reshape(MatrixAdapter matrix, int length, int i);

	void addi(MatrixAdapter matrix, double i);

	MatrixAdapter add(MatrixAdapter matrix, double i);

	void addi(MatrixAdapter matrix, MatrixAdapter pairwiseVectorProduct);

	void subi(MatrixAdapter matrix, MatrixAdapter matrix2);

	void divi(MatrixAdapter matrix, double i);

	MatrixAdapter mul(MatrixAdapter matrix, double scalingFactor);

	void muli(MatrixAdapter matrix, double scalingFactor);


	MatrixAdapter sub(MatrixAdapter matrix, MatrixAdapter desiredOutputs);

	MatrixAdapter transpose(MatrixAdapter matrix);

	MatrixAdapter getRowRange(MatrixAdapter matrix, int offset, int i, int j);

	double dot(MatrixAdapter matrix, MatrixAdapter matrix2);

	void put(MatrixAdapter matrix, int i, double log);

	MatrixAdapter rowSums(MatrixAdapter matrix);

	MatrixAdapter add(MatrixAdapter matrix, MatrixAdapter matrix2);

	MatrixAdapter div(MatrixAdapter matrix, double m);

	void putColumn(MatrixAdapter matrix, int i, MatrixAdapter zeros);

	void muli(MatrixAdapter matrix, MatrixAdapter matrix2);

	void put(MatrixAdapter matrix, int row, int inputInd, double d);

	MatrixAdapter getRow(MatrixAdapter matrix, int row);

	int[] findIndices(MatrixAdapter matrix);

	double sum(MatrixAdapter matrix);

	int[] rowArgmaxs(MatrixAdapter matrix);

	MatrixAdapter get(MatrixAdapter matrix,int[] rows, int[] cols);

	MatrixAdapter getColumns(MatrixAdapter matrix, int[] hiddenOutputGradientColumnsForRecurrentOutputUnits);

	MatrixAdapter getRows(MatrixAdapter matrix, int[] inputHiddenGradientRowsForRecurrentHiddenUnits);

	MatrixAdapter getColumn(MatrixAdapter matrix, int j);

	int argmax(MatrixAdapter matrix);


	

}
