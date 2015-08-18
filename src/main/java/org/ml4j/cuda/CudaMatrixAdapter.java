package org.ml4j.cuda;

import org.ml4j.MatrixAdapter;
import org.ml4j.jblas.JBlasMatrixAdapter;

public class CudaMatrixAdapter implements MatrixAdapter {

	public CudaDoubleMatrix matrix;

	public CudaMatrixAdapter(CudaDoubleMatrix matrix) {
		this.matrix = matrix;
	}

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;

	public CudaMatrixAdapter(int rows, int cols) {
		this.matrix = new CudaDoubleMatrix(rows, cols);
	}

	public CudaMatrixAdapter(int rows, int cols, double[] data) {
		this.matrix = new CudaDoubleMatrix(data, rows, cols);
	}

	public CudaMatrixAdapter() {
		this.matrix = new CudaDoubleMatrix();

	}

	public CudaMatrixAdapter(double[][] inputs) {
		this.matrix = new CudaDoubleMatrix(inputs);

	}

	public CudaMatrixAdapter(double[] inputToReconstruct) {
		this.matrix = new CudaDoubleMatrix(inputToReconstruct);

	}

	public double[][] toArray2() {
		return matrix.toArray2();
	}

	public double[] toArray() {
		return matrix.toArray();
	}

	public int getRows() {
		return matrix.getRows();
	}

	public static CudaDoubleMatrix createCudaDoubleMatrix(MatrixAdapter baseDoubleMatrix) {

		if (baseDoubleMatrix instanceof CudaMatrixAdapter) {
			return ((CudaMatrixAdapter) baseDoubleMatrix).matrix;
		}
		return new CudaDoubleMatrix(baseDoubleMatrix.toArray(), baseDoubleMatrix.getRows(),
				baseDoubleMatrix.getColumns());

	}

	public static CudaMatrixAdapter createCudaBaseDoubleMatrix(MatrixAdapter baseDoubleMatrix) {

		if (baseDoubleMatrix instanceof CudaMatrixAdapter) {
			return ((CudaMatrixAdapter) baseDoubleMatrix);
		}
		return new CudaMatrixAdapter(baseDoubleMatrix.getRows(), baseDoubleMatrix.getColumns(),
				baseDoubleMatrix.toArray());

	}

	public CudaMatrixAdapter mul(double scalingFactor) {

		CudaMatrixAdapter ret = new CudaMatrixAdapter(matrix.mul(scalingFactor));

		return ret;
	}

	public CudaMatrixAdapter muli(double scalingFactor) {
		matrix.muli(scalingFactor);

		return this;
	}

	public void put(int outputInd, int inputInd, int i) {
		matrix.put(outputInd, inputInd, i);
	}

	public MatrixAdapter sub(MatrixAdapter desiredOutputs) {
		return new CudaMatrixAdapter(matrix.sub(createCudaDoubleMatrix(desiredOutputs)));
	}

	public MatrixAdapter transpose() {

		return new CudaMatrixAdapter(matrix.transpose());
	}

	public MatrixAdapter copy(MatrixAdapter reshapeToVector) {

		return new CudaMatrixAdapter(this.matrix.copy(createCudaDoubleMatrix(reshapeToVector)));
	}

	public int getColumns() {
		return matrix.getColumns();
	}

	public MatrixAdapter dup() {
		return new CudaMatrixAdapter(matrix.dup());
	}

	public MatrixAdapter getRow(int row) {

		return new CudaMatrixAdapter(matrix.getRow(row));
	}

	public int[] findIndices() {

		return matrix.findIndices();
	}

	public double get(int i, int j) {
		return matrix.get(i, j);
	}

	public void put(int row, int inputInd, double d) {
		matrix.put(row, inputInd, d);

	}

	public MatrixAdapter muli(MatrixAdapter thetasMask) {

		matrix.muli(createCudaDoubleMatrix(thetasMask));

		return this;
	}

	public MatrixAdapter mmul(MatrixAdapter mul) {

		return new CudaMatrixAdapter(matrix.mmul(createCudaDoubleMatrix(mul)));
	}

	public MatrixAdapter mul(MatrixAdapter dropoutMask) {
		return new CudaMatrixAdapter(matrix.mul(createCudaDoubleMatrix(dropoutMask)));
	}

	public double sum() {

		return matrix.sum();
	}

	public int[] rowArgmaxs() {

		return matrix.rowArgmaxs();
	}

	public MatrixAdapter get(int[] rows, int[] cols) {

		return new CudaMatrixAdapter(matrix.get(rows, cols));

	}

	public void putColumn(int i, MatrixAdapter zeros) {

		matrix.putColumn(i, createCudaDoubleMatrix(zeros));
	}

	public MatrixAdapter div(double m) {

		return new CudaMatrixAdapter(matrix.div(m));
	}

	public MatrixAdapter add(MatrixAdapter mul) {

		return new CudaMatrixAdapter(matrix.add(createCudaDoubleMatrix(mul)));
	}

	public MatrixAdapter subi(MatrixAdapter mul) {

		matrix.subi(createCudaDoubleMatrix(mul));
		return this;
	}

	public MatrixAdapter divi(double i) {
		matrix.divi(i);
		return this;
	}

	public MatrixAdapter getColumns(int[] hiddenOutputGradientColumnsForRecurrentOutputUnits) {

		return new CudaMatrixAdapter(matrix.getColumns(hiddenOutputGradientColumnsForRecurrentOutputUnits));

	}

	public MatrixAdapter getRows(int[] inputHiddenGradientRowsForRecurrentHiddenUnits) {
		MatrixAdapter ret = new CudaMatrixAdapter(matrix.getRows(inputHiddenGradientRowsForRecurrentHiddenUnits));
		return ret;
	}

	public MatrixAdapter addi(MatrixAdapter pairwiseVectorProduct) {
		matrix.addi(createCudaDoubleMatrix(pairwiseVectorProduct));
		return this;

	}

	public MatrixAdapter getColumn(int j) {

		return new CudaMatrixAdapter(matrix.getColumn(j));

	}

	public double get(int i) {
		return matrix.get(i);
	}

	public int argmax() {
		return matrix.argmax();
	}

	public MatrixAdapter add(double i) {

		return new CudaMatrixAdapter(matrix.add(i));
	}

	public MatrixAdapter addi(double i) {

		matrix.addi(i);
		return this;
	}

	public MatrixAdapter rowSums() {
		return new CudaMatrixAdapter(matrix.rowSums());
	}

	public int getLength() {
		return matrix.getLength();
	}

	public void put(int i, double log) {
		matrix.put(i, log);
	}

	public double dot(MatrixAdapter s) {

		return matrix.dot(createCudaDoubleMatrix(s));
	}

	public MatrixAdapter getRowRange(int offset, int i, int j) {
		return new CudaMatrixAdapter(matrix.getRowRange(offset, i, j));
	}

	public void reshape(int length, int i) {
		matrix.reshape(length, i);
	}

	public void put(int[] indicies, int inputInd, MatrixAdapter x) {
		matrix.put(indicies, inputInd, createCudaDoubleMatrix(x));
	}

	public MatrixAdapter diviColumnVector(MatrixAdapter sums) {
		matrix.diviColumnVector(createCudaDoubleMatrix(sums));
		return this;
	}

	public void putRow(int i, MatrixAdapter zeros) {
		matrix.putRow(i, createCudaDoubleMatrix(zeros));
	}

	@Override
	public MatrixAdapter pow(int i) {

		return new CudaMatrixAdapter(CudaMatrixFunctions.pow(this.matrix, i));

	}

	@Override
	public MatrixAdapter log() {

		return new CudaMatrixAdapter(CudaMatrixFunctions.log(this.matrix));
	}

	@Override
	public MatrixAdapter expi() {
		CudaMatrixFunctions.expi(this.matrix);
		return this;
	}

	@Override
	public MatrixAdapter powi(int d) {

		CudaMatrixFunctions.powi(this.matrix, d);
		return this;
	}

	@Override
	public MatrixAdapter logi() {

		return new CudaMatrixAdapter(CudaMatrixFunctions.logi(this.matrix));
	}

	public MatrixAdapter asJBlasMatrix() {
		return JBlasMatrixAdapter.createJBlasBaseDoubleMatrix(this);
	}

	public MatrixAdapter asCudaMatrix() {
		return this;
	}

	@Override
	public MatrixAdapter sigmoid() {
		{
			return new CudaMatrixAdapter(CudaMatrixFunctions.sigmoid(matrix));
		}

	}

}
