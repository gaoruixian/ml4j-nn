package org.ml4j;

import java.io.Serializable;

import org.ml4j.jblas.JBlasMatrixAdapter;

public class DoubleMatrix implements Serializable, MatrixOperations<DoubleMatrix> {

	protected MatrixAdapter matrix;

	protected static MatrixAdapterStrategy strategy = new DefaultMatrixAdapterStrategy();

	public static void setDoubleMatrixStrategy(MatrixAdapterStrategy strategy1) {
		strategy = strategy1;
	}

	public DoubleMatrix asJBlasMatrix() {
		if (matrix instanceof JBlasMatrixAdapter) {
			return this;
		} else {
			return new DoubleMatrix(JBlasMatrixAdapter.createJBlasBaseDoubleMatrix(matrix));
		}
	}

	public DoubleMatrix asCudaMatrix() {
		throw new UnsupportedOperationException("Cuda matrices not yet supported");
	}

	public DoubleMatrix(MatrixAdapter matrix) {
		this.matrix = matrix;
	}

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;

	public DoubleMatrix(int rows, int cols) {
		this.matrix = strategy.createMatrix(rows, cols);
	}

	public DoubleMatrix(int rows, int cols, double[] data) {
		this.matrix = strategy.createMatrix(rows, cols, data);
	}

	public DoubleMatrix() {
		this.matrix = strategy.createMatrix();

	}

	public DoubleMatrix(double[][] data) {
		this.matrix = strategy.createMatrix(data);

	}

	public DoubleMatrix(double[] data) {
		this.matrix = strategy.createMatrix(data);

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

	public static DoubleMatrix ones(int rows) {
		return new DoubleMatrix(strategy.createOnes(rows));
	}

	public static DoubleMatrix concatHorizontally(DoubleMatrix left, DoubleMatrix right) {
		return new DoubleMatrix(strategy.concatHorizontally(left.matrix, right.matrix));
	}

	public static DoubleMatrix concatVertically(DoubleMatrix top, DoubleMatrix bottom) {
		return new DoubleMatrix(strategy.concatVertically(top.matrix, bottom.matrix));
	}

	public static DoubleMatrix ones(int rows, int cols) {
		return new DoubleMatrix(strategy.createOnes(rows, cols));
	}

	public DoubleMatrix mul(double scalingFactor) {
		return new DoubleMatrix(strategy.mul(this.matrix, scalingFactor));
	}

	public DoubleMatrix muli(double v) {
		strategy.muli(this.matrix, v);

		return this;
	}

	public void put(int r, int c, int v) {
		strategy.put(this.matrix, r, c, v);
	}

	public DoubleMatrix sub(DoubleMatrix m) {
		return new DoubleMatrix(strategy.sub(this.matrix, m.matrix));
	}

	public DoubleMatrix transpose() {
		return new DoubleMatrix(strategy.transpose(this.matrix));
	}

	public DoubleMatrix copy(DoubleMatrix m) {
		return new DoubleMatrix(this.matrix.copy(m.matrix));
	}

	public int getColumns() {
		return matrix.getColumns();
	}

	public DoubleMatrix dup() {
		return new DoubleMatrix(matrix.dup());
	}

	public static DoubleMatrix randn(int r, int c) {
		return new DoubleMatrix(strategy.createRandn(r, c));
	}

	public DoubleMatrix getRow(int row) {
		return new DoubleMatrix(strategy.getRow(matrix, row));
	}

	public int[] findIndices() {
		return strategy.findIndices(matrix);
	}

	public double get(int i, int j) {
		return matrix.get(i, j);
	}

	public void put(int r, int c, double v) {
		strategy.put(matrix, r, c, v);
		;
	}

	public DoubleMatrix muli(DoubleMatrix m) {

		strategy.muli(matrix, m.matrix);
		return this;
	}

	public DoubleMatrix mmul(DoubleMatrix m) {

		return new DoubleMatrix(strategy.mmul(matrix, m.matrix));
	}

	public DoubleMatrix mul(DoubleMatrix m) {

		return new DoubleMatrix(strategy.mul(matrix, m.matrix));
	}

	public double sum() {
		return strategy.sum(matrix);
	}

	public static DoubleMatrix zeros(int rows, int cols) {
		return new DoubleMatrix(strategy.createZeros(rows, cols));
	}

	public int[] rowArgmaxs() {
		return strategy.rowArgmaxs(matrix);
	}

	public DoubleMatrix get(int[] rows, int[] cols) {
		return new DoubleMatrix(strategy.get(matrix, rows, cols));
	}

	public void putColumn(int i, DoubleMatrix m) {
		strategy.putColumn(matrix, i, m.matrix);
	}

	public DoubleMatrix div(double v) {
		return new DoubleMatrix(strategy.div(matrix, v));
	}

	public DoubleMatrix add(DoubleMatrix m) {
		return new DoubleMatrix(strategy.add(matrix, m.matrix));
	}

	public DoubleMatrix subi(DoubleMatrix m) {
		strategy.subi(this.matrix, m.matrix);
		return this;
	}

	public DoubleMatrix divi(double v) {
		strategy.divi(this.matrix, v);

		return this;
	}

	public DoubleMatrix getColumns(int[] cols) {
		return new DoubleMatrix(strategy.getColumns(matrix, cols));
	}

	public DoubleMatrix getRows(int[] rows) {
		return new DoubleMatrix(strategy.getRows(matrix, rows));
	}

	public static DoubleMatrix rand(int r, int c) {
		return new DoubleMatrix(strategy.createRand(r, c));
	}

	public DoubleMatrix addi(DoubleMatrix m) {
		strategy.addi(this.matrix, m.matrix);
		return this;
	}

	public DoubleMatrix getColumn(int j) {
		return new DoubleMatrix(strategy.getColumn(matrix, j));
	}

	public double get(int i) {
		return matrix.get(i);
	}

	public int argmax() {
		return strategy.argmax(matrix);
	}

	public DoubleMatrix add(double v) {
		return new DoubleMatrix(strategy.add(this.matrix, v));
	}

	public DoubleMatrix addi(double v) {
		strategy.addi(this.matrix, v);
		return this;
	}

	public DoubleMatrix rowSums() {
		return new DoubleMatrix(strategy.rowSums(matrix));
	}

	public int getLength() {
		return matrix.getLength();
	}

	public void put(int i, double log) {
		strategy.put(matrix, i, log);
	}

	public double dot(DoubleMatrix s) {
		return strategy.dot(this.matrix, s.matrix);
	}

	public DoubleMatrix getRowRange(int offset, int i, int j) {

		return new DoubleMatrix(strategy.getRowRange(this.matrix, offset, i, j));
	}

	public void reshape(int r, int c) {
		strategy.reshape(this.matrix, r, c);
	}

	public void put(int[] indicies, int inputInd, DoubleMatrix x) {
		strategy.put(this.matrix, indicies, inputInd, x);
		// matrix.put(indicies, inputInd,x.matrix);
	}

	public DoubleMatrix diviColumnVector(DoubleMatrix sums) {
		strategy.diviColumnVector(this.matrix, sums.matrix);
		// matrix.diviColumnVector(sums.matrix);
		return this;
	}

	public void putRow(int i, DoubleMatrix zeros) {
		strategy.putRow(this.matrix, i, zeros.matrix);
		// matrix.putRow(i, zeros.matrix);
	}

	@Override
	public int hashCode() {
		final int prime = 31;
		int result = 1;
		result = prime * result + ((matrix == null) ? 0 : matrix.hashCode());
		return result;
	}

	@Override
	public boolean equals(Object obj) {
		if (this == obj)
			return true;
		if (obj == null)
			return false;
		if (getClass() != obj.getClass())
			return false;
		DoubleMatrix other = (DoubleMatrix) obj;
		if (matrix == null) {
			if (other.matrix != null)
				return false;
		} else if (!matrix.equals(other.matrix))
			return false;
		return true;
	}
	
	

}
