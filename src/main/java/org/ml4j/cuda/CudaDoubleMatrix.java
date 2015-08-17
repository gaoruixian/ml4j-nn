package org.ml4j.cuda;

import java.io.Serializable;
import java.util.Arrays;

import org.ml4j.MatrixOperations;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.jcublas.SimpleJCublas;

public class CudaDoubleMatrix implements Serializable, MatrixOperations<CudaDoubleMatrix> {

	protected INDArray matrix;

	protected void setMatrix(INDArray matrix) {
		this.matrix = matrix;
	}

	public CudaDoubleMatrix(INDArray matrix) {
		this.matrix = matrix;
	}

	public org.jblas.DoubleMatrix createJblasDoubleMatrix() {
		org.jblas.DoubleMatrix m = (matrix == null) ? new org.jblas.DoubleMatrix() : new org.jblas.DoubleMatrix(
				matrix.rows(), matrix.columns(), matrix.data().asDouble());
		return m;
	}

	protected CudaDoubleMatrix(org.jblas.DoubleMatrix matrix) {
		this.matrix = createIndArray(matrix.toArray(), matrix.getRows(), matrix.getColumns());
	}

	protected INDArray createIndArray(double[] data, int rows, int cols) {

		INDArray i = Nd4j.create(data, new int[] { rows, cols });
		return i;
	}

	protected INDArray createIndArray(int rows, int cols) {

		INDArray i = Nd4j.create(new double[rows * cols], new int[] { rows, cols });
		return i;
	}

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;

	public CudaDoubleMatrix(int rows, int cols) {
		this.matrix = createIndArray(rows, cols);

	}

	public CudaDoubleMatrix() {
		this.matrix = null;

	}

	public CudaDoubleMatrix(double[][] data) {

		this.matrix = createIndArray(new org.jblas.DoubleMatrix(data).toArray(), data.length, data[0].length);
	}

	public CudaDoubleMatrix(double[] data) {
		this.matrix = createIndArray(data, data.length, 1);
	}

	public CudaDoubleMatrix(double[] data, int rows, int columns) {
		this.matrix = createIndArray(data, rows, columns);

	}

	public double[][] toArray2() {
		return new org.jblas.DoubleMatrix(matrix.rows(), matrix.columns(), matrix.data().asDouble()).toArray2();
	}

	public double[] toArray() {
		return matrix.data().asDouble();
	}

	public int getRows() {
		return matrix.rows();
	}

	public static CudaDoubleMatrix ones(int rows) {
		return new CudaDoubleMatrix(org.jblas.DoubleMatrix.ones(rows));
	}

	public static CudaDoubleMatrix concatHorizontally(CudaDoubleMatrix A, CudaDoubleMatrix B) {

		CudaDoubleMatrix m = new CudaDoubleMatrix(A.getRows(), A.getColumns() + B.getColumns());

		if (A.getRows() == 1) {
			double[] data = new double[A.getColumns() + B.getColumns()];
			for (int i = 0; i < A.getColumns(); i++) {
				data[i] = A.get(0, i);
			}
			for (int i = 0; i < B.getColumns(); i++) {
				data[i + A.getColumns()] = B.get(0, i);
			}
			return new CudaDoubleMatrix(data, 1, A.getColumns() + B.getColumns());
		}

		for (int i = 0; i < A.getColumns(); i++) {
			m.putColumn(i, A.getColumn(i));
		}
		for (int i = 0; i < B.getColumns(); i++) {
			m.putColumn(i + A.getColumns(), B.getColumn(i));
		}

		return m;
	}

	public static CudaDoubleMatrix concatVertically(CudaDoubleMatrix A, CudaDoubleMatrix B) {

		CudaDoubleMatrix m = new CudaDoubleMatrix(A.getRows() + B.getRows(), A.getColumns());

		if (A.getColumns() == 1) {
			double[] data = new double[A.getRows() + B.getRows()];
			for (int i = 0; i < A.getRows(); i++) {
				data[i] = A.get(i, 0);
			}
			for (int i = 0; i < B.getRows(); i++) {
				data[i + A.getRows()] = B.get(i, 0);
			}
			return new CudaDoubleMatrix(data, A.getRows() + B.getRows(), 1);
		}

		for (int i = 0; i < A.getRows(); i++) {
			m.putRow(i, A.getRow(i));
		}
		for (int i = 0; i < B.getRows(); i++) {
			m.putRow(i + A.getRows(), B.getRow(i));
		}

		return m;
	}

	public static CudaDoubleMatrix ones(int rows, int cols) {
		return new CudaDoubleMatrix(org.jblas.DoubleMatrix.ones(rows, cols));
	}

	public CudaDoubleMatrix mul(double v) {

		CudaDoubleMatrix r = new CudaDoubleMatrix(matrix.mul(v));
		return r;
	}

	public CudaDoubleMatrix muli(double v) {

		CudaDoubleMatrix r = new CudaDoubleMatrix(matrix.muli(v));
		return r;
	}

	public CudaDoubleMatrix sub(CudaDoubleMatrix m) {

		CudaDoubleMatrix r = new CudaDoubleMatrix(matrix.sub(m.matrix));

		return r;
	}

	public CudaDoubleMatrix transpose() {

		CudaDoubleMatrix m = new CudaDoubleMatrix(matrix.transpose());
		return m;
	}

	public CudaDoubleMatrix copy(CudaDoubleMatrix m) {

		CudaDoubleMatrix c = new CudaDoubleMatrix(this.createJblasDoubleMatrix().copy(m.createJblasDoubleMatrix()));

		return c;
	}

	public int getColumns() {
		return matrix.columns();
	}

	public CudaDoubleMatrix dup() {

		CudaDoubleMatrix d = new CudaDoubleMatrix(matrix.dup());

		return d;
	}

	public static CudaDoubleMatrix randn(int r, int c) {
		return new CudaDoubleMatrix(org.jblas.DoubleMatrix.randn(r, c));
	}

	public CudaDoubleMatrix getRow(int row) {

		INDArray r = matrix.getRows(new int[] { row });
		return new CudaDoubleMatrix(r);
	}

	public boolean sameSize(CudaDoubleMatrix a) {
		return getRows() == a.getRows() && getColumns() == a.getColumns();
	}

	@Override
	public boolean equals(Object o) {
		if (!(o instanceof CudaDoubleMatrix)) {
			return false;
		}

		CudaDoubleMatrix other = (CudaDoubleMatrix) o;

		if (!sameSize(other)) {
			return false;
		} else {
			return Arrays.equals(matrix.data().asDouble(), other.matrix.data().asDouble());
		}
	}

	@Override
	public int hashCode() {
		int hash = 7;
		hash = 83 * hash + this.getRows();
		hash = 83 * hash + this.getColumns();
		hash = 83 * hash + Arrays.hashCode(this.matrix.data().asDouble());
		return hash;
	}

	public int[] findIndices() {
		int len = 0;
		for (int i = 0; i < getLength(); i++) {
			if (get(i) != 0.0) {
				len++;
			}
		}

		int[] indices = new int[len];
		int c = 0;

		for (int i = 0; i < getLength(); i++) {
			if (get(i) != 0.0) {
				indices[c++] = i;
			}
		}

		return indices;
	}

	public double get(int i, int j) {
		return matrix.getDouble(i, j);
	}

	public void put(int row, int col, double v) {
		matrix.put(row, col, v);
	}

	public CudaDoubleMatrix muli(CudaDoubleMatrix m) {

		matrix.muli(m.matrix);

		return this;
	}

	public CudaDoubleMatrix mmul(CudaDoubleMatrix m) {

		CudaDoubleMatrix r = new CudaDoubleMatrix(matrix.mmul(m.matrix));
		return r;
	}

	public CudaDoubleMatrix mul(CudaDoubleMatrix m) {
		CudaDoubleMatrix r = new CudaDoubleMatrix(matrix.mul(m.matrix));
		return r;
	}

	public double sum() {

		double s = 0.0;
		for (int i = 0; i < getLength(); i++) {
			s += get(i);
		}
		return s;

	}

	public static CudaDoubleMatrix zeros(int rows, int cols) {
		return new CudaDoubleMatrix(org.jblas.DoubleMatrix.zeros(rows, cols));
	}

	public int[] rowArgmaxs() {
		return createJblasDoubleMatrix().rowArgmaxs();
	}

	public CudaDoubleMatrix get(int[] rows, int[] cols) {
		return new CudaDoubleMatrix(createJblasDoubleMatrix().get(rows, cols));
	}

	public void putColumn(int i, CudaDoubleMatrix m) {
		matrix.putColumn(i, m.matrix);
	}

	public void putRow(int i, CudaDoubleMatrix m) {
		matrix.putRow(i, m.matrix);
	}

	public CudaDoubleMatrix div(double v) {

		CudaDoubleMatrix m1 = new CudaDoubleMatrix(matrix.div(v));

		return m1;
	}

	public CudaDoubleMatrix add(CudaDoubleMatrix m) {

		CudaDoubleMatrix r = new CudaDoubleMatrix(matrix.add(m.matrix));
		return r;
	}

	public CudaDoubleMatrix subi(CudaDoubleMatrix m) {

		matrix.subi(m.matrix);

		return this;
	}

	public CudaDoubleMatrix divi(double i) {

		matrix.divi(i);

		return this;
	}

	public CudaDoubleMatrix getColumns(int[] cols) {

		if (matrix.rows() == 1) {
			double[] values = new double[cols.length];
			for (int i = 0; i < values.length; i++) {
				values[i] = cols[i];
			}
			return new CudaDoubleMatrix(values, 1, values.length);
		} else {
			return new CudaDoubleMatrix(matrix.getColumns(cols));
		}
	}

	public CudaDoubleMatrix getRows(int[] rows) {
		return new CudaDoubleMatrix(matrix.getRows(rows));
	}

	public static CudaDoubleMatrix rand(int r, int c) {
		return new CudaDoubleMatrix(org.jblas.DoubleMatrix.rand(r, c));
	}

	public CudaDoubleMatrix addi(CudaDoubleMatrix m) {

		matrix.addi(m.matrix);
		return this;

	}

	public CudaDoubleMatrix getColumn(int j) {
		return new CudaDoubleMatrix(matrix.getColumn(j));
	}

	public double get(int i) {
		return matrix.getDouble(i);
	}

	public int argmax() {

		return createJblasDoubleMatrix().argmax();
	}

	public CudaDoubleMatrix add(double v) {

		CudaDoubleMatrix m = new CudaDoubleMatrix(matrix.add(v));

		return m;
	}

	public CudaDoubleMatrix addi(double v) {

		matrix.add(v);

		return this;
	}

	public CudaDoubleMatrix rowSums() {
		CudaDoubleMatrix m = null;
		if (this.getColumns() == 1) {
			m = dup();
		} else {
			CudaDoubleMatrix v = new CudaDoubleMatrix(getRows(), 1);

			for (int c = 0; c < getColumns(); c++) {
				for (int r = 0; r < getRows(); r++) {
					v.put(r, v.get(r) + get(r, c));
				}
			}

			m = v;
		}

		return m;
	}

	public int getLength() {
		return matrix.rows() * matrix.columns();
	}

	public void put(int i, double v) {

		matrix.putScalar(i, v);

	}

	public double dot(CudaDoubleMatrix s) {

		double d = SimpleJCublas.dot(this.matrix, s.matrix);

		return d;
	}

	public CudaDoubleMatrix getRowRange(int a, int b, int c) {

		CudaDoubleMatrix result = new CudaDoubleMatrix(b - a, 1);

		for (int k = 0; k < b - a; k++) {
			result.put(k, get(a + k, c));
		}

		return result;

	}

	public void put(int[] indicies, int inputInd, CudaDoubleMatrix x) {

		org.jblas.DoubleMatrix m = createJblasDoubleMatrix();

		org.jblas.DoubleMatrix result = m.put(indicies, inputInd, x.createJblasDoubleMatrix());

		this.matrix = createIndArray(result.toArray(), result.getRows(), result.getColumns());
	}

	public CudaDoubleMatrix diviColumnVector(CudaDoubleMatrix m) {

		matrix.diviColumnVector(m.matrix);

		return this;
	}

	public void reshape(int r, int c) {
		matrix.reshape(r, c);
	}

	public CudaDoubleMatrix getMatrix() {
		return this;
	}

}
