package org.ml4j;

import java.io.Serializable;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.atomic.AtomicLong;

import org.ml4j.cuda.CudaMatrixAdapter;
import org.ml4j.jblas.JBlasMatrixAdapter;

public class DoubleMatrix implements Serializable, MatrixOperations<DoubleMatrix> {

	public MatrixAdapter matrix;

	

	public DoubleMatrix asJBlasMatrix() {
		if (matrix instanceof JBlasMatrixAdapter) {
			return this;
		} else {
			return new DoubleMatrix(JBlasMatrixAdapter.createJBlasBaseDoubleMatrix(matrix));
		}
	}

	public DoubleMatrix asCudaMatrix() {

	
		if (matrix instanceof CudaMatrixAdapter) {
			return this;
		} else {
			return new DoubleMatrix(CudaMatrixAdapter.createCudaBaseDoubleMatrix(matrix));
		}
	}

	public DoubleMatrix(MatrixAdapter matrix) {
		this.matrix = matrix;
	}

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;

	public DoubleMatrix(int rows, int cols) {
		this.matrix = DoubleMatrixConfig.getDoubleMatrixStrategy().getMatrixAdapterFactory().createMatrix(rows, cols);
	}

	public DoubleMatrix(int rows, int cols, double[] data) {
		this.matrix = DoubleMatrixConfig.getDoubleMatrixStrategy().getMatrixAdapterFactory().createMatrix(rows, cols, data);
	}

	public DoubleMatrix() {
		this.matrix = DoubleMatrixConfig.getDoubleMatrixStrategy().getMatrixAdapterFactory().createMatrix();

	}

	public DoubleMatrix(double[][] data) {
		this.matrix = DoubleMatrixConfig.getDoubleMatrixStrategy().getMatrixAdapterFactory().createMatrix(data);

	}

	public DoubleMatrix(double[] data) {
		this.matrix = DoubleMatrixConfig.getDoubleMatrixStrategy().getMatrixAdapterFactory().createMatrix(data);

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
		return new DoubleMatrix(DoubleMatrixConfig.getDoubleMatrixStrategy().getMatrixAdapterFactory().createOnes(rows));
	}

	public static DoubleMatrix concatHorizontally(DoubleMatrix left, DoubleMatrix right) {
		return new DoubleMatrix(DoubleMatrixConfig.getDoubleMatrixStrategy().getMatrixAdapterFactory().createHorizontalConcatenation(left.matrix, right.matrix));
	}

	public static DoubleMatrix concatVertically(DoubleMatrix top, DoubleMatrix bottom) {
		return new DoubleMatrix(DoubleMatrixConfig.getDoubleMatrixStrategy().getMatrixAdapterFactory().createVerticalConcatenation(top.matrix, bottom.matrix));
	}

	public static DoubleMatrix ones(int rows, int cols) {
		return new DoubleMatrix(DoubleMatrixConfig.getDoubleMatrixStrategy().getMatrixAdapterFactory().createOnes(rows, cols));
	}

	public DoubleMatrix mul(double scalingFactor) {
		return new DoubleMatrix(DoubleMatrixConfig.getDoubleMatrixStrategy().mul(this.matrix, scalingFactor));
	}

	public DoubleMatrix muli(double v) {
		DoubleMatrixConfig.getDoubleMatrixStrategy().muli(this.matrix, v);

		return this;
	}

	public void put(int r, int c, int v) {
		DoubleMatrixConfig.getDoubleMatrixStrategy().put(this.matrix, r, c, v);
	}

	public DoubleMatrix sub(DoubleMatrix m) {
		return new DoubleMatrix(DoubleMatrixConfig.getDoubleMatrixStrategy().sub(this.matrix, m.matrix));
	}

	public DoubleMatrix transpose() {
		return new DoubleMatrix(DoubleMatrixConfig.getDoubleMatrixStrategy().transpose(this.matrix));
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
		return new DoubleMatrix(DoubleMatrixConfig.getDoubleMatrixStrategy().getMatrixAdapterFactory().createRandn(r, c));
	}

	public DoubleMatrix getRow(int row) {
		return new DoubleMatrix(DoubleMatrixConfig.getDoubleMatrixStrategy().getRow(matrix, row));
	}

	public int[] findIndices() {
		return DoubleMatrixConfig.getDoubleMatrixStrategy().findIndices(matrix);
	}

	public double get(int i, int j) {
		return matrix.get(i, j);
	}

	public void put(int r, int c, double v) {
		DoubleMatrixConfig.getDoubleMatrixStrategy().put(matrix, r, c, v);
		;
	}

	public DoubleMatrix muli(DoubleMatrix m) {

		DoubleMatrixConfig.getDoubleMatrixStrategy().muli(matrix, m.matrix);
		return this;
	}

	public DoubleMatrix mmul(DoubleMatrix m) {

		return new DoubleMatrix(DoubleMatrixConfig.getDoubleMatrixStrategy().mmul(matrix, m.matrix));
	}

	public DoubleMatrix mul(DoubleMatrix m) {

		return new DoubleMatrix(DoubleMatrixConfig.getDoubleMatrixStrategy().mul(matrix, m.matrix));
	}

	public double sum() {
		return DoubleMatrixConfig.getDoubleMatrixStrategy().sum(matrix);
	}

	public static DoubleMatrix zeros(int rows, int cols) {
		return new DoubleMatrix(DoubleMatrixConfig.getDoubleMatrixStrategy().getMatrixAdapterFactory().createZeros(rows, cols));
	}

	public int[] rowArgmaxs() {
		return DoubleMatrixConfig.getDoubleMatrixStrategy().rowArgmaxs(matrix);
	}

	public DoubleMatrix get(int[] rows, int[] cols) {
		return new DoubleMatrix(DoubleMatrixConfig.getDoubleMatrixStrategy().get(matrix, rows, cols));
	}

	public void putColumn(int i, DoubleMatrix m) {
		DoubleMatrixConfig.getDoubleMatrixStrategy().putColumn(matrix, i, m.matrix);
	}

	public DoubleMatrix div(double v) {
		return new DoubleMatrix(DoubleMatrixConfig.getDoubleMatrixStrategy().div(matrix, v));
	}

	public DoubleMatrix add(DoubleMatrix m) {
		return new DoubleMatrix(DoubleMatrixConfig.getDoubleMatrixStrategy().add(matrix, m.matrix));
	}

	public DoubleMatrix subi(DoubleMatrix m) {
		DoubleMatrixConfig.getDoubleMatrixStrategy().subi(this.matrix, m.matrix);
		return this;
	}

	public DoubleMatrix divi(double v) {
		DoubleMatrixConfig.getDoubleMatrixStrategy().divi(this.matrix, v);

		return this;
	}

	public DoubleMatrix getColumns(int[] cols) {
		return new DoubleMatrix(DoubleMatrixConfig.getDoubleMatrixStrategy().getColumns(matrix, cols));
	}

	public DoubleMatrix getRows(int[] rows) {
		return new DoubleMatrix(DoubleMatrixConfig.getDoubleMatrixStrategy().getRows(matrix, rows));
	}

	public static DoubleMatrix rand(int r, int c) {
		return new DoubleMatrix(DoubleMatrixConfig.getDoubleMatrixStrategy().getMatrixAdapterFactory().createRand(r, c));
	}

	public DoubleMatrix addi(DoubleMatrix m) {
		DoubleMatrixConfig.getDoubleMatrixStrategy().addi(this.matrix, m.matrix);
		return this;
	}

	public DoubleMatrix getColumn(int j) {
		return new DoubleMatrix(DoubleMatrixConfig.getDoubleMatrixStrategy().getColumn(matrix, j));
	}

	public double get(int i) {
		return matrix.get(i);
	}

	public int argmax() {
		return DoubleMatrixConfig.getDoubleMatrixStrategy().argmax(matrix);
	}

	public DoubleMatrix add(double v) {
		return new DoubleMatrix(DoubleMatrixConfig.getDoubleMatrixStrategy().add(this.matrix, v));
	}

	public DoubleMatrix addi(double v) {
		DoubleMatrixConfig.getDoubleMatrixStrategy().addi(this.matrix, v);
		return this;
	}

	public DoubleMatrix rowSums() {
		return new DoubleMatrix(DoubleMatrixConfig.getDoubleMatrixStrategy().rowSums(matrix));
	}

	public int getLength() {
		return matrix.getLength();
	}

	public void put(int i, double log) {
		DoubleMatrixConfig.getDoubleMatrixStrategy().put(matrix, i, log);
	}

	public double dot(DoubleMatrix s) {
		return DoubleMatrixConfig.getDoubleMatrixStrategy().dot(this.matrix, s.matrix);
	}

	public DoubleMatrix getRowRange(int offset, int i, int j) {

		return new DoubleMatrix(DoubleMatrixConfig.getDoubleMatrixStrategy().getRowRange(this.matrix, offset, i, j));
	}

	public void reshape(int r, int c) {
		DoubleMatrixConfig.getDoubleMatrixStrategy().reshape(this.matrix, r, c);
	}

	public void put(int[] indicies, int inputInd, DoubleMatrix x) {
		DoubleMatrixConfig.getDoubleMatrixStrategy().put(this.matrix, indicies, inputInd, x);
		// matrix.put(indicies, inputInd,x.matrix);
	}

	public DoubleMatrix diviColumnVector(DoubleMatrix sums) {
		DoubleMatrixConfig.getDoubleMatrixStrategy().diviColumnVector(this.matrix, sums.matrix);
		// matrix.diviColumnVector(sums.matrix);
		return this;
	}

	public void putRow(int i, DoubleMatrix zeros) {
		DoubleMatrixConfig.getDoubleMatrixStrategy().putRow(this.matrix, i, zeros.matrix);
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
	

	public static void addTiming(String method, long time) {
		synchronized (methodTimings) {
			AtomicLong a = methodTimings.get(method);
			if (a == null) {
				a = new AtomicLong(0);
				methodTimings.put(method, a);
			}
			a.addAndGet(time);
		}

	}
	
	private static Map<String, AtomicLong> methodTimings = new HashMap<String, AtomicLong>();

	
	public boolean isCudaMatrix()
	{
		return this.matrix instanceof CudaMatrixAdapter;
	}
	
	public static void printTimings() {
		long t = 0;
		synchronized (methodTimings) {
			for (Map.Entry<String, AtomicLong> l : methodTimings.entrySet()) {
				if (!l.getKey().equals("total")) {
					long v = l.getValue().get();
					t = t + v;
				}
			}

			methodTimings.put("total", new AtomicLong(t));
			System.out.println(methodTimings);
			methodTimings.clear();
		}
	}

}
