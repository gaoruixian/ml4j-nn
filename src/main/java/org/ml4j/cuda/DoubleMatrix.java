package org.ml4j.cuda;

import java.io.Serializable;
import java.util.Arrays;
import java.util.Date;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.atomic.AtomicLong;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.jcublas.SimpleJCublas;

public class DoubleMatrix implements Serializable {

	public INDArray matrix;

	private static Map<String, AtomicLong> methodTimings = new HashMap<String, AtomicLong>();

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

	public DoubleMatrix(INDArray matrix) {
		this.matrix = matrix;

	}

	protected DoubleMatrix(org.jblas.DoubleMatrix matrix) {

		this.matrix = createIndArray(matrix.toArray(), matrix.getRows(), matrix.getColumns());

	}

	public org.jblas.DoubleMatrix createJblasDoubleMatrix() {
		long s = new Date().getTime();
		org.jblas.DoubleMatrix m = (matrix == null) ? new org.jblas.DoubleMatrix() : new org.jblas.DoubleMatrix(
				matrix.rows(), matrix.columns(), matrix.data().asDouble());
		long e = new Date().getTime();
		addTiming("createJblas", e - s);
		return m;
	}

	protected INDArray createIndArray(double[] data, int rows, int cols) {
		long s = new Date().getTime();

		INDArray i = Nd4j.create(data, new int[] { rows, cols });
		long e = new Date().getTime();
		addTiming("createIndArray1", e - s);
		return i;
	}

	protected INDArray createIndArray(int rows, int cols) {
		long s = new Date().getTime();

		INDArray i = Nd4j.create(new double[rows * cols], new int[] { rows, cols });
		long e = new Date().getTime();
		addTiming("createIndArray2", e - s);
		return i;
	}

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;

	public DoubleMatrix(int rows, int cols) {
		this.matrix = createIndArray(rows, cols);

	}

	public DoubleMatrix() {
		this.matrix = null;

	}

	public DoubleMatrix(double[][] inputs) {

		this.matrix = createIndArray(new org.jblas.DoubleMatrix(inputs).toArray(), inputs.length, inputs[0].length);
	}

	public DoubleMatrix(double[] inputToReconstruct) {
		this.matrix = createIndArray(inputToReconstruct, inputToReconstruct.length, 1);
	}

	public DoubleMatrix(double[] inputToReconstruct, int rows, int columns) {
		this.matrix = createIndArray(inputToReconstruct, rows, columns);

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

	public static DoubleMatrix ones(int rows) {
		return new DoubleMatrix(org.jblas.DoubleMatrix.ones(rows));
	}

	public static DoubleMatrix concatHorizontally(DoubleMatrix A, DoubleMatrix B) {

		long s = new Date().getTime();

		DoubleMatrix m = new DoubleMatrix(A.getRows(), A.getColumns() + B.getColumns());

		if (A.getRows() == 1) {
			double[] data = new double[A.getColumns() + B.getColumns()];
			for (int i = 0; i < A.getColumns(); i++) {
				data[i] = A.get(0, i);
			}
			for (int i = 0; i < B.getColumns(); i++) {
				data[i + A.getColumns()] = B.get(0, i);
			}
			return new DoubleMatrix(data, 1, A.getColumns() + B.getColumns());
		}

		for (int i = 0; i < A.getColumns(); i++) {
			m.putColumn(i, A.getColumn(i));
		}
		for (int i = 0; i < B.getColumns(); i++) {
			m.putColumn(i + A.getColumns(), B.getColumn(i));
		}

		long e = new Date().getTime();
		addTiming("ch", e - s);

		return m;
	}

	public static DoubleMatrix ones(int rows, int cols) {
		return new DoubleMatrix(org.jblas.DoubleMatrix.ones(rows, cols));
	}

	public DoubleMatrix mul(double scalingFactor) {

		long s = new Date().getTime();
		DoubleMatrix r = new DoubleMatrix(matrix.mul(scalingFactor));
		long e = new Date().getTime();
		addTiming("mul1", e - s);
		return r;
	}

	public DoubleMatrix muli(double scalingFactor) {

		long s = new Date().getTime();
		DoubleMatrix r = new DoubleMatrix(matrix.muli(scalingFactor));
		long e = new Date().getTime();
		addTiming("muliDouble", e - s);
		return r;
	}

	public void put(int outputInd, int inputInd, int i) {
		long s = new Date().getTime();

		matrix.put(outputInd, inputInd, i);
		long e = new Date().getTime();
		addTiming("put2", e - s);

	}

	public DoubleMatrix sub(DoubleMatrix desiredOutputs) {
		long s = new Date().getTime();

		DoubleMatrix r = new DoubleMatrix(matrix.sub(desiredOutputs.matrix));
		long e = new Date().getTime();
		addTiming("sub2", e - s);

		return r;
	}

	public DoubleMatrix transpose() {
		long s = new Date().getTime();

		DoubleMatrix m = new DoubleMatrix(matrix.transpose());
		long e = new Date().getTime();
		addTiming("transpose1", e - s);
		return m;
	}

	public DoubleMatrix transpose(DoubleMatrix target) {

		long s = new Date().getTime();

		for (int i = 0; i < getRows(); i++) {
			target.matrix.putColumn(i, this.matrix.getRow(i));
		}

		long e = new Date().getTime();
		addTiming("transpose2", e - s);
		return target;
	}

	public DoubleMatrix copy(DoubleMatrix reshapeToVector) {
		long s = new Date().getTime();

		DoubleMatrix c = new DoubleMatrix(this.createJblasDoubleMatrix()
				.copy(reshapeToVector.createJblasDoubleMatrix()));
		long e = new Date().getTime();
		addTiming("copy", e - s);

		return c;
	}

	public int getColumns() {
		return matrix.columns();
	}

	public DoubleMatrix dup() {
		long s = new Date().getTime();

		DoubleMatrix d = new DoubleMatrix(matrix.dup());
		long e = new Date().getTime();
		addTiming("dup", e - s);

		return d;
	}

	public static DoubleMatrix randn(int outputNeuronCount, int i) {
		return new DoubleMatrix(org.jblas.DoubleMatrix.randn(outputNeuronCount, i));
	}

	public DoubleMatrix getRow(int row) {

		INDArray r = matrix.getRows(new int[] { row });
		return new DoubleMatrix(r);
	}

	public boolean sameSize(DoubleMatrix a) {
		return getRows() == a.getRows() && getColumns() == a.getColumns();
	}

	@Override
	public boolean equals(Object o) {
		if (!(o instanceof DoubleMatrix)) {
			return false;
		}

		DoubleMatrix other = (DoubleMatrix) o;

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

	public void put(int row, int inputInd, double d) {
		matrix.put(row, inputInd, d);
	}

	public DoubleMatrix muli(DoubleMatrix thetasMask) {

		long s = new Date().getTime();

		matrix.muli(thetasMask.matrix);

		long e = new Date().getTime();
		addTiming("muli", e - s);

		return this;
	}

	public DoubleMatrix mmul(DoubleMatrix mul) {

		long s = new Date().getTime();

		DoubleMatrix r = new DoubleMatrix(matrix.mmul(mul.matrix));
		long e = new Date().getTime();
		addTiming("mmul1", e - s);

		return r;
	}

	public DoubleMatrix mmul(DoubleMatrix mul, DoubleMatrix target) {

		// System.out.println("mmul");
		long s = new Date().getTime();

		DoubleMatrix m = new DoubleMatrix(matrix.mmul(mul.matrix, target.matrix));
		long e = new Date().getTime();
		addTiming("mmul2", e - s);

		return m;
	}

	public DoubleMatrix mul(DoubleMatrix dropoutMask) {
		// System.out.println("mul");
		long s = new Date().getTime();
		DoubleMatrix m = new DoubleMatrix(matrix.mul(dropoutMask.matrix));
		long e = new Date().getTime();

		addTiming("mul2", e - s);

		return m;
	}

	public DoubleMatrix mul(DoubleMatrix dropoutMask, DoubleMatrix target) {
		// System.out.println("mul");
		long s = new Date().getTime();

		DoubleMatrix m = new DoubleMatrix(matrix.mul(dropoutMask.matrix, target.matrix));

		long e = new Date().getTime();
		addTiming("mul3", e - s);

		return m;
	}

	public double sum() {
		long st = new Date().getTime();

		double s = 0.0;
		for (int i = 0; i < getLength(); i++) {
			s += get(i);
		}
		long e = new Date().getTime();
		addTiming("sum", e - st);
		return s;

	}

	public static DoubleMatrix zeros(int rows, int cols) {
		return new DoubleMatrix(org.jblas.DoubleMatrix.zeros(rows, cols));
	}

	public int[] rowArgmaxs() {
		return createJblasDoubleMatrix().rowArgmaxs();
	}

	public DoubleMatrix get(int[] rows, int[] cols) {
		return new DoubleMatrix(createJblasDoubleMatrix().get(rows, cols));
	}

	public void putColumn(int i, DoubleMatrix zeros) {
		matrix.putColumn(i, zeros.matrix);
	}

	public DoubleMatrix div(double m) {

		long s = new Date().getTime();

		DoubleMatrix m1 = new DoubleMatrix(matrix.div(m));
		long e = new Date().getTime();

		addTiming("div", e - s);

		return m1;
	}

	public DoubleMatrix add(DoubleMatrix mul) {
		long s = new Date().getTime();

		DoubleMatrix m = new DoubleMatrix(matrix.add(mul.matrix));
		long e = new Date().getTime();
		addTiming("add", e - s);
		return m;
	}

	public DoubleMatrix subi(DoubleMatrix mul) {
		long s = new Date().getTime();

		matrix.subi(mul.matrix);
		long e = new Date().getTime();

		addTiming("subi", e - s);

		return this;
	}

	public DoubleMatrix divi(double i) {
		long s = new Date().getTime();

		matrix.divi(i);
		long e = new Date().getTime();

		addTiming("divi", e - s);
		return this;
	}

	public DoubleMatrix getColumns(int[] hiddenOutputGradientColumnsForRecurrentOutputUnits) {

		if (matrix.rows() == 1) {
			double[] values = new double[hiddenOutputGradientColumnsForRecurrentOutputUnits.length];
			for (int i = 0; i < values.length; i++) {
				values[i] = hiddenOutputGradientColumnsForRecurrentOutputUnits[i];
			}
			return new DoubleMatrix(values, 1, values.length);
		} else {
			return new DoubleMatrix(matrix.getColumns(hiddenOutputGradientColumnsForRecurrentOutputUnits));
		}
	}

	public DoubleMatrix getRows(int[] inputHiddenGradientRowsForRecurrentHiddenUnits) {
		return new DoubleMatrix(matrix.getRows(inputHiddenGradientRowsForRecurrentHiddenUnits));
	}

	public static DoubleMatrix rand(int i, int length) {
		return new DoubleMatrix(org.jblas.DoubleMatrix.rand(i, length));
	}

	public DoubleMatrix addi(DoubleMatrix pairwiseVectorProduct) {
		long s = new Date().getTime();

		matrix.addi(pairwiseVectorProduct.matrix);
		long e = new Date().getTime();
		addTiming("addi", e - s);
		return this;

	}

	public DoubleMatrix getColumn(int j) {
		return new DoubleMatrix(matrix.getColumn(j));
	}

	public double get(int i) {
		return matrix.getDouble(i);
	}

	public int argmax() {

		return createJblasDoubleMatrix().argmax();
	}

	public DoubleMatrix add(double i) {
		long s = new Date().getTime();

		DoubleMatrix m = new DoubleMatrix(matrix.add(i));
		long e = new Date().getTime();
		addTiming("addDouble", e - s);
		return m;
	}

	public DoubleMatrix rowSums() {
		DoubleMatrix m = null;
		long s = new Date().getTime();
		if (this.getColumns() == 1) {
			m = dup();
		} else {
			DoubleMatrix v = new DoubleMatrix(getRows(), 1);

			for (int c = 0; c < getColumns(); c++) {
				for (int r = 0; r < getRows(); r++) {
					v.put(r, v.get(r) + get(r, c));
				}
			}

			m = v;
		}
		long e = new Date().getTime();
		addTiming("rowSums", e - s);
		return m;
	}

	public int getLength() {
		return matrix.rows() * matrix.columns();
	}

	public void put(int i, double log) {

		long s = new Date().getTime();

		matrix.putScalar(i, log);

		long e = new Date().getTime();

		addTiming("put", e - s);

	}

	public double dot(DoubleMatrix s) {

		long st = new Date().getTime();

		double d = SimpleJCublas.dot(this.matrix, s.matrix);

		long e = new Date().getTime();
		addTiming("dot", e - st);

		return d;
	}

	public DoubleMatrix getRowRange(int a, int b, int c) {

		DoubleMatrix result = new DoubleMatrix(b - a, 1);

		for (int k = 0; k < b - a; k++) {
			result.put(k, get(a + k, c));
		}

		return result;

	}

	public void put(int[] indicies, int inputInd, DoubleMatrix x) {

		org.jblas.DoubleMatrix m = createJblasDoubleMatrix();

		org.jblas.DoubleMatrix result = m.put(indicies, inputInd, x.createJblasDoubleMatrix());

		this.matrix = createIndArray(result.toArray(), result.getRows(), result.getColumns());
	}

	public DoubleMatrix diviColumnVector(DoubleMatrix sums) {

		long s = new Date().getTime();
		matrix.diviColumnVector(sums.matrix);
		long e = new Date().getTime();
		addTiming("diviColumnVector", e - s);
		return this;
	}

	public void reshape(int length, int i) {
		matrix.reshape(length, i);
	}

}
