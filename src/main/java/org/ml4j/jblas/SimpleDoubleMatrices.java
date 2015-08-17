package org.ml4j.jblas;

import java.util.Vector;

import org.ml4j.DoubleMatrices;
import org.ml4j.DoubleMatrix;

public class SimpleDoubleMatrices implements DoubleMatrices<DoubleMatrix> {

	private DoubleMatrix[] doubleMatrices;
	private int matrixCount;

	public SimpleDoubleMatrices(int matrixCount) {
		this.doubleMatrices = new DoubleMatrix[matrixCount];
		this.matrixCount = matrixCount;

	}

	public SimpleDoubleMatrices(DoubleMatrix[] matrices) {
		this.doubleMatrices = matrices;
		this.matrixCount = matrices.length;

	}

	public SimpleDoubleMatrices(Vector<DoubleMatrix> matrices) {
		this.doubleMatrices = matrices
				.toArray(new DoubleMatrix[matrices.size()]);
		this.matrixCount = matrices.size();

	}

	public int getMatrixCount() {
		return matrixCount;
	}

	public DoubleMatrix getMatrix(int index) {
		return doubleMatrices[index];
	}

	@Override
	public DoubleMatrices<DoubleMatrix> add(DoubleMatrices<DoubleMatrix> other) {

		int index = 0;
		DoubleMatrix[] result = new DoubleMatrix[other.getMatrixCount()];
		DoubleMatrix[] matrices = other.getMatrices();

		for (DoubleMatrix m : doubleMatrices) {
			result[index] = m.add(matrices[index]);
			index++;
		}

		return new SimpleDoubleMatrices(result);
	}

	@Override
	public DoubleMatrices<DoubleMatrix> addi(DoubleMatrices<DoubleMatrix> other) {

		int index = 0;
		DoubleMatrix[] result = new DoubleMatrix[other.getMatrixCount()];
		DoubleMatrix[] matrices = other.getMatrices();

		for (DoubleMatrix m : doubleMatrices) {
			result[index] = m.addi(matrices[index]);
			index++;
		}

		return new SimpleDoubleMatrices(result);
	}

	@Override
	public DoubleMatrices<DoubleMatrix> sub(DoubleMatrices<DoubleMatrix> other) {
		int index = 0;
		DoubleMatrix[] result = new DoubleMatrix[other.getMatrixCount()];
		DoubleMatrix[] matrices = other.getMatrices();
		for (DoubleMatrix m : doubleMatrices) {
			result[index] = m.sub(matrices[index]);
			index++;
		}

		return new SimpleDoubleMatrices(result);
	}

	@Override
	public DoubleMatrices<DoubleMatrix> subi(DoubleMatrices<DoubleMatrix> other) {
		int index = 0;
		DoubleMatrix[] result = new DoubleMatrix[other.getMatrixCount()];
		DoubleMatrix[] matrices = other.getMatrices();

		for (DoubleMatrix m : doubleMatrices) {
			result[index] = m.subi(matrices[index]);
			index++;
		}

		return new SimpleDoubleMatrices(result);
	}

	@Override
	public DoubleMatrices<DoubleMatrix> mul(double d) {
		int index = 0;
		DoubleMatrix[] result = new DoubleMatrix[getMatrixCount()];
		for (DoubleMatrix m : doubleMatrices) {
			result[index] = m.mul(d);
			index++;
		}

		return new SimpleDoubleMatrices(result);
	}

	@Override
	public DoubleMatrices<DoubleMatrix> muli(double d) {
		int index = 0;
		DoubleMatrix[] result = new DoubleMatrix[getMatrixCount()];
		for (DoubleMatrix m : doubleMatrices) {
			result[index] = m.muli(d);
			index++;
		}
		return new SimpleDoubleMatrices(result);
	}

	@Override
	public double dot(DoubleMatrices<DoubleMatrix> other) {
		int index = 0;
		double total = 0;
		DoubleMatrix[] matrices = other.getMatrices();

		for (DoubleMatrix m : doubleMatrices) {
			total = total + m.dot(matrices[index]);
			index++;
		}

		return total;
	}

	@Override
	public DoubleMatrices<DoubleMatrix> copy(DoubleMatrices<DoubleMatrix> source) {
		int index = 0;
		DoubleMatrix[] result = new DoubleMatrix[source.getMatrixCount()];
		DoubleMatrix[] matrices = source.getMatrices();

		for (DoubleMatrix m : doubleMatrices) {
			if (m == null) {
				m = new DoubleMatrix();
			}
			result[index] = m.copy(matrices[index]);
			index++;
		}

		return new SimpleDoubleMatrices(result);
	}

	@Override
	public DoubleMatrix[] getMatrices() {
		return doubleMatrices;
	}

}
