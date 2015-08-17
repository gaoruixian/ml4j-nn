package org.ml4j.nn.util;

import java.util.Iterator;
import java.util.Vector;

import org.ml4j.DoubleMatrix;
import org.ml4j.DoubleMatrixConfig;
import org.ml4j.MatrixFunctions;

public class NeuralNetworkUtils {

	/**
	 * Using given column matrix and given topology, takes elements from column
	 * matrix (possibly generated from reshapeToVector) and organizes them into
	 * a List (in this case Vector) of weight matrices based on given topology.
	 */
	public static Vector<DoubleMatrix> reshapeToList(DoubleMatrix x, int[] topology,boolean[] biasUnits) {
		Vector<DoubleMatrix> result = new Vector<DoubleMatrix>();
		int layers = topology.length;

		int rows, cols;
		int offset = 0;
		for (int i = 0; i < layers - 1; i++) {
			rows = topology[i + 1];
			cols = topology[i] + (biasUnits[i] ? 1 : 0);
			DoubleMatrix Theta = new DoubleMatrix(rows, cols);
			for (int j = 0; j < cols; j++) {
				Theta.putColumn(j, x.getRowRange(offset, offset + rows, 0));
				offset += rows;
			}
			result.add(Theta);
		}
		return result;
	}

	/**
	 * Using given column matrix and given topology, takes elements from column
	 * matrix (possibly generated from reshapeToVector) and organizes them into
	 * a List (in this case Vector) of weight matrices based on given topology.
	 */
	public static Vector<DoubleMatrix> reshapeToList(DoubleMatrix x, int[][] topologies) {
		Vector<DoubleMatrix> result = new Vector<DoubleMatrix>();
		int layers = topologies.length;
		int rows, cols;
		int offset = 0;
		for (int i = 0; i < layers; i++) {
			int[] topology = topologies[i];
			rows = topology[0];
			cols = topology[1];
			DoubleMatrix Theta = new DoubleMatrix(rows, cols);
			for (int j = 0; j < cols; j++) {
				Theta.putColumn(j, x.getRowRange(offset, offset + rows, 0));
				offset += rows;
			}
			result.add(Theta);
		}
		return result;
	}

	/**
	 * Takes a List (in this case Vector) and takes each element of each
	 * DoubleMatrix and places it into a column matrix. note: the reason it is
	 * named reshapeToVector has to do with the resulting matrix, not the Java
	 * the Vector data structure note: can be undone with reshapeToList
	 * (assuming a neural network topology exists.
	 */
	public static DoubleMatrix reshapeToVector(Vector<DoubleMatrix> inputTheta) {

		Iterator<DoubleMatrix> iter = inputTheta.iterator();
		int length = 0;
		while (iter.hasNext()) {
			length += iter.next().getLength();
		}
		iter = inputTheta.iterator();
		DoubleMatrix result = new DoubleMatrix(length, 1);
		DoubleMatrix x;
		int offset = 0;
		while (iter.hasNext()) {
			x = iter.next();
			x.reshape(x.getLength(), 1);
			int[] indicies = new int[x.getLength()];
			for (int i = 0; i < x.getLength(); i++) {
				indicies[i] = offset + i;
			}
			offset += x.getLength();
			result.put(indicies, 0, x);
		}
		return result;
	}

	/**
	 * Returns a matrix that has the sigmoid function applied to each element of
	 * given input matrix http://en.wikipedia.org/wiki/Sigmoid_function
	 */
	
	public static DoubleMatrix sigmoid(DoubleMatrix x) {
		return new DoubleMatrix(DoubleMatrixConfig.getDoubleMatrixStrategy().sigmoid(x.matrix));
		/*
		DoubleMatrix result = new DoubleMatrix();
		result = x;
		result = MatrixFunctions.expi(result.mul(-1));
		result = result.add(1);
		result = MatrixFunctions.powi(result, -1);
		return result;
			*/
	}
	
	/*
	public static org.ml4j.jblas.DoubleMatrix sigmoid(org.ml4j.jblas.DoubleMatrix x) {
		DoubleMatrix result = new DoubleMatrix();
		result = x.matrix;
		result = MatrixFunctions.expi(result.mul(-1));
		result = result.add(1);
		result = MatrixFunctions.powi(result, -1);
		return new org.ml4j.jblas.DoubleMatrix(result);
	}
	*/

	/**
	 * Returns a matrix that has the first derivative of the sigmoid function
	 * applied to each element of given input matrix.
	 * http://en.wikipedia.org/wiki/Sigmoid_function
	 */
	public static DoubleMatrix sigmoidGradiant(DoubleMatrix x) {
		DoubleMatrix result = new DoubleMatrix();
		result = x;
		DoubleMatrix s = sigmoid(result);
		result = s.subi(s.mul(s));

		// result = s.mul( s.mul(-1).add(1) );
		return result;
	}

	/**
	 * Returns a matrix that has the first derivative of the sigmoid function
	 * applied to each element of given input matrix.
	 * http://en.wikipedia.org/wiki/Sigmoid_function
	 */
	public static DoubleMatrix softmaxGradient(DoubleMatrix x) {
		DoubleMatrix result = new DoubleMatrix();
		result = x;
		DoubleMatrix s = softmax(result);
		result = s.subi(s.mul(s));

		// result = s.mul( s.mul(-1).add(1) );
		return result;
	}

	/**
	 * Returns a matrix that has the sigmoid function applied to each element of
	 * given input matrix http://en.wikipedia.org/wiki/Sigmoid_function
	 */
	public static DoubleMatrix softmax(DoubleMatrix x) {
		DoubleMatrix exp = MatrixFunctions.expi(x);
		DoubleMatrix sums = exp.rowSums();
		return exp.diviColumnVector(sums);
	}
	
	public static DoubleMatrix removeInterceptColumn(DoubleMatrix in) {
		DoubleMatrix result = new DoubleMatrix(in.getRows(), in.getColumns() - 1);
		for (int i = 0; i < result.getRows(); i++) {
			for (int j = 0; j < result.getColumns() - 1; j++) {
				result.put(i, j, in.get(i, j + 1));
			}
		}
		return result;
	}
	
	public static DoubleMatrix removeInterceptRow(DoubleMatrix in) {
		DoubleMatrix result = new DoubleMatrix(in.getRows() - 1, in.getColumns());
		for (int i = 0; i < result.getRows() - 1; i++) {
			for (int j = 0; j < result.getColumns(); j++) {
				result.put(i, j, in.get(i + 1, j));
			}
		}
		return result;
	}

}
