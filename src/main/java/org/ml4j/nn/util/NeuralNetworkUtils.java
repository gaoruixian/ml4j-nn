package org.ml4j.nn.util;

import java.util.Iterator;
import java.util.Vector;

import org.jblas.DoubleMatrix;
import org.jblas.MatrixFunctions;

public class NeuralNetworkUtils {

	/**
	 * Using given column matrix and given topology, takes elements from column
	 * matrix (possibly generated from reshapeToVector) and organizes them into
	 * a List (in this case Vector) of weight matrices based on given topology.
	 */
	public static Vector<DoubleMatrix> reshapeToList(DoubleMatrix x, int[] topology) {
		Vector<DoubleMatrix> result = new Vector<DoubleMatrix>();
		int layers = topology.length;

		int rows, cols;
		int offset = 0;
		for (int i = 0; i < layers - 1; i++) {
			rows = topology[i + 1];
			cols = topology[i] + 1;
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
		DoubleMatrix result = new DoubleMatrix();
		result = x;
		result = MatrixFunctions.expi(result.mul(-1));
		result = result.add(1);
		result = MatrixFunctions.powi(result, -1);
		return result;
	}

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

}
