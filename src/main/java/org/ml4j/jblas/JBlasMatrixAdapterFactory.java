package org.ml4j.jblas;

import org.ml4j.DoubleMatrix;
import org.ml4j.MatrixAdapter;
import org.ml4j.MatrixAdapterFactory;

import com.google.common.base.Stopwatch;

public class JBlasMatrixAdapterFactory implements MatrixAdapterFactory {

	@Override
	public MatrixAdapter createMatrix(int rows, int cols) {
		return new JBlasMatrixAdapter(rows,cols);
	}

	@Override
	public MatrixAdapter createMatrix(int rows, int cols, double[] data) {
		return new JBlasMatrixAdapter(rows,cols,data);
	}

	@Override
	public MatrixAdapter createMatrix() {
		return new JBlasMatrixAdapter();
	}

	@Override
	public MatrixAdapter createMatrix(double[][] data) {
		return new JBlasMatrixAdapter(data);
	}

	@Override
	public MatrixAdapter createMatrix(double[] data) {
		return new JBlasMatrixAdapter(data);
	}

	@Override
	public MatrixAdapter createOnes(int rows) {
		return new JBlasMatrixAdapter(JBlasDoubleMatrix.ones(rows));
	}

	@Override
	public MatrixAdapter createHorizontalConcatenation(MatrixAdapter matrix, MatrixAdapter matrix2) {
		Stopwatch timer = createStartedTimer();
		MatrixAdapter ret =  new JBlasMatrixAdapter(JBlasDoubleMatrix.concatHorizontally(JBlasMatrixAdapter.createJBlasDoubleMatrix(matrix),JBlasMatrixAdapter.createJBlasDoubleMatrix(matrix2)));
		DoubleMatrix.addTiming("createHorizontalConcatentationJblas",timer.elapsedMillis());
		return ret;
	}
	

	@Override
	public MatrixAdapter createVerticalConcatenation(MatrixAdapter matrix, MatrixAdapter matrix2) {
		Stopwatch timer = createStartedTimer();

		MatrixAdapter ret =  new JBlasMatrixAdapter(JBlasDoubleMatrix.concatVertically(JBlasMatrixAdapter.createJBlasDoubleMatrix(matrix),JBlasMatrixAdapter.createJBlasDoubleMatrix(matrix2)));
		DoubleMatrix.addTiming("createVerticalConcatentationJblas",timer.elapsedMillis());

		return ret;
	}

	@Override
	public MatrixAdapter createOnes(int rows, int cols) {
		return new JBlasMatrixAdapter(JBlasDoubleMatrix.ones(rows,cols));
	}

	@Override
	public MatrixAdapter createRandn(int r, int c) {
		return new JBlasMatrixAdapter(JBlasDoubleMatrix.randn(r,c));
	}

	@Override
	public MatrixAdapter createZeros(int rows, int cols) {
		return new JBlasMatrixAdapter(JBlasDoubleMatrix.zeros(rows,cols));
	}

	@Override
	public MatrixAdapter createRand(int r, int c) {
		return new JBlasMatrixAdapter(JBlasDoubleMatrix.rand(r,c));
	}

	
	private Stopwatch createStartedTimer()
	{
		Stopwatch timer = new Stopwatch();
		timer.start();
		return timer;
	}

}