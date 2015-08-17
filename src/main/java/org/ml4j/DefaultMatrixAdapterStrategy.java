package org.ml4j;

import org.ml4j.jblas.JBlasDoubleMatrix;
import org.ml4j.jblas.JBlasMatrixAdapter;

public class DefaultMatrixAdapterStrategy implements MatrixAdapterStrategy {

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
	public MatrixAdapter concatHorizontally(MatrixAdapter matrix, MatrixAdapter matrix2) {
		return new JBlasMatrixAdapter(JBlasDoubleMatrix.concatHorizontally(JBlasMatrixAdapter.createJBlasDoubleMatrix(matrix),JBlasMatrixAdapter.createJBlasDoubleMatrix(matrix2)));
	}

	@Override
	public MatrixAdapter concatVertically(MatrixAdapter matrix, MatrixAdapter matrix2) {
		return new JBlasMatrixAdapter(JBlasDoubleMatrix.concatVertically(JBlasMatrixAdapter.createJBlasDoubleMatrix(matrix),JBlasMatrixAdapter.createJBlasDoubleMatrix(matrix2)));
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


	@Override
	public MatrixAdapter pow(MatrixAdapter matrix, int i) {
		return matrix.pow(i);
	}

	@Override
	public MatrixAdapter log(MatrixAdapter matrix) {
		return matrix.logi();
	}

	@Override
	public void expi(MatrixAdapter matrix) {
		matrix.expi();
	}

	@Override
	public void powi(MatrixAdapter matrix, int d) {
		matrix.powi(d);

	}

	@Override
	public void logi(MatrixAdapter matrix) {
		 matrix.logi();
	}

	@Override
	public MatrixAdapter mul(MatrixAdapter matrix, MatrixAdapter matrix2) {
		return matrix.mul(matrix2);
	}

	@Override
	public MatrixAdapter mmul(MatrixAdapter matrix, MatrixAdapter matrix2) {
		return matrix.mmul(matrix2);
	}

	@Override
	public void putRow(MatrixAdapter matrix, int i, MatrixAdapter matrix2) {
		matrix.putRow(i, matrix2);
	}

	@Override
	public void diviColumnVector(MatrixAdapter matrix, MatrixAdapter matrix2) {
		matrix.diviColumnVector(matrix2);
	}

	@Override
	public void put(MatrixAdapter matrix, int[] indicies, int inputInd, DoubleMatrix x) {
		matrix.put(indicies,inputInd,x.matrix);
		
	}

	@Override
	public void reshape(MatrixAdapter matrix, int r, int c) {
		matrix.reshape(r, c);
	}

	@Override
	public void addi(MatrixAdapter matrix, double v) {
		matrix.addi(v);
	}

	@Override
	public MatrixAdapter add(MatrixAdapter matrix, double v) {
		return matrix.add(v);
	}

	@Override
	public void addi(MatrixAdapter matrix, MatrixAdapter m) {
		matrix.addi(m);
	}

	@Override
	public void subi(MatrixAdapter matrix, MatrixAdapter matrix2) {
		matrix.subi(matrix2);

	}

	@Override
	public void divi(MatrixAdapter matrix, double v) {
		matrix.divi(v);

	}

	@Override
	public MatrixAdapter mul(MatrixAdapter matrix, double v) {
		return matrix.mul(v);
	}

	@Override
	public void muli(MatrixAdapter matrix, double v) {
		matrix.muli(v);
	}

	@Override
	public MatrixAdapter sub(MatrixAdapter matrix, MatrixAdapter m) {
		return matrix.sub(m);
	}

	@Override
	public MatrixAdapter transpose(MatrixAdapter matrix) {
		return matrix.transpose();
	}

	@Override
	public MatrixAdapter getRowRange(MatrixAdapter matrix, int offset, int i, int j) {
		return matrix.getRowRange(offset, i, j);
	}

	@Override
	public double dot(MatrixAdapter matrix, MatrixAdapter matrix2) {
		return matrix.dot(matrix2);
	}

	@Override
	public void put(MatrixAdapter matrix, int i, double v) {
		matrix.put(i,v);
	}

	@Override
	public MatrixAdapter rowSums(MatrixAdapter matrix) {
		return matrix.rowSums();
	}

	@Override
	public MatrixAdapter add(MatrixAdapter matrix, MatrixAdapter matrix2) {
		return matrix.add(matrix2);
	}

	@Override
	public MatrixAdapter div(MatrixAdapter matrix, double v) {
		return matrix.div(v);
	}

	@Override
	public void putColumn(MatrixAdapter matrix, int i, MatrixAdapter m) {
		matrix.putColumn(i, m);
	}

	@Override
	public void muli(MatrixAdapter matrix, MatrixAdapter matrix2) {
		matrix.muli(matrix2);
	}

	@Override
	public void put(MatrixAdapter matrix, int row, int col, double d) {
		matrix.put(row,col,d);
	}

	@Override
	public MatrixAdapter getRow(MatrixAdapter matrix, int row) {
		return matrix.getRow(row);
	}

	@Override
	public int[] findIndices(MatrixAdapter matrix) {
		return matrix.findIndices();
	}

	@Override
	public double sum(MatrixAdapter matrix) {
		return matrix.sum();
	}

	@Override
	public int[] rowArgmaxs(MatrixAdapter matrix) {
		return matrix.rowArgmaxs();
	}

	@Override
	public MatrixAdapter get(MatrixAdapter matrix,int[] rows, int[] cols) {
		return matrix.get(rows,cols);
	}

	@Override
	public MatrixAdapter getColumns(MatrixAdapter matrix, int[] columns) {
		return matrix.getColumns(columns);
	}

	@Override
	public MatrixAdapter getRows(MatrixAdapter matrix, int[] rows) {		
		return matrix.getRows(rows);
	}

	@Override
	public MatrixAdapter getColumn(MatrixAdapter matrix, int j) {
		return matrix.getColumn(j);
	}

	@Override
	public int argmax(MatrixAdapter matrix) {
		return matrix.argmax();
	}

	

}
