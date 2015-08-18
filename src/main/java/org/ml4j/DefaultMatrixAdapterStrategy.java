package org.ml4j;

import java.io.Serializable;

import org.ml4j.jblas.JBlasMatrixAdapterFactory;


public class DefaultMatrixAdapterStrategy implements MatrixAdapterStrategy,Serializable{

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;
	private MatrixAdapterFactory matrixAdapterFactory;
	
	public DefaultMatrixAdapterStrategy(MatrixAdapterFactory matrixAdapterFactory)
	{
		this.matrixAdapterFactory = matrixAdapterFactory;
	}
	
	public DefaultMatrixAdapterStrategy()
	{
		this.matrixAdapterFactory = new JBlasMatrixAdapterFactory();
	}
	
	@Override
	public MatrixAdapter pow(MatrixAdapter matrix, int i) {
		return matrix.pow(i);
	}

	@Override
	public MatrixAdapter log(MatrixAdapter matrix) {
		return matrix.log();
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

	@Override
	public MatrixAdapterFactory getMatrixAdapterFactory() {
		return matrixAdapterFactory;
	}

	@Override
	public MatrixAdapter sigmoid(MatrixAdapter matrix) {
		return matrix.sigmoid();
	}


}
