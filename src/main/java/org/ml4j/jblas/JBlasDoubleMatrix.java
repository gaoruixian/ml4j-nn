package org.ml4j.jblas;

import java.io.Serializable;

import org.ml4j.MatrixOperations;

public class JBlasDoubleMatrix implements Serializable,MatrixOperations<JBlasDoubleMatrix> {

	public org.jblas.DoubleMatrix matrix;
	
	public JBlasDoubleMatrix(org.jblas.DoubleMatrix matrix)
	{
		this.matrix = matrix;
	}
	
	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;

	public JBlasDoubleMatrix(int rows, int cols) {
		this.matrix = new org.jblas.DoubleMatrix(rows,cols);
	}
	
	
	public JBlasDoubleMatrix(int rows, int cols,double[] data) {
		this.matrix = new org.jblas.DoubleMatrix(rows,cols,data);
	}
	
	

	public JBlasDoubleMatrix() {
		this.matrix = new org.jblas.DoubleMatrix();

	}

	public JBlasDoubleMatrix(double[][] inputs) {
		this.matrix = new org.jblas.DoubleMatrix(inputs);

	}

	public JBlasDoubleMatrix(double[] inputToReconstruct) {
		this.matrix = new org.jblas.DoubleMatrix(inputToReconstruct);

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

	public static JBlasDoubleMatrix ones(int rows) {
		return new JBlasDoubleMatrix(org.jblas.DoubleMatrix.ones(rows));
	}

	public static JBlasDoubleMatrix concatHorizontally(JBlasDoubleMatrix ones,
			JBlasDoubleMatrix thetasMask) {
		return new JBlasDoubleMatrix(org.jblas.DoubleMatrix.concatHorizontally(ones.matrix, thetasMask.matrix));
	}
	
	public static JBlasDoubleMatrix concatVertically(JBlasDoubleMatrix ones,
			JBlasDoubleMatrix thetasMask) {
		return new JBlasDoubleMatrix(org.jblas.DoubleMatrix.concatVertically(ones.matrix, thetasMask.matrix));
	}

	public static JBlasDoubleMatrix ones(int rows, int cols) {
		return new JBlasDoubleMatrix(org.jblas.DoubleMatrix.ones(rows,cols));
	}

	public JBlasDoubleMatrix mul(double scalingFactor) {
		return new JBlasDoubleMatrix(matrix.mul(scalingFactor));
	}
	
	
	public JBlasDoubleMatrix muli(double scalingFactor) {
		matrix.muli(scalingFactor);
		return this;
	}

	public void put(int outputInd, int inputInd, int i) {
		matrix.put(outputInd, inputInd,i);
	}

	public JBlasDoubleMatrix sub(JBlasDoubleMatrix desiredOutputs) {
		return new JBlasDoubleMatrix(matrix.sub(desiredOutputs.matrix));
	}

	public JBlasDoubleMatrix transpose() {
		//return new JBlasDoubleMatrix(createIndArray(matrix).transpose());
		
		return new JBlasDoubleMatrix(matrix.transpose());
	}

	public JBlasDoubleMatrix copy(JBlasDoubleMatrix reshapeToVector) {
		return new JBlasDoubleMatrix(this.matrix.copy(reshapeToVector.matrix));
	}

	public int getColumns() {
		return matrix.getColumns();
	}

	public JBlasDoubleMatrix dup() {
		return new JBlasDoubleMatrix(matrix.dup());
	}

	public static JBlasDoubleMatrix randn(int outputNeuronCount, int i) {
		return new JBlasDoubleMatrix(org.jblas.DoubleMatrix.randn(outputNeuronCount,i));
	}

	public JBlasDoubleMatrix getRow(int row) {
		return new JBlasDoubleMatrix(matrix.getRow(row));
	}

	public int[] findIndices() {
		return matrix.findIndices();
	}

	public double get(int i, int j) {
		return matrix.get(i,j);
	}

	public void put(int row, int inputInd, double d) {
		matrix.put(row, inputInd,d);
	}

	public JBlasDoubleMatrix muli(JBlasDoubleMatrix thetasMask) {
		
		matrix.muli(thetasMask.matrix);
		return this;
	}

	public JBlasDoubleMatrix mmul(JBlasDoubleMatrix mul) {
			
		//return new JBlasDoubleMatrix(createIndArray(matrix).mmul(createIndArray(mul.matrix)));
		return new JBlasDoubleMatrix(matrix.mmul(mul.matrix));
	}
	
	
	public JBlasDoubleMatrix mmul(JBlasDoubleMatrix mul,JBlasDoubleMatrix o) {
		
		//return new JBlasDoubleMatrix(createIndArray(matrix).mmul(createIndArray(mul.matrix)));
		return new JBlasDoubleMatrix(matrix.mmul(mul.matrix));
	}

	public JBlasDoubleMatrix mul(JBlasDoubleMatrix dropoutMask) {
		//return new JBlasDoubleMatrix(createIndArray(matrix).mul(createIndArray(dropoutMask.matrix)));

		return new JBlasDoubleMatrix(matrix.mul(dropoutMask.matrix));
	}

	public double sum() {
		return matrix.sum();
	}

	public static JBlasDoubleMatrix zeros(int rows, int cols) {
		return new JBlasDoubleMatrix(org.jblas.DoubleMatrix.zeros(rows,cols));
	}

	public int[] rowArgmaxs() {
		return matrix.rowArgmaxs();
	}

	public JBlasDoubleMatrix get(int[] rows, int[] cols) {
		return new JBlasDoubleMatrix(matrix.get(rows,cols));
	}

	public void putColumn(int i, JBlasDoubleMatrix zeros) {
		matrix.putColumn(i, zeros.matrix);
	}

	public JBlasDoubleMatrix div(double m) {
		return new JBlasDoubleMatrix(matrix.div(m));
	}

	public JBlasDoubleMatrix add(JBlasDoubleMatrix mul) {
		return new JBlasDoubleMatrix(matrix.add(mul.matrix));
	}

	public JBlasDoubleMatrix subi(JBlasDoubleMatrix mul) {
		matrix.subi(mul.matrix);
		return this;
	}

	public JBlasDoubleMatrix divi(double i) {
		matrix.divi(i);
		return this;
	}

	public JBlasDoubleMatrix getColumns(
			int[] hiddenOutputGradientColumnsForRecurrentOutputUnits) {
		return new JBlasDoubleMatrix(matrix.getColumns(hiddenOutputGradientColumnsForRecurrentOutputUnits));
	}

	public JBlasDoubleMatrix getRows(
			int[] inputHiddenGradientRowsForRecurrentHiddenUnits) {
		return new JBlasDoubleMatrix(matrix.getRows(inputHiddenGradientRowsForRecurrentHiddenUnits));
	}

	public static JBlasDoubleMatrix rand(int i, int length) {
		return new JBlasDoubleMatrix(org.jblas.DoubleMatrix.rand(i,length));
	}

	public JBlasDoubleMatrix addi(JBlasDoubleMatrix pairwiseVectorProduct) {
		// TODO Auto-generated method stub
		return new JBlasDoubleMatrix(matrix.addi(pairwiseVectorProduct.matrix));

	}

	public JBlasDoubleMatrix getColumn(int j) {
		return new JBlasDoubleMatrix(matrix.getColumn(j));
	}

	public double get(int i) {
		return matrix.get(i);
	}

	public int argmax() {
		return matrix.argmax();
	}

	public JBlasDoubleMatrix add(double i) {
		return new JBlasDoubleMatrix(matrix.add(i));
	}
	
	public JBlasDoubleMatrix addi(double i) {
		matrix.addi(i);
		return this;
	}

	public JBlasDoubleMatrix rowSums() {
		return new JBlasDoubleMatrix(matrix.rowSums());
	}

	public int getLength() {
		return matrix.getLength();
	}

	public void put(int i, double log) {
		matrix.put(i, log);
	}

	public double dot(JBlasDoubleMatrix s) {
		return matrix.dot(s.matrix);
	}

	public JBlasDoubleMatrix getRowRange(int offset, int i, int j) {
		return new JBlasDoubleMatrix(matrix.getRowRange(offset,i,j));
	}

	public void reshape(int length, int i) {
		matrix.reshape(length, i);
	}

	
	
	public void put(int[] indicies, int inputInd, JBlasDoubleMatrix x) {
		matrix.put(indicies, inputInd,x.matrix);
	}

	public JBlasDoubleMatrix diviColumnVector(JBlasDoubleMatrix sums) {
		matrix.diviColumnVector(sums.matrix);
		return this;
	}


	public JBlasDoubleMatrix transpose(JBlasDoubleMatrix thetasTransposeTarget2) {
		 return transpose();
	}


	public JBlasDoubleMatrix mul(JBlasDoubleMatrix dropoutMask,
			JBlasDoubleMatrix inputsMulDropoutsTarget) {
		return mul(dropoutMask);
	}



	public void putRow(int i, JBlasDoubleMatrix zeros) {
		matrix.putRow(i,zeros.matrix);
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
		JBlasDoubleMatrix other = (JBlasDoubleMatrix) obj;
		if (matrix == null) {
			if (other.matrix != null)
				return false;
		} else if (!matrix.equals(other.matrix))
			return false;
		return true;
	}


	
}
