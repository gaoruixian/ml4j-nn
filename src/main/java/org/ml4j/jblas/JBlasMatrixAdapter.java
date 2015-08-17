package org.ml4j.jblas;

import org.ml4j.MatrixAdapter;


public class JBlasMatrixAdapter implements MatrixAdapter {

	public JBlasDoubleMatrix matrix;
	
	public JBlasMatrixAdapter(JBlasDoubleMatrix matrix)
	{
		this.matrix = matrix;
	}
	
	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;

	public JBlasMatrixAdapter(int rows, int cols) {
		this.matrix = new JBlasDoubleMatrix(rows,cols);
	}
	
	
	public JBlasMatrixAdapter(int rows, int cols,double[] data) {
		this.matrix = new JBlasDoubleMatrix(rows,cols,data);
	}
	
	

	public JBlasMatrixAdapter() {
		this.matrix = new JBlasDoubleMatrix();

	}

	public JBlasMatrixAdapter(double[][] inputs) {
		this.matrix = new JBlasDoubleMatrix(inputs);

	}

	public JBlasMatrixAdapter(double[] inputToReconstruct) {
		this.matrix = new JBlasDoubleMatrix(inputToReconstruct);

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

	public static JBlasDoubleMatrix createJBlasDoubleMatrix(MatrixAdapter baseDoubleMatrix)
	{
	
		if (baseDoubleMatrix instanceof JBlasMatrixAdapter)
		{
			return ((JBlasMatrixAdapter)baseDoubleMatrix).matrix;
		}
		return new JBlasDoubleMatrix(baseDoubleMatrix.getRows(),baseDoubleMatrix.getColumns(),baseDoubleMatrix.toArray());
	
	}
	
	public static JBlasMatrixAdapter createJBlasBaseDoubleMatrix(MatrixAdapter baseDoubleMatrix)
	{
	
		if (baseDoubleMatrix instanceof JBlasMatrixAdapter)
		{
			return ((JBlasMatrixAdapter)baseDoubleMatrix);
		}
		return new JBlasMatrixAdapter(baseDoubleMatrix.getRows(),baseDoubleMatrix.getColumns(),baseDoubleMatrix.toArray());
	
	}

	public JBlasMatrixAdapter mul(double scalingFactor) {
		return new JBlasMatrixAdapter(matrix.mul(scalingFactor));
	}
	
	
	public JBlasMatrixAdapter muli(double scalingFactor) {
		matrix.muli(scalingFactor);
		return this;
	}

	public void put(int outputInd, int inputInd, int i) {
		matrix.put(outputInd, inputInd,i);
	}

	public MatrixAdapter sub(MatrixAdapter desiredOutputs) {
		return new JBlasMatrixAdapter(matrix.sub(createJBlasDoubleMatrix(desiredOutputs)));
	}

	public MatrixAdapter transpose() {
		//return new JBlasDoubleMatrix(createIndArray(matrix).transpose());
		
		return new JBlasMatrixAdapter(matrix.transpose());
	}

	public MatrixAdapter copy(MatrixAdapter reshapeToVector) {
		return new JBlasMatrixAdapter(this.matrix.copy(createJBlasDoubleMatrix(reshapeToVector)));
	}

	public int getColumns() {
		return matrix.getColumns();
	}

	public MatrixAdapter dup() {
		return new JBlasMatrixAdapter(matrix.dup());
	}

	

	public MatrixAdapter getRow(int row) {
		return new JBlasMatrixAdapter(matrix.getRow(row));
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

	public MatrixAdapter muli(MatrixAdapter thetasMask) {
		
		matrix.muli(createJBlasDoubleMatrix(thetasMask));
		return this;
	}

	public MatrixAdapter mmul(MatrixAdapter mul) {
			
		//return new JBlasDoubleMatrix(createIndArray(matrix).mmul(createIndArray(mul.matrix)));
		return new JBlasMatrixAdapter(matrix.mmul(createJBlasDoubleMatrix(mul)));
	}
	
	
	public MatrixAdapter mul(MatrixAdapter dropoutMask) {
		//return new JBlasDoubleMatrix(createIndArray(matrix).mul(createIndArray(dropoutMask.matrix)));

		return new JBlasMatrixAdapter(matrix.mul(createJBlasDoubleMatrix(dropoutMask)));
	}

	public double sum() {
		return matrix.sum();
	}

	public int[] rowArgmaxs() {
		return matrix.rowArgmaxs();
	}

	public MatrixAdapter get(int[] rows, int[] cols) {
		return new JBlasMatrixAdapter(matrix.get(rows,cols));
	}

	public void putColumn(int i, MatrixAdapter zeros) {
		matrix.putColumn(i, createJBlasDoubleMatrix(zeros));
	}

	public MatrixAdapter div(double m) {
		return new JBlasMatrixAdapter(matrix.div(m));
	}

	public MatrixAdapter add(MatrixAdapter mul) {
		return new JBlasMatrixAdapter(matrix.add(createJBlasDoubleMatrix(mul)));
	}

	public MatrixAdapter subi(MatrixAdapter mul) {
		matrix.subi(createJBlasDoubleMatrix(mul));
		return this;
	}

	public MatrixAdapter divi(double i) {
		matrix.divi(i);
		return this;
	}

	public MatrixAdapter getColumns(
			int[] hiddenOutputGradientColumnsForRecurrentOutputUnits) {
		return new JBlasMatrixAdapter(matrix.getColumns(hiddenOutputGradientColumnsForRecurrentOutputUnits));
	}

	public MatrixAdapter getRows(
			int[] inputHiddenGradientRowsForRecurrentHiddenUnits) {
		return new JBlasMatrixAdapter(matrix.getRows(inputHiddenGradientRowsForRecurrentHiddenUnits));
	}

	
	public MatrixAdapter addi(MatrixAdapter pairwiseVectorProduct) {
		// TODO Auto-generated method stub
		return new JBlasMatrixAdapter(matrix.addi(createJBlasDoubleMatrix(pairwiseVectorProduct)));

	}

	public MatrixAdapter getColumn(int j) {
		return new JBlasMatrixAdapter(matrix.getColumn(j));
	}

	public double get(int i) {
		return matrix.get(i);
	}

	public int argmax() {
		return matrix.argmax();
	}

	public MatrixAdapter add(double i) {
		return new JBlasMatrixAdapter(matrix.add(i));
	}
	
	public MatrixAdapter addi(double i) {
		matrix.addi(i);
		return this;
	}

	public MatrixAdapter rowSums() {
		return new JBlasMatrixAdapter(matrix.rowSums());
	}

	public int getLength() {
		return matrix.getLength();
	}

	public void put(int i, double log) {
		matrix.put(i, log);
	}

	public double dot(MatrixAdapter s) {
		return matrix.dot(createJBlasDoubleMatrix(s));
	}

	public MatrixAdapter getRowRange(int offset, int i, int j) {
		return new JBlasMatrixAdapter(matrix.getRowRange(offset,i,j));
	}

	public void reshape(int length, int i) {
		matrix.reshape(length, i);
	}

	
	
	public void put(int[] indicies, int inputInd, MatrixAdapter x) {
		matrix.put(indicies, inputInd,createJBlasDoubleMatrix(x));
	}

	public MatrixAdapter diviColumnVector(MatrixAdapter sums) {
		matrix.diviColumnVector(createJBlasDoubleMatrix(sums));
		return this;
	}


	

	public void putRow(int i, MatrixAdapter zeros) {
		matrix.putRow(i,createJBlasDoubleMatrix(zeros));
	}


	@Override
	public MatrixAdapter pow(int i) {
		return new JBlasMatrixAdapter(JBlasMatrixFunctions.pow(this.matrix, i));
	}


	@Override
	public MatrixAdapter log() {
		return new JBlasMatrixAdapter(JBlasMatrixFunctions.log(this.matrix));
	}


	@Override
	public MatrixAdapter expi() {
		return new JBlasMatrixAdapter(JBlasMatrixFunctions.expi(this.matrix));
	}


	@Override
	public MatrixAdapter powi(int d) {
		return new JBlasMatrixAdapter(JBlasMatrixFunctions.powi(this.matrix,d));
	}


	@Override
	public MatrixAdapter logi() {
		return new JBlasMatrixAdapter(JBlasMatrixFunctions.logi(this.matrix));
	}

	public MatrixAdapter asJBlasMatrix()
	{
		return this;
	}
	
	public MatrixAdapter asCudaMatrix()
	{
		throw new UnsupportedOperationException("Cuda matrices not yet supported");
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
		JBlasMatrixAdapter other = (JBlasMatrixAdapter) obj;
		if (matrix == null) {
			if (other.matrix != null)
				return false;
		} else if (!matrix.equals(other.matrix))
			return false;
		return true;
	}

	
}
