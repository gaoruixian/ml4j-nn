package org.ml4j.jblas;

import org.ml4j.DoubleMatrix;
import org.ml4j.MatrixAdapter;
import org.ml4j.cuda.CudaMatrixAdapter;

import com.google.common.base.Stopwatch;


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
	
	private Stopwatch createStartedTimer()
	{
		Stopwatch timer = new Stopwatch();
		timer.start();
		return timer;
	}

	public JBlasMatrixAdapter mul(double scalingFactor) {
		
		Stopwatch timer = createStartedTimer();
		JBlasMatrixAdapter adapter= new JBlasMatrixAdapter(matrix.mul(scalingFactor));
		DoubleMatrix.addTiming("mulDoubleJblas",timer.elapsedMillis());
		return adapter;
	}
	
	
	public JBlasMatrixAdapter muli(double scalingFactor) {
		Stopwatch timer = createStartedTimer();

		matrix.muli(scalingFactor);
		DoubleMatrix.addTiming("muliDoubleJblas",timer.elapsedMillis());

		return this;
	}

	public MatrixAdapter sub(MatrixAdapter desiredOutputs) {
		Stopwatch timer = createStartedTimer();

		MatrixAdapter ret = new JBlasMatrixAdapter(matrix.sub(createJBlasDoubleMatrix(desiredOutputs)));
		DoubleMatrix.addTiming("subJblas",timer.elapsedMillis());

		return ret;
	}

	public MatrixAdapter transpose() {
		//return new JBlasDoubleMatrix(createIndArray(matrix).transpose());
		Stopwatch timer = createStartedTimer();

		MatrixAdapter ret =  new JBlasMatrixAdapter(matrix.transpose());
		DoubleMatrix.addTiming("transposeJblas",timer.elapsedMillis());

		return ret;
	}

	public MatrixAdapter copy(MatrixAdapter reshapeToVector) {
		Stopwatch timer = createStartedTimer();

		MatrixAdapter ret =  new JBlasMatrixAdapter(this.matrix.copy(createJBlasDoubleMatrix(reshapeToVector)));
		DoubleMatrix.addTiming("copyJblas",timer.elapsedMillis());

		return ret;
	}

	public int getColumns() {
		return matrix.getColumns();
	}

	public MatrixAdapter dup() {
		Stopwatch timer = createStartedTimer();

		MatrixAdapter ret = new JBlasMatrixAdapter(matrix.dup());
		DoubleMatrix.addTiming("dupJBlas",timer.elapsedMillis());

		return ret;
	}

	

	public MatrixAdapter getRow(int row) {
		Stopwatch timer = createStartedTimer();

		MatrixAdapter adapter  =  new JBlasMatrixAdapter(matrix.getRow(row));
		DoubleMatrix.addTiming("getRowJBlas",timer.elapsedMillis());

		return adapter;
	}

	public int[] findIndices() {
		Stopwatch timer = createStartedTimer();
		int[] ret =  matrix.findIndices();
		DoubleMatrix.addTiming("findIndicesJBlas",timer.elapsedMillis());

		return ret;
	}

	public double get(int i, int j) {
		return matrix.get(i,j);
	}

	public void put(int row, int inputInd, double d) {

		matrix.put(row, inputInd,d);

	}

	public MatrixAdapter muli(MatrixAdapter thetasMask) {
		
		Stopwatch timer = createStartedTimer();

		matrix.muli(createJBlasDoubleMatrix(thetasMask));
		DoubleMatrix.addTiming("muliJBlas",timer.elapsedMillis());

		return this;
	}

	public MatrixAdapter mmul(MatrixAdapter mul) {
		
		Stopwatch timer = createStartedTimer();

		MatrixAdapter ret =  new JBlasMatrixAdapter(matrix.mmul(createJBlasDoubleMatrix(mul)));
		DoubleMatrix.addTiming("mmulJBlas",timer.elapsedMillis());

		return ret;
	}
	
	
	public MatrixAdapter mul(MatrixAdapter dropoutMask) {
		//return new JBlasDoubleMatrix(createIndArray(matrix).mul(createIndArray(dropoutMask.matrix)));
		Stopwatch timer = createStartedTimer();

		MatrixAdapter ret =  new JBlasMatrixAdapter(matrix.mul(createJBlasDoubleMatrix(dropoutMask)));
		DoubleMatrix.addTiming("mulJBlas",timer.elapsedMillis());

		return ret;
	}

	public double sum() {
		
		Stopwatch timer = createStartedTimer();
		double ret =  matrix.sum();
		DoubleMatrix.addTiming("sumJBlas",timer.elapsedMillis());

		return ret;
	}

	public int[] rowArgmaxs() {
		return matrix.rowArgmaxs();
	}

	public MatrixAdapter get(int[] rows, int[] cols) {
		Stopwatch timer = createStartedTimer();
		MatrixAdapter ret =  new JBlasMatrixAdapter(matrix.get(rows,cols));
		DoubleMatrix.addTiming("getJBlas",timer.elapsedMillis());

		return ret;
	}

	public void putColumn(int i, MatrixAdapter zeros) {
		Stopwatch timer = createStartedTimer();
		matrix.putColumn(i, createJBlasDoubleMatrix(zeros));
		DoubleMatrix.addTiming("putColumnJBlas",timer.elapsedMillis());

	}

	public MatrixAdapter div(double m) {
		Stopwatch timer = createStartedTimer();

		MatrixAdapter ret =  new JBlasMatrixAdapter(matrix.div(m));
		DoubleMatrix.addTiming("divJBlas",timer.elapsedMillis());

		return ret;
	}

	public MatrixAdapter add(MatrixAdapter mul) {
		Stopwatch timer = createStartedTimer();

		MatrixAdapter ret =  new JBlasMatrixAdapter(matrix.add(createJBlasDoubleMatrix(mul)));
		DoubleMatrix.addTiming("addJblas",timer.elapsedMillis());

		return ret;
		}

	public MatrixAdapter subi(MatrixAdapter mul) {
		Stopwatch timer = createStartedTimer();

		matrix.subi(createJBlasDoubleMatrix(mul));
		DoubleMatrix.addTiming("subJBlas",timer.elapsedMillis());

		return this;
	}

	public MatrixAdapter divi(double i) {
		Stopwatch timer = createStartedTimer();

		matrix.divi(i);
		DoubleMatrix.addTiming("diviDoubleJBlas",timer.elapsedMillis());

		return this;
	}

	public MatrixAdapter getColumns(int[] hiddenOutputGradientColumnsForRecurrentOutputUnits) {
		Stopwatch timer = createStartedTimer();
		MatrixAdapter ret =  new JBlasMatrixAdapter(matrix.getColumns(hiddenOutputGradientColumnsForRecurrentOutputUnits));
		DoubleMatrix.addTiming("getColumnsJBlas",timer.elapsedMillis());

		return ret;
	}

	public MatrixAdapter getRows(
			int[] inputHiddenGradientRowsForRecurrentHiddenUnits) {
		Stopwatch timer = createStartedTimer();

		MatrixAdapter ret= new JBlasMatrixAdapter(matrix.getRows(inputHiddenGradientRowsForRecurrentHiddenUnits));
		DoubleMatrix.addTiming("getRowsJBlas",timer.elapsedMillis());

		return ret;
	}

	
	public MatrixAdapter addi(MatrixAdapter pairwiseVectorProduct) {
		// TODO Auto-generated method stub
		Stopwatch timer = createStartedTimer();

		matrix.addi(createJBlasDoubleMatrix(pairwiseVectorProduct));
		DoubleMatrix.addTiming("addiJBlas",timer.elapsedMillis());

		return this;

	}

	public MatrixAdapter getColumn(int j) {
		Stopwatch timer = createStartedTimer();
		MatrixAdapter ret = new JBlasMatrixAdapter(matrix.getColumn(j));
		DoubleMatrix.addTiming("getColumnJBlas",timer.elapsedMillis());

		return ret;
	}

	public double get(int i) {
		return matrix.get(i);
	}

	public int argmax() {
		Stopwatch timer = createStartedTimer();

		int argMax = matrix.argmax();
		DoubleMatrix.addTiming("argmaxJBlas",timer.elapsedMillis());

		return argMax;
	}

	public MatrixAdapter add(double i) {
		Stopwatch timer = createStartedTimer();

		MatrixAdapter ret =  new JBlasMatrixAdapter(matrix.add(i));
		DoubleMatrix.addTiming("addJBlas",timer.elapsedMillis());

		return ret;
	}
	
	public MatrixAdapter addi(double i) {
		Stopwatch timer = createStartedTimer();

		matrix.addi(i);
		DoubleMatrix.addTiming("addiJBlas",timer.elapsedMillis());

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
		Stopwatch timer = createStartedTimer();

		double ret =  matrix.dot(createJBlasDoubleMatrix(s));
		DoubleMatrix.addTiming("dotJBlas",timer.elapsedMillis());

		return ret;
	}

	public MatrixAdapter getRowRange(int offset, int i, int j) {
		Stopwatch timer = createStartedTimer();

		MatrixAdapter ret =  new JBlasMatrixAdapter(matrix.getRowRange(offset,i,j));
		DoubleMatrix.addTiming("getRowRangeJBlas",timer.elapsedMillis());

		return ret;
	}

	public void reshape(int length, int i) {
		Stopwatch timer = createStartedTimer();

		matrix.reshape(length, i);
		DoubleMatrix.addTiming("reshapeJBlas",timer.elapsedMillis());

	}

	
	
	public void put(int[] indicies, int inputInd, MatrixAdapter x) {
		Stopwatch timer = createStartedTimer();

		matrix.put(indicies, inputInd,createJBlasDoubleMatrix(x));
		DoubleMatrix.addTiming("putlJBlas",timer.elapsedMillis());

	}

	public MatrixAdapter diviColumnVector(MatrixAdapter sums) {
		Stopwatch timer = createStartedTimer();

		matrix.diviColumnVector(createJBlasDoubleMatrix(sums));
		DoubleMatrix.addTiming("diviColumnVectorJBlas",timer.elapsedMillis());

		return this;
	}


	

	public void putRow(int i, MatrixAdapter zeros) {
		Stopwatch timer = createStartedTimer();

		matrix.putRow(i,createJBlasDoubleMatrix(zeros));
		DoubleMatrix.addTiming("putRowJBlas",timer.elapsedMillis());

	}


	@Override
	public MatrixAdapter pow(int i) {
		Stopwatch timer = createStartedTimer();

		MatrixAdapter ret =  new JBlasMatrixAdapter(JBlasMatrixFunctions.pow(this.matrix, i));
		DoubleMatrix.addTiming("powJBlas",timer.elapsedMillis());

		return ret;
	}


	@Override
	public MatrixAdapter log() {
		Stopwatch timer = createStartedTimer();

		MatrixAdapter ret =  new JBlasMatrixAdapter(JBlasMatrixFunctions.log(this.matrix));
		DoubleMatrix.addTiming("logJBlas",timer.elapsedMillis());

		return ret;
	}


	@Override
	public MatrixAdapter expi() {
		Stopwatch timer = createStartedTimer();

		JBlasMatrixFunctions.expi(this.matrix);
		DoubleMatrix.addTiming("expiJBlas",timer.elapsedMillis());

		return this;
	}


	@Override
	public MatrixAdapter powi(int d) {
		Stopwatch timer = createStartedTimer();

		JBlasMatrixFunctions.powi(this.matrix,d);
		DoubleMatrix.addTiming("powiJBlas",timer.elapsedMillis());

		return this;
	}


	@Override
	public MatrixAdapter logi() {
		Stopwatch timer = createStartedTimer();

		JBlasMatrixFunctions.logi(this.matrix);
		DoubleMatrix.addTiming("logiJBlas",timer.elapsedMillis());

		return this;
	}

	public MatrixAdapter asJBlasMatrix()
	{
		return this;
	}
	
	public MatrixAdapter asCudaMatrix()
	{
		Stopwatch timer = createStartedTimer();

		MatrixAdapter ret =  new CudaMatrixAdapter(matrix.getRows(),matrix.getColumns(),matrix.toArray());
		DoubleMatrix.addTiming("asCudaMatrixJBlas",timer.elapsedMillis());

		return ret;
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


	@Override
	public MatrixAdapter sigmoid() {
		return new JBlasMatrixAdapter(JBlasMatrixFunctions.sigmoid(matrix));
	}
	
	
	
}
