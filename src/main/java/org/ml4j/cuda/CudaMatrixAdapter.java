package org.ml4j.cuda;

import org.ml4j.DoubleMatrix;
import org.ml4j.MatrixAdapter;
import org.ml4j.jblas.JBlasMatrixAdapter;

import com.google.common.base.Stopwatch;


public class CudaMatrixAdapter implements MatrixAdapter {

	public CudaDoubleMatrix matrix;
	
	public CudaMatrixAdapter(CudaDoubleMatrix matrix)
	{
		this.matrix = matrix;
	}
	
	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;

	public CudaMatrixAdapter(int rows, int cols) {
		this.matrix = new CudaDoubleMatrix(rows,cols);
	}
	
	
	public CudaMatrixAdapter(int rows, int cols,double[] data) {
		this.matrix = new CudaDoubleMatrix(data,rows,cols);
	}
	
	

	public CudaMatrixAdapter() {
		this.matrix = new CudaDoubleMatrix();

	}

	public CudaMatrixAdapter(double[][] inputs) {
		this.matrix = new CudaDoubleMatrix(inputs);

	}

	public CudaMatrixAdapter(double[] inputToReconstruct) {
		this.matrix = new CudaDoubleMatrix(inputToReconstruct);

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

	public static CudaDoubleMatrix createCudaDoubleMatrix(MatrixAdapter baseDoubleMatrix)
	{
	
		if (baseDoubleMatrix instanceof CudaMatrixAdapter)
		{
			return ((CudaMatrixAdapter)baseDoubleMatrix).matrix;
		}
		return new CudaDoubleMatrix(baseDoubleMatrix.toArray(),baseDoubleMatrix.getRows(),baseDoubleMatrix.getColumns());
	
	}
	
	
	public static CudaMatrixAdapter createCudaBaseDoubleMatrix(MatrixAdapter baseDoubleMatrix)
	{
	
		if (baseDoubleMatrix instanceof CudaMatrixAdapter)
		{
			return ((CudaMatrixAdapter)baseDoubleMatrix);
		}
		return new CudaMatrixAdapter(baseDoubleMatrix.getRows(),baseDoubleMatrix.getColumns(),baseDoubleMatrix.toArray());
	
	}

	public CudaMatrixAdapter mul(double scalingFactor) {
		Stopwatch timer = createStartedTimer();

		CudaMatrixAdapter ret =  new CudaMatrixAdapter(matrix.mul(scalingFactor));
	
		DoubleMatrix.addTiming("mulCuda", timer.elapsedMillis());
		return ret;
	}
	
	
	public CudaMatrixAdapter muli(double scalingFactor) {
		Stopwatch timer = createStartedTimer();

		matrix.muli(scalingFactor);
		DoubleMatrix.addTiming("muliCuda", timer.elapsedMillis());

		return this;
	}

	public void put(int outputInd, int inputInd, int i) {
		Stopwatch timer = createStartedTimer();

		matrix.put(outputInd, inputInd,i);
		DoubleMatrix.addTiming("putCuda", timer.elapsedMillis());

	}

	public MatrixAdapter sub(MatrixAdapter desiredOutputs) {
		Stopwatch timer = createStartedTimer();

		MatrixAdapter ret =  new CudaMatrixAdapter(matrix.sub(createCudaDoubleMatrix(desiredOutputs)));
		DoubleMatrix.addTiming("subCuda", timer.elapsedMillis());

		return ret;
	}

	public MatrixAdapter transpose() {
		
		Stopwatch timer = createStartedTimer();

		MatrixAdapter ret =  new CudaMatrixAdapter(matrix.transpose());
		DoubleMatrix.addTiming("transposeCuda", timer.elapsedMillis());

		return ret;
	}

	public MatrixAdapter copy(MatrixAdapter reshapeToVector) {
		Stopwatch timer = createStartedTimer();

		MatrixAdapter ret =  new CudaMatrixAdapter(this.matrix.copy(createCudaDoubleMatrix(reshapeToVector)));
		DoubleMatrix.addTiming("copyCuda", timer.elapsedMillis());

		return ret;
	}

	public int getColumns() {
		return matrix.getColumns();
	}

	public MatrixAdapter dup() {
		Stopwatch timer = createStartedTimer();

		MatrixAdapter ret =  new CudaMatrixAdapter(matrix.dup());
		DoubleMatrix.addTiming("dupCuda", timer.elapsedMillis());

		return ret;
	}

	

	public MatrixAdapter getRow(int row) {
		Stopwatch timer = createStartedTimer();

		MatrixAdapter ret =  new CudaMatrixAdapter(matrix.getRow(row));
		DoubleMatrix.addTiming("getRowCuda", timer.elapsedMillis());

		return ret;
	}

	public int[] findIndices() {
		Stopwatch timer = createStartedTimer();

		int[] ret =  matrix.findIndices();
		DoubleMatrix.addTiming("findIndicesCuda", timer.elapsedMillis());

		return ret;
	}

	public double get(int i, int j) {
		return matrix.get(i,j);
	}

	public void put(int row, int inputInd, double d) {
		Stopwatch timer = createStartedTimer();
		matrix.put(row, inputInd,d);
		DoubleMatrix.addTiming("putCuda", timer.elapsedMillis());

	}

	public MatrixAdapter muli(MatrixAdapter thetasMask) {
		
		Stopwatch timer = createStartedTimer();
		matrix.muli(createCudaDoubleMatrix(thetasMask));
		DoubleMatrix.addTiming("muliCuda", timer.elapsedMillis());

		return this;
	}

	public MatrixAdapter mmul(MatrixAdapter mul) {
		
		Stopwatch timer = createStartedTimer();

		//return new JBlasDoubleMatrix(createIndArray(matrix).mmul(createIndArray(mul.matrix)));
		MatrixAdapter ret =  new CudaMatrixAdapter(matrix.mmul(createCudaDoubleMatrix(mul)));
		DoubleMatrix.addTiming("mmulCuda", timer.elapsedMillis());
		return ret;
	}
	
	
	public MatrixAdapter mul(MatrixAdapter dropoutMask) {
		//return new JBlasDoubleMatrix(createIndArray(matrix).mul(createIndArray(dropoutMask.matrix)));
		Stopwatch timer = createStartedTimer();

		MatrixAdapter ret =  new CudaMatrixAdapter(matrix.mul(createCudaDoubleMatrix(dropoutMask)));
		DoubleMatrix.addTiming("mulCuda", timer.elapsedMillis());

		return ret;
	}

	public double sum() {
		Stopwatch timer = createStartedTimer();

		double ret =  matrix.sum();
		DoubleMatrix.addTiming("sumCuda", timer.elapsedMillis());

		return ret;
	}

	public int[] rowArgmaxs() {
		Stopwatch timer = createStartedTimer();

		int[] ret =  matrix.rowArgmaxs();
		DoubleMatrix.addTiming("rowArgmaxsCuda", timer.elapsedMillis());

		return ret;
	}

	public MatrixAdapter get(int[] rows, int[] cols) {
		Stopwatch timer = createStartedTimer();

		MatrixAdapter ret =  new CudaMatrixAdapter(matrix.get(rows,cols));
		DoubleMatrix.addTiming("getCuda", timer.elapsedMillis());

		return ret;
	}

	public void putColumn(int i, MatrixAdapter zeros) {
		Stopwatch timer = createStartedTimer();

		matrix.putColumn(i, createCudaDoubleMatrix(zeros));
		DoubleMatrix.addTiming("putColumnCuda", timer.elapsedMillis());

	}

	public MatrixAdapter div(double m) {
		Stopwatch timer = createStartedTimer();

		MatrixAdapter ret =  new CudaMatrixAdapter(matrix.div(m));
		DoubleMatrix.addTiming("divCuda", timer.elapsedMillis());

		return ret;
	}

	public MatrixAdapter add(MatrixAdapter mul) {
		Stopwatch timer = createStartedTimer();

		MatrixAdapter ret =  new CudaMatrixAdapter(matrix.add(createCudaDoubleMatrix(mul)));
		DoubleMatrix.addTiming("divCuda", timer.elapsedMillis());

		return ret;
	}

	public MatrixAdapter subi(MatrixAdapter mul) {
		Stopwatch timer = createStartedTimer();

		matrix.subi(createCudaDoubleMatrix(mul));
		DoubleMatrix.addTiming("subiCuda", timer.elapsedMillis());

		return this;
	}

	public MatrixAdapter divi(double i) {
		Stopwatch timer = createStartedTimer();

		matrix.divi(i);
		DoubleMatrix.addTiming("diviCuda", timer.elapsedMillis());

		return this;
	}

	public MatrixAdapter getColumns(
			int[] hiddenOutputGradientColumnsForRecurrentOutputUnits) {
		Stopwatch timer = createStartedTimer();

		MatrixAdapter ret =  new CudaMatrixAdapter(matrix.getColumns(hiddenOutputGradientColumnsForRecurrentOutputUnits));
		DoubleMatrix.addTiming("getColumnsCuda", timer.elapsedMillis());

		return ret;
	}

	public MatrixAdapter getRows(
			int[] inputHiddenGradientRowsForRecurrentHiddenUnits) {
		Stopwatch timer = createStartedTimer();

		MatrixAdapter ret =  new CudaMatrixAdapter(matrix.getRows(inputHiddenGradientRowsForRecurrentHiddenUnits));
		DoubleMatrix.addTiming("getRowsCuda", timer.elapsedMillis());

		return ret;}

	
	public MatrixAdapter addi(MatrixAdapter pairwiseVectorProduct) {
		Stopwatch timer = createStartedTimer();
		matrix.addi(createCudaDoubleMatrix(pairwiseVectorProduct));
		DoubleMatrix.addTiming("addiCuda", timer.elapsedMillis());

		return this;

	}

	public MatrixAdapter getColumn(int j) {
		Stopwatch timer = createStartedTimer();

		MatrixAdapter ret =  new CudaMatrixAdapter(matrix.getColumn(j));
		DoubleMatrix.addTiming("getColumnCuda", timer.elapsedMillis());

		return ret;
	}

	public double get(int i) {
		return matrix.get(i);
	}

	public int argmax() {
		return matrix.argmax();
	}

	public MatrixAdapter add(double i) {
		Stopwatch timer = createStartedTimer();

		MatrixAdapter ret =  new CudaMatrixAdapter(matrix.add(i));
		DoubleMatrix.addTiming("addCuda", timer.elapsedMillis());

		return ret;
	}
	
	public MatrixAdapter addi(double i) {
		Stopwatch timer = createStartedTimer();

		matrix.addi(i);
		DoubleMatrix.addTiming("addiCuda", timer.elapsedMillis());

		return this;
	}

	public MatrixAdapter rowSums() {
		return new CudaMatrixAdapter(matrix.rowSums());
	}

	public int getLength() {
		return matrix.getLength();
	}

	public void put(int i, double log) {
		matrix.put(i, log);
	}

	public double dot(MatrixAdapter s) {
		Stopwatch timer = createStartedTimer();

		double ret =  matrix.dot(createCudaDoubleMatrix(s));
		DoubleMatrix.addTiming("dotCuda", timer.elapsedMillis());

		return ret;
	}

	public MatrixAdapter getRowRange(int offset, int i, int j) {
		return new CudaMatrixAdapter(matrix.getRowRange(offset,i,j));
	}

	public void reshape(int length, int i) {
		matrix.reshape(length, i);
	}

	
	
	public void put(int[] indicies, int inputInd, MatrixAdapter x) {
		matrix.put(indicies, inputInd,createCudaDoubleMatrix(x));
	}

	public MatrixAdapter diviColumnVector(MatrixAdapter sums) {
		matrix.diviColumnVector(createCudaDoubleMatrix(sums));
		return this;
	}


	

	public void putRow(int i, MatrixAdapter zeros) {
		matrix.putRow(i,createCudaDoubleMatrix(zeros));
	}


	@Override
	public MatrixAdapter pow(int i) {
		Stopwatch timer = createStartedTimer();

		MatrixAdapter ret =  new CudaMatrixAdapter(CudaMatrixFunctions.pow(this.matrix, i));
		DoubleMatrix.addTiming("powCuda", timer.elapsedMillis());

		return ret;
	}


	@Override
	public MatrixAdapter log() {
		Stopwatch timer = createStartedTimer();

		MatrixAdapter ret =  new CudaMatrixAdapter(CudaMatrixFunctions.log(this.matrix));
		DoubleMatrix.addTiming("logCuda", timer.elapsedMillis());

		return ret;
	}


	@Override
	public MatrixAdapter expi() {
		Stopwatch timer = createStartedTimer();
		CudaMatrixFunctions.expi(this.matrix);
		DoubleMatrix.addTiming("expiCuda", timer.elapsedMillis());

		return this;
	}


	@Override
	public MatrixAdapter powi(int d) {
		Stopwatch timer = createStartedTimer();

		CudaMatrixFunctions.powi(this.matrix,d);
		DoubleMatrix.addTiming("powiCuda", timer.elapsedMillis());

		return this;
	}


	@Override
	public MatrixAdapter logi() {
		Stopwatch timer = createStartedTimer();

		MatrixAdapter ret =  new CudaMatrixAdapter(CudaMatrixFunctions.logi(this.matrix));
		DoubleMatrix.addTiming("logiCuda", timer.elapsedMillis());
		return ret;
	}


	public MatrixAdapter asJBlasMatrix()
	{
		return JBlasMatrixAdapter.createJBlasBaseDoubleMatrix(this);
	}
	
	public MatrixAdapter asCudaMatrix()
	{
		return this;
	}

	private Stopwatch createStartedTimer()
	{
		Stopwatch timer = new Stopwatch();
		timer.start();
		return timer;
	}


	@Override
	public MatrixAdapter sigmoid() {
	{
		Stopwatch timer = createStartedTimer();
		MatrixAdapter ret =  new CudaMatrixAdapter(CudaMatrixFunctions.sigmoid(matrix));
		DoubleMatrix.addTiming("sigmoidCuda", timer.elapsedMillis());

		return ret;
	}

	}

}
