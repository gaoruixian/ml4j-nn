package org.ml4j;

public class CudaForMMulStrategy extends DefaultMatrixAdapterStrategy {

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;

	public CudaForMMulStrategy() {
		super(new CustomCudaConcatenationMatrixAdapterFactory());
	}

	
	
	@Override
	public void divi(MatrixAdapter matrix, double v) {
		super.divi(matrix.asJBlasMatrix(), v);
	}



	@Override
	public MatrixAdapter div(MatrixAdapter matrix, double v) {
		return super.div(matrix.asJBlasMatrix(), v);
	}



	@Override
	public MatrixAdapter mul(MatrixAdapter matrix, MatrixAdapter matrix2) {
		return super.mul(matrix.asJBlasMatrix(), matrix2.asJBlasMatrix());
	}

	@Override
	public MatrixAdapter mmul(MatrixAdapter matrix, MatrixAdapter matrix2) {
		return super.mmul(matrix.asCudaMatrix(), matrix2.asCudaMatrix());
	}
	
	
	@Override
	public MatrixAdapter mul(MatrixAdapter matrix, double v) {
		return super.mul(matrix.asJBlasMatrix(), v);
	}

	@Override
	public void muli(MatrixAdapter matrix, double v) {
		super.muli(matrix, v);
	}

	@Override
	public void muli(MatrixAdapter matrix, MatrixAdapter matrix2) {
		super.muli(matrix.asJBlasMatrix(), matrix2.asJBlasMatrix());
	}
	
	

	
	
}
