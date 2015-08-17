package org.ml4j;




public interface MatrixOperations<M extends MatrixOperations<M>> {

	M add(M matrices);

	double[][] toArray2();

	double[] toArray();
	

	M sub(M matrices);
		
	M subi(M matrixOperations);

	M mul(double d);
	M mul(M matrix);
	M muli(M matrix);

	M mmul(M matrix);

	M muli(double d);

	double dot(M matrices);

	M copy(M matrices);

	M addi(M matrices);

	M transpose();

	M divi(double v);

	int[] rowArgmaxs();

	void putColumn(int i, M matrixOperations);

	void putRow(int i, M matrixOperations);

	
	M add(double v);

	M addi(double v);

	int getLength();

	M rowSums();

	void put(int i, double j);
	
	void put(int a, int b,double c);


	M getRowRange(int offset, int i, int j);

	void reshape(int rows, int cols);

	void put(int[] indicies, int inputInd, M matrixOperations);

	M diviColumnVector(M matrixOperations);

	double sum();

	M get(int[] rows, int[] cols);

	int getRows();

	int getColumns();

	M dup();

	M getRow(int row);

	int[] findIndices();

	double get(int i, int j);

	M div(double v);

	M getColumns(int[] colInds);

	M getRows(int[] rowInds);

	M getColumn(int j);

	double get(int i);

	int argmax();


	
	
}
