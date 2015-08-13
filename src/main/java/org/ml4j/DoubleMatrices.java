package org.ml4j;



public interface DoubleMatrices<M> {

	DoubleMatrices<M> add(DoubleMatrices<M> matrices);

	public int getMatrixCount();
	
	public M[] getMatrices();
	
	DoubleMatrices<M> sub(DoubleMatrices<M> matrices);
		
	DoubleMatrices<M> subi(DoubleMatrices<M> matrices);

	DoubleMatrices<M> mul(double d);
	DoubleMatrices<M> muli(double d);

	double dot(DoubleMatrices<M> matrices);

	DoubleMatrices<M> copy(DoubleMatrices<M> matrices);

	DoubleMatrices<M> addi(DoubleMatrices<M> matrices);
}
