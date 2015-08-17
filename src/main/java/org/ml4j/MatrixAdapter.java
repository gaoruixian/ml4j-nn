package org.ml4j;

import java.io.Serializable;

public interface MatrixAdapter extends Serializable,MatrixOperations<MatrixAdapter> {

	MatrixAdapter pow(int i);

	MatrixAdapter log();

	MatrixAdapter expi();
	
	MatrixAdapter sigmoid();


	MatrixAdapter powi(int d);

	MatrixAdapter logi();
	
	MatrixAdapter asJBlasMatrix();
	
	MatrixAdapter asCudaMatrix();
	
	public int hashCode();
	public boolean equals(Object o);

	
}
