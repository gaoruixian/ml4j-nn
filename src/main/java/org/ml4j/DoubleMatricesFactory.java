package org.ml4j;

import java.util.Vector;

public interface DoubleMatricesFactory<M> {

	public DoubleMatrices<M> copy(DoubleMatrices<M> doubleMatrices);
	public DoubleMatrices<M> create(Vector<M> doubleMatrices);

}
