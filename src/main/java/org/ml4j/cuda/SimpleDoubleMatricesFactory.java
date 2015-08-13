package org.ml4j.cuda;

import java.util.Vector;

import org.ml4j.DoubleMatrices;
import org.ml4j.DoubleMatricesFactory;

public class SimpleDoubleMatricesFactory implements DoubleMatricesFactory<DoubleMatrix> {

	private int[][] topologies;
	
	public SimpleDoubleMatricesFactory(int[][] topologies)
	{
		this.topologies = topologies;
	}
	
	@Override
	public DoubleMatrices<DoubleMatrix> copy(DoubleMatrices<DoubleMatrix> doubleMatrices) {
		return new SimpleDoubleMatrices(topologies.length).copy(doubleMatrices);
	}

	@Override
	public DoubleMatrices<DoubleMatrix> create(
			Vector<DoubleMatrix> doubleMatrices) {
		return new SimpleDoubleMatrices(doubleMatrices);
	}

}