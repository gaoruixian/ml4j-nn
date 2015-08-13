package org.ml4j.cuda;

import java.util.Vector;

import org.ml4j.DoubleMatrices;
import org.ml4j.DoubleMatricesFactory;

public class FlattenedDoubleMatricesFactory implements DoubleMatricesFactory<DoubleMatrix> {

	private int[][] topologies;
	
	public FlattenedDoubleMatricesFactory(int[][] topologies)
	{
		this.topologies = topologies;
	}
	
	@Override
	public DoubleMatrices<DoubleMatrix> copy(DoubleMatrices<DoubleMatrix> doubleMatrices) {
		return new FlattenedDoubleMatrices(new DoubleMatrix(),topologies).copy(doubleMatrices);
	}

	@Override
	public DoubleMatrices<DoubleMatrix> create(
			Vector<DoubleMatrix> doubleMatrices) {
		return new FlattenedDoubleMatrices(doubleMatrices,topologies);
	}

}
