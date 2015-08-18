package org.ml4j.jblas;

import java.util.Vector;

import org.ml4j.DoubleMatrices;
import org.ml4j.DoubleMatrix;
import org.ml4j.nn.util.NeuralNetworkUtils;

public class FlattenedDoubleMatrices implements DoubleMatrices<DoubleMatrix> {

	private DoubleMatrix flattened;
	private int[][] topologies;
	
	protected FlattenedDoubleMatrices(DoubleMatrix flattened,int[][] topologies)
	{
		this.flattened = flattened;
		this.topologies = topologies;
	}
	
	private FlattenedDoubleMatrices verifiedCast(DoubleMatrices<DoubleMatrix> flattened)
	{
		if (flattened instanceof FlattenedDoubleMatrices)
		{
			return (FlattenedDoubleMatrices)flattened;
		}
		else
		{
			throw new IllegalArgumentException("Not instance of AltDoubleMatrices");
		}
	}
	
	protected FlattenedDoubleMatrices(Vector<DoubleMatrix> matrices,int[][] topologies)
	{
		if (matrices != null)
		{
			this.flattened = NeuralNetworkUtils.reshapeToVector(matrices);
		}
		this.topologies = topologies;
	}

	@Override
	public DoubleMatrices<DoubleMatrix> add(DoubleMatrices<DoubleMatrix> mul) {
		return new FlattenedDoubleMatrices(flattened.add(verifiedCast(mul).flattened),topologies);

	}

	@Override
	public int getMatrixCount() {
		return topologies.length;
	}

	@Override
	public DoubleMatrix[] getMatrices() {


		Vector<DoubleMatrix> reshaped = NeuralNetworkUtils.reshapeToList(flattened, topologies);
	    DoubleMatrix[] matrices = new DoubleMatrix[reshaped.size()];
		for (int i = 0; i < matrices.length; i++)
		{
			matrices[i] = reshaped.get(i);
		}
		return matrices;
	}

	@Override
	public DoubleMatrices<DoubleMatrix> sub(DoubleMatrices<DoubleMatrix> df2) {
		return new FlattenedDoubleMatrices(flattened.sub(verifiedCast(df2).flattened),topologies);
	}

	@Override
	public DoubleMatrices<DoubleMatrix> subi(DoubleMatrices<DoubleMatrix> df2) {
		flattened.subi(verifiedCast(df2).flattened);
		return this;
	}

	@Override
	public FlattenedDoubleMatrices mul(double d) {
		return new FlattenedDoubleMatrices(flattened.mul(d),topologies);
	}

	@Override
	public FlattenedDoubleMatrices muli(double d) {		// TODO Auto-generated method stub
		flattened.muli(d);
		return this;
	}

	@Override
	public double dot(DoubleMatrices<DoubleMatrix> s) {
		return flattened.dot(verifiedCast(s).flattened);
	}

	@Override
	public DoubleMatrices<DoubleMatrix> copy(DoubleMatrices<DoubleMatrix> source) {
		Vector<DoubleMatrix> matrices = new Vector<DoubleMatrix>();
		
		for (DoubleMatrix m : source.getMatrices())
		{
			matrices.add(m);
		}
		
		this.flattened = NeuralNetworkUtils.reshapeToVector(matrices);

		return new FlattenedDoubleMatrices(flattened.copy(verifiedCast(source).flattened),topologies);
	}

	@Override
	public DoubleMatrices<DoubleMatrix> addi(DoubleMatrices<DoubleMatrix> mul) {
		flattened.addi(verifiedCast(mul).flattened);
		return this;
	}

	
	

}
