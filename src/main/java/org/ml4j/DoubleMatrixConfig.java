package org.ml4j;

public class DoubleMatrixConfig {
	
	private static MatrixAdapterStrategy strategy = new DefaultMatrixAdapterStrategy();

	public static void setDoubleMatrixStrategy(MatrixAdapterStrategy strategy1) {
		strategy = strategy1;
	}
	
	public static MatrixAdapterStrategy getDoubleMatrixStrategy()
	{
		return strategy;
	}

}
