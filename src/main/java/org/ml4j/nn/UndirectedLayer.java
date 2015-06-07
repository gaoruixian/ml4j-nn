package org.ml4j.nn;


public abstract class UndirectedLayer<L extends UndirectedLayer<L>> extends BaseLayer<L>{

	
	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;

	public UndirectedLayer(boolean retrainable) {
		super(retrainable);
	}

	

}
