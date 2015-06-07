package org.ml4j.nn;

import java.io.Serializable;

public abstract class BaseLayer<L extends BaseLayer<?>> implements Serializable {


	
	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;

	
	private boolean retrainable;


	public boolean isRetrainable() {
		return retrainable;
	}


	public void setRetrainable(boolean retrainable) {
		this.retrainable = retrainable;
	}
	
	public BaseLayer(boolean retrainable)
	{
		this.retrainable = retrainable;
	}

	public abstract L dup(boolean retrainable) ;
}
