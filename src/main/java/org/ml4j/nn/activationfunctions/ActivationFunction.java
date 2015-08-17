package org.ml4j.nn.activationfunctions;

import java.io.Serializable;

import org.ml4j.DoubleMatrix;

public interface ActivationFunction extends Serializable {

	public DoubleMatrix activate(DoubleMatrix input);

}
