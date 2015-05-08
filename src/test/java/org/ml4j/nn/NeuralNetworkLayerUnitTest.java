package org.ml4j.nn;

import static org.hamcrest.CoreMatchers.is;
import static org.hamcrest.CoreMatchers.notNullValue;
import static org.junit.Assert.assertThat;

import org.jblas.DoubleMatrix;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.ml4j.nn.activationfunctions.ActivationFunction;
import org.ml4j.nn.activationfunctions.SigmoidActivationFunction;
import org.mockito.runners.MockitoJUnitRunner;

@RunWith(MockitoJUnitRunner.class)
public class NeuralNetworkLayerUnitTest {
	
	@Test
	public void testUntrainedLayerConstructor_whenValidArguments()
	{
		ActivationFunction activationFunction = new SigmoidActivationFunction();
		NeuralNetworkLayer layer = new NeuralNetworkLayer(100,10,activationFunction);
		assertThat(layer, is(notNullValue()));
		assertThat(layer.getActivationFunction(), is(notNullValue()));
		assertThat(layer.getActivationFunction(), is(activationFunction));
		assertThat(layer.getInputNeuronCount(), is(100));
		assertThat(layer.getOutputNeuronCount(), is(10));
		DoubleMatrix layerThetas = layer.getClonedThetas();
		assertThat(layerThetas.getRows(),is(10));
		assertThat(layerThetas.getColumns(),is(101));
		assertThat(layer.isRetrainable(),is(true));

	}
	
	@Test(expected=IllegalArgumentException.class)
	public void testUntrainedLayerConstructor_whenActivationFunctionIsNull()
	{
			new NeuralNetworkLayer(100,10,null);
	}
	
	@Test
	public void testPretrainedRetrainableLayerConstructor_whenValidArguments()
	{
		ActivationFunction activationFunction = new SigmoidActivationFunction();
		DoubleMatrix thetas = new DoubleMatrix(10,101);
		NeuralNetworkLayer layer = new NeuralNetworkLayer(100,10,thetas,activationFunction,true);
		assertThat(layer, is(notNullValue()));
		assertThat(layer.getActivationFunction(), is(notNullValue()));
		assertThat(layer.getActivationFunction(), is(activationFunction));
		assertThat(layer.getInputNeuronCount(), is(100));
		assertThat(layer.getOutputNeuronCount(), is(10));
		DoubleMatrix layerThetas = layer.getClonedThetas();
		assertThat(layerThetas.getRows(),is(10));
		assertThat(layerThetas.getColumns(),is(101));
		assertThat(layer.isRetrainable(),is(true));

	}
	
	
	@Test(expected=IllegalArgumentException.class)
	public void testPretrainedLayerConstructor_whenActivationFunctionIsNull()
	{
		
			new NeuralNetworkLayer(100,10,new DoubleMatrix(100,10),null,true);
	}
	
	@Test(expected=IllegalArgumentException.class)
	public void testPretrainedLayerConstructor_whenThetasIsNull()
	{
		
			new NeuralNetworkLayer(100,10,null,new SigmoidActivationFunction(),true);
	}
	
	@Test(expected=IllegalArgumentException.class)
	public void testPretrainedLayerConstructor_whenThetasRowSizeIsIncorrect()
	{
		
			new NeuralNetworkLayer(100,10,new DoubleMatrix(9,101),new SigmoidActivationFunction(),true);
	}
	
	@Test(expected=IllegalArgumentException.class)
	public void testPretrainedLayerConstructor_whenThetasColumnSizeIsIncorrect()
	{
		
			new NeuralNetworkLayer(100,10,new DoubleMatrix(10,102),new SigmoidActivationFunction(),true);
	}
	
	@Test
	public void testPretrainedUnretrainableLayerConstructor_whenValidArguments()
	{
		ActivationFunction activationFunction = new SigmoidActivationFunction();
		DoubleMatrix thetas = new DoubleMatrix(10,101);
		NeuralNetworkLayer layer = new NeuralNetworkLayer(100,10,thetas,activationFunction,false);
		assertThat(layer, is(notNullValue()));
		assertThat(layer.getActivationFunction(), is(notNullValue()));
		assertThat(layer.getActivationFunction(), is(activationFunction));
		assertThat(layer.getInputNeuronCount(), is(100));
		assertThat(layer.getOutputNeuronCount(), is(10));
		DoubleMatrix layerThetas = layer.getClonedThetas();
		assertThat(layerThetas.getRows(),is(10));
		assertThat(layerThetas.getColumns(),is(101));
		assertThat(layer.isRetrainable(),is(false));
	}
	


}
