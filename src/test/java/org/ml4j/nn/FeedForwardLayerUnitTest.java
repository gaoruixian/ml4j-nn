package org.ml4j.nn;

import static org.hamcrest.CoreMatchers.is;
import static org.hamcrest.CoreMatchers.notNullValue;
import static org.junit.Assert.assertThat;

import org.hamcrest.CoreMatchers;
import org.jblas.DoubleMatrix;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.ml4j.nn.activationfunctions.DifferentiableActivationFunction;
import org.ml4j.nn.activationfunctions.SigmoidActivationFunction;
import org.mockito.runners.MockitoJUnitRunner;

@RunWith(MockitoJUnitRunner.class)
public class FeedForwardLayerUnitTest {
	
	@Test
	public void testUntrainedLayerConstructor_whenValidArguments()
	{
		DifferentiableActivationFunction activationFunction = new SigmoidActivationFunction();
		FeedForwardLayer layer = new FeedForwardLayer(100,10,activationFunction);
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
	
	@Test
	public void testUpdateRetrainableLayerThetas_whenValidArguments_whenPermitFurtherRetrainsFalse()
	{
		DifferentiableActivationFunction activationFunction = new SigmoidActivationFunction();

		FeedForwardLayer layer = new FeedForwardLayer(100,10,activationFunction);
		DoubleMatrix thetas = new DoubleMatrix(10,101);

		assertThat(layer.isRetrainable(),is(true));
		assertThat(layer.getClonedThetas(),CoreMatchers.not(thetas));

		layer.updateThetas(thetas, 0, false);
		
		assertThat(layer.getClonedThetas(),CoreMatchers.is(thetas));

		assertThat(layer.isRetrainable(),is(false));
		
	}
	
	
	@Test
	public void testUpdateRetrainableLayerThetas_whenValidArguments_whenPermitFurtherRetrainsTrue()
	{
		DifferentiableActivationFunction activationFunction = new SigmoidActivationFunction();

		FeedForwardLayer layer = new FeedForwardLayer(100,10,activationFunction);
		DoubleMatrix thetas = new DoubleMatrix(10,101);

		assertThat(layer.isRetrainable(),is(true));
		assertThat(layer.getClonedThetas(),CoreMatchers.not(thetas));

		layer.updateThetas(thetas, 0, true);
		
		assertThat(layer.getClonedThetas(),CoreMatchers.is(thetas));

		assertThat(layer.isRetrainable(),is(true));
		
	}
	
	@Test
	public void testDup_settingNotRetrainable()
	{
		DifferentiableActivationFunction activationFunction = new SigmoidActivationFunction();
		FeedForwardLayer layer = new FeedForwardLayer(100,10,activationFunction);
		assertThat(layer.isRetrainable(),is(true));

		FeedForwardLayer dupLayer = layer.dup(false);
		
		assertThat(layer.isRetrainable(),is(true));

		assertThat(dupLayer.isRetrainable(),is(false));
		assertThat(dupLayer.getClonedThetas(),CoreMatchers.is(layer.getClonedThetas()));
		assertThat(dupLayer.getActivationFunction(),CoreMatchers.is(layer.getActivationFunction()));
		assertThat(dupLayer == layer,is(false));

	}
	
	
	@Test
	public void testActivate_DoubleMatrix()
	{
		DifferentiableActivationFunction activationFunction = new SigmoidActivationFunction();
		FeedForwardLayer layer = new FeedForwardLayer(100,10,activationFunction);
		DoubleMatrix inputs = DoubleMatrix.rand(10, 100);
		DoubleMatrix activations = layer.activate(inputs);
		DoubleMatrix inputsWithIntercept = DoubleMatrix.concatHorizontally(DoubleMatrix.ones(10),inputs);
		assertThat(activations,is(activationFunction.activate(inputsWithIntercept.mmul(layer.getClonedThetas().transpose()))));
	}
	
	@Test
	public void testForwardPropagate()
	{
		DifferentiableActivationFunction activationFunction = new SigmoidActivationFunction();
		FeedForwardLayer layer = new FeedForwardLayer(100,10,activationFunction);
		DoubleMatrix inputs = DoubleMatrix.rand(10, 101);
		NeuralNetworkLayerActivation activation = layer.forwardPropagate(inputs);
		assertThat(activation.getOutputActivations(),is(activationFunction.activate(inputs.mmul(layer.getClonedThetas().transpose()))));
		assertThat(activation.getInputActivations(),is(inputs));
		assertThat(activation.getLayer(),is(layer));
	}
	
	@Test(expected=IllegalArgumentException.class)
	public void testActivate_IncorrectColumns()
	{
		DifferentiableActivationFunction activationFunction = new SigmoidActivationFunction();
		FeedForwardLayer layer = new FeedForwardLayer(100,10,activationFunction);
		DoubleMatrix inputs = DoubleMatrix.rand(10, 101);
		layer.activate(inputs);
	}
	
	@Test(expected=IllegalArgumentException.class)
	public void testForwardPropagate_IncorrectColumns()
	{
		DifferentiableActivationFunction activationFunction = new SigmoidActivationFunction();
		FeedForwardLayer layer = new FeedForwardLayer(100,10,activationFunction);
		DoubleMatrix inputs = DoubleMatrix.rand(10, 100);
		layer.forwardPropagate(inputs);
	}
	
	@Test
	public void testActivate_Arrays()
	{
		DifferentiableActivationFunction activationFunction = new SigmoidActivationFunction();
		FeedForwardLayer layer = new FeedForwardLayer(100,10,activationFunction);
		DoubleMatrix inputs = DoubleMatrix.rand(10, 100);
		double[][] inputArrays = inputs.toArray2();
		DoubleMatrix activations = layer.activate(inputArrays);
		DoubleMatrix inputsWithIntercept = DoubleMatrix.concatHorizontally(DoubleMatrix.ones(10),inputs);

		assertThat(activations,is(activationFunction.activate(inputsWithIntercept.mmul(layer.getClonedThetas().transpose()))));
	}
	
	@Test
	public void testGetClonedThetas()
	{
		DifferentiableActivationFunction activationFunction = new SigmoidActivationFunction();
		FeedForwardLayer layer = new FeedForwardLayer(100,10,activationFunction);
		DoubleMatrix thetas = new DoubleMatrix(10,101);

		layer.updateThetas(thetas, 0, true);
		
		
		assertThat(layer.getClonedThetas(),CoreMatchers.is(thetas));
		
		assertThat(layer.getClonedThetas() == thetas,is(false));
	}
	
	@Test
	public void testDup_settingRetrainable()
	{
		DifferentiableActivationFunction activationFunction = new SigmoidActivationFunction();
		FeedForwardLayer layer = new FeedForwardLayer(100,10,activationFunction);
		assertThat(layer.isRetrainable(),is(true));

		FeedForwardLayer dupLayer = layer.dup(true);
		
		assertThat(layer.isRetrainable(),is(true));

		assertThat(dupLayer.isRetrainable(),is(true));
		assertThat(dupLayer.getClonedThetas(),CoreMatchers.is(layer.getClonedThetas()));
		assertThat(dupLayer.getActivationFunction(),CoreMatchers.is(layer.getActivationFunction()));
		assertThat(dupLayer == layer,is(false));

	}
	
	@Test(expected=IllegalStateException.class)
	public void testUpdateNotRetrainableLayerThetas_whenValidArguments()
	{
		DifferentiableActivationFunction activationFunction = new SigmoidActivationFunction();

		FeedForwardLayer layer = new FeedForwardLayer(100,10,activationFunction);
		layer.setRetrainable(false);
		DoubleMatrix thetas = new DoubleMatrix(10,101);

		assertThat(layer.isRetrainable(),is(false));
		assertThat(layer.getClonedThetas(),CoreMatchers.not(thetas));

		layer.updateThetas(thetas, 0, true);
		
	}
	
	@Test(expected=IllegalArgumentException.class)
	public void testUpdateRetrainableLayerThetas_whenThetasIncorrectRows()
	{
		DifferentiableActivationFunction activationFunction = new SigmoidActivationFunction();

		FeedForwardLayer layer = new FeedForwardLayer(100,10,activationFunction);
		DoubleMatrix thetas = new DoubleMatrix(11,101);

		layer.updateThetas(thetas, 0, true);
		
	}
	
	@Test(expected=IllegalArgumentException.class)
	public void testUpdateRetrainableLayerThetas_whenThetasIncorrectColumns()
	{
		DifferentiableActivationFunction activationFunction = new SigmoidActivationFunction();
		FeedForwardLayer layer = new FeedForwardLayer(100,10,activationFunction);
		DoubleMatrix thetas = new DoubleMatrix(10,100);
		layer.updateThetas(thetas, 0, true);
		
	}
	
	@Test(expected=IllegalArgumentException.class)
	public void testUpdateRetrainableLayerThetas_whenLayerNumberBelowZero()
	{
		DifferentiableActivationFunction activationFunction = new SigmoidActivationFunction();
		FeedForwardLayer layer = new FeedForwardLayer(100,10,activationFunction);
		DoubleMatrix thetas = new DoubleMatrix(10,101);
		layer.updateThetas(thetas, -1, true);
		
	}

	
	@Test(expected=IllegalArgumentException.class)
	public void testUntrainedLayerConstructor_whenActivationFunctionIsNull()
	{
			new FeedForwardLayer(100,10,null);
	}
	
	@Test
	public void testPretrainedRetrainableLayerConstructor_whenValidArguments()
	{
		DifferentiableActivationFunction activationFunction = new SigmoidActivationFunction();
		DoubleMatrix thetas = new DoubleMatrix(10,101);
		FeedForwardLayer layer = new FeedForwardLayer(100,10,thetas,activationFunction,true);
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
		
			new FeedForwardLayer(100,10,new DoubleMatrix(100,10),null,true);
	}
	
	@Test(expected=IllegalArgumentException.class)
	public void testPretrainedLayerConstructor_whenThetasIsNull()
	{
		
			new FeedForwardLayer(100,10,null,new SigmoidActivationFunction(),true);
	}
	
	@Test(expected=IllegalArgumentException.class)
	public void testPretrainedLayerConstructor_whenThetasRowSizeIsIncorrect()
	{
		
			new FeedForwardLayer(100,10,new DoubleMatrix(9,101),new SigmoidActivationFunction(),true);
	}
	
	@Test(expected=IllegalArgumentException.class)
	public void testPretrainedLayerConstructor_whenThetasColumnSizeIsIncorrect()
	{
		
			new FeedForwardLayer(100,10,new DoubleMatrix(10,102),new SigmoidActivationFunction(),true);
	}
	
	@Test
	public void testPretrainedUnretrainableLayerConstructor_whenValidArguments()
	{
		DifferentiableActivationFunction activationFunction = new SigmoidActivationFunction();
		DoubleMatrix thetas = new DoubleMatrix(10,101);
		FeedForwardLayer layer = new FeedForwardLayer(100,10,thetas,activationFunction,false);
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
