package org.ml4j.nn;

import org.jblas.DoubleMatrix;
import org.jblas.MatrixFunctions;

public class NeuralNetworkLayerActivation {

	private DoubleMatrix inputActivations;
	private DoubleMatrix outputActivations;
	private NeuralNetworkLayer layer;

	// private DoubleMatrix delta;

	public double getRegularisationCost(int m, double lambda) {
		DoubleMatrix currentTheta = layer.getThetas();

		// int m = X.getRows();

		int[] rows = new int[currentTheta.getRows()];
		int[] cols = new int[currentTheta.getColumns() - 1];
		for (int j = 0; j < currentTheta.getRows(); j++) {
			rows[j] = j;
		}
		for (int j = 1; j < currentTheta.getColumns(); j++) {
			cols[j - 1] = j;
		}
		double ThetaReg = MatrixFunctions.pow(currentTheta.get(rows, cols), 2).sum();
		return ((lambda) * ThetaReg) / (2 * m); // Add the non regularization
												// and regularization cost
												// together

	}

	protected NeuralNetworkLayerErrorGradient getErrorGradient(DoubleMatrix D, double lambda, int m) {
		// DoubleMatrix D = delta.get(i);
		DoubleMatrix inputActivations = getInputActivations();
		NeuralNetworkLayerErrorGradient grad = new NeuralNetworkLayerErrorGradient(getLayer(), D, m, lambda,
				inputActivations);
		return grad;
	}

	public NeuralNetworkLayerActivation(NeuralNetworkLayer layer, DoubleMatrix inputActivations,
			DoubleMatrix outputActivations) {
		this.inputActivations = inputActivations;

		this.outputActivations = outputActivations;
		this.layer = layer;
	}

	public DoubleMatrix getInputActivations() {
		return inputActivations;
	}

	public DoubleMatrix getOutputActivations() {
		return outputActivations;
	}

	public NeuralNetworkLayer getLayer() {
		return layer;
	}

	/*
	 * public DoubleMatrix getDelta() { return delta; } public DoubleMatrix
	 * getDELTA() { return getDelta().mmul(inputActivations); }
	 * 
	 * /* public void setDelta(NeuralNetworkLayerActivation previousActivation)
	 * { // TO FIX System.out.println("prev:" +
	 * previousActivation.getLayer().getLayerNum()); this.delta =
	 * getInnerCurrentD
	 * (previousActivation.delta,this.getInputActivations(),previousActivation
	 * .getLayer().getThetas(),layer.getThetas()).getMatrix();
	 * 
	 * }
	 */

	/*
	 * private DoubleMatrix getInnerCurrentD(DoubleMatrix currentD,DoubleMatrix
	 * inputAct,DoubleMatrix previousTheta,DoubleMatrix thisTheta) {
	 * 
	 * 
	 * System.out.println("InputAct:" + inputAct.getRows() + ":" +
	 * inputAct.getColumns()); System.out.println("ThisThetaTranspose:" +
	 * thisTheta.transpose().getRows() + ":" +
	 * thisTheta.transpose().getColumns());
	 * 
	 * DoubleMatrix sigable = new
	 * DoubleMatrix(inputAct.mmul(thisTheta.transpose()),"inputAct" +
	 * " times theta(somethin" + "-1)Transpose");
	 * 
	 * 
	 * sigable = new LabeledMatrix(DoubleMatrix.concatHorizontally(
	 * DoubleMatrix.ones(sigable.getMatrix().getRows()),
	 * sigable.getMatrix()),sigable.getLabel() + " withOnes");
	 * 
	 * 
	 * // System.out.println("First:" + "theta" + l + "Transpose"); //
	 * System.out.println("Second:" + currentD.getLabel() ); //
	 * System.out.println("Third:" + "transposedSigmoidGradientOf(" +
	 * sigable.getLabel() + "Transpose" + ")");
	 * 
	 * //System.out.println("Left:" + Theta.get(l).transpose().getRows() + ":" +
	 * Theta.get(l).transpose().getColumns()); //System.out.println("Right:" +
	 * currentD.getMatrix().getRows() + ":" +
	 * currentD.getMatrix().getColumns());
	 * 
	 * LabeledMatrix previousD = new
	 * LabeledMatrix(previousTheta.transpose().mmul
	 * (currentD).mul(NeuralNetwork.sigmoidGradiant
	 * (sigable.getMatrix().transpose())).transpose(),"thetaSomethin" +
	 * "Transpose times + "+ "currentD" + " times sigmoidGradientOf(" +
	 * sigable.getLabel() + ") transpose");
	 * //.mmul(NeuralNetwork.sigmoidGradiant(sigable).transpose()).transpose();
	 * 
	 * 
	 * //System.out.println(previousD.getLabel()); int [] rows = new
	 * int[previousD.getMatrix().getRows()]; int [] cols = new
	 * int[previousD.getMatrix().getColumns() - 1]; for (int j = 0;
	 * j<previousD.getMatrix().getRows(); j++ ) { rows[j]=j; } for (int j =1;
	 * j<previousD.getMatrix().getColumns();j++) { cols[j - 1]= j; }
	 * 
	 * previousD = new
	 * LabeledMatrix(previousD.getMatrix().get(rows,cols),previousD.getLabel());
	 * 
	 * 
	 * return new
	 * LabeledMatrix(previousD.getMatrix().transpose(),previousD.getLabel() +
	 * " Transpose");
	 * 
	 * }
	 */

	/*
	 * public void setDelta(DoubleMatrix Y) { this.delta =
	 * outputActivations.sub(Y).transpose(); }
	 */

	/*
	 * public DoubleMatrix getErrorGradients(int m, double lambda) { if (delta
	 * == null) { throw new RuntimeException("delta not set"); } return
	 * getGradients(getDELTA(),m,lambda); }
	 * 
	 * 
	 * public DoubleMatrix getGradients(DoubleMatrix DELTA,int m,double lambda)
	 * {
	 * 
	 * DoubleMatrix currentTheta = layer.getThetas(); DoubleMatrix modTheta =
	 * new DoubleMatrix().copy(currentTheta);
	 * modTheta.putColumn(0,DoubleMatrix.zeros(currentTheta.getRows(),1));
	 * return DELTA.div(m).add(modTheta.mul(lambda/m)) ;
	 * 
	 * 
	 * 
	 * }
	 */

}
