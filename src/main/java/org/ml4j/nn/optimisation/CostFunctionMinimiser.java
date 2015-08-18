package org.ml4j.nn.optimisation;

import org.ml4j.DoubleMatrices;
import org.ml4j.DoubleMatrix;

public class CostFunctionMinimiser {

	/**
	 * Optimizes the weight matrix using a given cost function. Obtained from
	 * https://github.com/thomasjungblut/ A few minor changes were made to make
	 * the function compatible with jblas library.
	 */
	public static DoubleMatrices<DoubleMatrix>  fmincg(MinimisableCostAndGradientFunction f, DoubleMatrices<DoubleMatrix> pInput, int max_iter,
			boolean verbose) {

		/*
		 * Minimize a continuous differentialble multivariate function. Starting
		 * point is given by "X" (D by 1), and the function named in the string
		 * "f", must return a function value and a vector of partial
		 * derivatives. The Polack- Ribiere flavour of conjugate gradients is
		 * used to compute search directions, and a line search using quadratic
		 * and cubic polynomial approximations and the Wolfe-Powell stopping
		 * criteria is used together with the slope ratio method for guessing
		 * initial step sizes. Additionally a bunch of checks are made to make
		 * sure that exploration is taking place and that extrapolation will not
		 * be unboundedly large. The "length" gives the length of the run: if it
		 * is positive, it gives the maximum number of line searches, if
		 * negative its absolute gives the maximum allowed number of function
		 * evaluations. You can (optionally) give "length" a second component,
		 * which will indicate the reduction in function value to be expected in
		 * the first line-search (defaults to 1.0). The function returns when
		 * either its length is up, or if no further progress can be made (ie,
		 * we are at a minimum, or so close that due to numerical problems, we
		 * cannot get any closer). If the function terminates within a few
		 * iterations, it could be an indication that the function value and
		 * derivatives are not consistent (ie, there may be a bug in the
		 * implementation of your "f" function). The function returns the found
		 * solution "X", a vector of function values "fX" indicating the
		 * progress made and "i" the number of iterations (line searches or
		 * function evaluations, depending on the sign of "length") used.
		 * 
		 * Usage: [X, fX, i] = fmincg(f, X, options, P1, P2, P3, P4, P5)
		 * 
		 * See also: checkgrad
		 * 
		 * Copyright (C) 2001 and 2002 by Carl Edward Rasmussen. Date 2002-02-13
		 * 
		 * 
		 * (C) Copyright 1999, 2000 & 2001, Carl Edward Rasmussen Permission is
		 * granted for anyone to copy, use, or modify these programs and
		 * accompanying documents for purposes of research or education,
		 * provided this copyright notice is retained, and note is made of any
		 * changes that have been made.
		 * 
		 * These programs and documents are distributed without any warranty,
		 * express or implied. As the programs were written for research
		 * purposes only, they have not been tested to the degree that would be
		 * advisable in any important application. All use of these programs is
		 * entirely at the user's own risk.
		 * 
		 * [ml-class] Changes Made: 1) Function name and argument specifications
		 * 2) Output display
		 * 
		 * [tjungblut] Changes Made: 1) translated from octave to java 2) added
		 * an interface to exchange minimizers more easily BTW "fmincg" stands
		 * for Function minimize nonlinear conjugate gradient
		 * 
		 * [David Vincent] Changes Made: 1) changed matrix data structers and
		 * matrix operatons to use jblas library 2) removed (fX) column matrix
		 * that stored the cost of each iteration.
		 */
		final double RHO = 0.01; // a bunch of constants for line
		// searches
		final double SIG = 0.5; // RHO and SIG are the constants in
		// the
		// Wolfe-Powell conditions
		final double INT = 0.1; // don't reevaluate within 0.1 of the
		// limit of the current bracket
		final double EXT = 3.0; // extrapolate maximum 3 times the
		// current bracket
		final int MAX = 30; // max 20 function evaluations per line
		// search
		final int RATIO = 100; // maximum allowed slope ratio
		DoubleMatrices<DoubleMatrix> input = pInput;
		int M = 0;
		int i = 0; // zero the run length counter
		int red = 1; // starting point
		int ls_failed = 0; // no previous line search has failed
		// get function value and gradient
		final Tuple<Double, DoubleMatrices<DoubleMatrix>> evaluateCost = f.evaluateCost(input);
		double f1 = evaluateCost.getFirst();
		DoubleMatrices<DoubleMatrix> df1 = evaluateCost.getSecond();
		i = i + (max_iter < 0 ? 1 : 0);
		DoubleMatrices<DoubleMatrix> s = df1.mul(-1.0d); // search direction is
		// steepest

		double d1 = s.mul(-1.0d).dot(s); // this is the slope
		double z1 = red / (1.0 - d1); // initial step is red/(|s|+1)

		while (i < Math.abs(max_iter)) {
			i = i + (max_iter > 0 ? 1 : 0);// count iterations?!
			// make a copy of current values
			DoubleMatrices<DoubleMatrix> X0 = f.getDoubleMatricesFactory().copy(input);
			double f0 = f1;
			DoubleMatrices<DoubleMatrix> df0 = f.getDoubleMatricesFactory().copy(df1);
			// begin line search
			input = input.add(s.mul(z1));
			final Tuple<Double, DoubleMatrices<DoubleMatrix> > evaluateCost2 = f.evaluateCost(input);
			double f2 = evaluateCost2.getFirst();
			DoubleMatrices<DoubleMatrix>  df2 = evaluateCost2.getSecond();

			i = i + (max_iter < 0 ? 1 : 0); // count epochs?!
			double d2 = df2.dot(s);
			// initialize point 3 equal to point 1
			double f3 = f1;
			double d3 = d1;
			double z3 = -z1;
			if (max_iter > 0) {
				M = MAX;
			} else {
				M = Math.min(MAX, -max_iter - i);
			}
			// initialize quanteties
			int success = 0;
			double limit = -1;

			while (true) {
				while (((f2 > f1 + z1 * RHO * d1) | (d2 > -SIG * d1)) && (M > 0)) {
					limit = z1; // tighten the bracket
					double z2 = 0.0d;
					double A = 0.0d;
					double B = 0.0d;
					if (f2 > f1) {
						// quadratic fit
						z2 = z3 - (0.5 * d3 * z3 * z3) / (d3 * z3 + f2 - f3);
					} else {
						A = 6 * (f2 - f3) / z3 + 3 * (d2 + d3); // cubic fit
						B = 3 * (f3 - f2) - z3 * (d3 + 2 * d2);
						// numerical error possible - ok!
						z2 = (Math.sqrt(B * B - A * d2 * z3 * z3) - B) / A;
					}
					if (Double.isNaN(z2) || Double.isInfinite(z2)) {
						z2 = z3 / 2.0d; // if we had a numerical problem then
						// bisect
					}
					// don't accept too close to limits
					z2 = Math.max(Math.min(z2, INT * z3), (1 - INT) * z3);
					z1 = z1 + z2; // update the step
					input = input.add(s.mul(z2));
					final Tuple<Double, DoubleMatrices<DoubleMatrix> > evaluateCost3 = f.evaluateCost(input);
					f2 = evaluateCost3.getFirst();
					df2 = evaluateCost3.getSecond();
					M = M - 1;
					i = i + (max_iter < 0 ? 1 : 0); // count epochs?!
					d2 = df2.dot(s);
					z3 = z3 - z2; // z3 is now relative to the location of z2
				}
				if (f2 > f1 + z1 * RHO * d1 || d2 > -SIG * d1) {
					break; // this is a failure
				} else if (d2 > SIG * d1) {
					success = 1;
					break; // success
				} else if (M == 0) {
					break; // failure
				}
				double A = 6 * (f2 - f3) / z3 + 3 * (d2 + d3); // make cubic
				// extrapolation
				double B = 3 * (f3 - f2) - z3 * (d3 + 2 * d2);
				double z2 = -d2 * z3 * z3 / (B + Math.sqrt(B * B - A * d2 * z3 * z3));
				// num prob or wrong sign?
				if (Double.isNaN(z2) || Double.isInfinite(z2) || z2 < 0)
					if (limit < -0.5) { // if we have no upper limit
						z2 = z1 * (EXT - 1); // the extrapolate the maximum
						// amount
					} else {
						z2 = (limit - z1) / 2; // otherwise bisect
					}
				else if ((limit > -0.5) && (z2 + z1 > limit)) {
					// extraplation beyond max?
					z2 = (limit - z1) / 2; // bisect
				} else if ((limit < -0.5) && (z2 + z1 > z1 * EXT)) {
					// extrapolationbeyond limit
					z2 = z1 * (EXT - 1.0); // set to extrapolation limit
				} else if (z2 < -z3 * INT) {
					z2 = -z3 * INT;
				} else if ((limit > -0.5) && (z2 < (limit - z1) * (1.0 - INT))) {
					// too close to the limit
					z2 = (limit - z1) * (1.0 - INT);
				}
				// set point 3 equal to point 2
				f3 = f2;
				d3 = d2;
				z3 = -z2;
				z1 = z1 + z2;
				// update current estimates
				input = input.add(s.mul(z2));
				final Tuple<Double, DoubleMatrices<DoubleMatrix> > evaluateCost3 = f.evaluateCost(input);
				f2 = evaluateCost3.getFirst();
				df2 = evaluateCost3.getSecond();
				M = M - 1;
				i = i + (max_iter < 0 ? 1 : 0); // count epochs?!
				d2 = df2.dot(s);
			}// end of line search

			DoubleMatrices<DoubleMatrix>  tmp = null;

			if (success == 1) { // if line search succeeded
				f1 = f2;
				if (verbose)
					System.out.print("Iteration " + i + " | Cost: " + f1 + "\r");
				// Polack-Ribiere direction: s =
				// (df2'*df2-df1'*df2)/(df1'*df1)*s - df2;
				final double numerator = (df2.dot(df2) - df1.dot(df2)) / df1.dot(df1);
				s = s.mul(numerator).sub(df2);
				tmp = df1;
				df1 = df2;
				df2 = tmp; // swap derivatives
				d2 = df1.dot(s);
				if (d2 > 0) { // new slope must be negative
					s = df1.mul(-1.0d); // otherwise use steepest direction
					d2 = s.mul(-1.0d).dot(s);
				}
				// realmin in octave = 2.2251e-308
				// slope ratio but max RATIO
				z1 = z1 * Math.min(RATIO, d1 / (d2 - 2.2251e-308));
				d1 = d2;
				ls_failed = 0; // this line search did not fail
			} else {
				input = X0;
				f1 = f0;
				df1 = df0; // restore point from before failed line search
				// line search failed twice in a row?
				if (ls_failed == 1 || i > Math.abs(max_iter)) {
					break; // or we ran out of time, so we give up
				}
				tmp = df1;
				df1 = df2;
				df2 = tmp; // swap derivatives
				s = df1.mul(-1.0d); // try steepest
				d1 = s.mul(-1.0d).dot(s);
				z1 = 1.0d / (1.0d - d1);
				ls_failed = 1; // this line search failed
			}
		}

		return input;
	}
}
