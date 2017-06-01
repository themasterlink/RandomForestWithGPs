/*
 * kernelCalc.h
 *
 *  Created on: 18.05.2016
 *      Author: Max
 */

#ifndef KERNELCALC_H_
#define KERNELCALC_H_

#include "Base/Types.h"

namespace Kernel2 {


/*
 Matrix xA(10,1);
 VectorX col; // input data
 xA(0,0) = 0;
 xA(1,0) = 0.1;
 xA(2,0) = 0.2;
 xA(3,0) =  0.3;
 xA(4,0) = 0.6;
 xA(5,0) =  1.;
 xA(6,0) =  1.1;
 xA(7,0) =  1.5;
 xA(8,0) =  1.9;
 xA(9,0) =  2.0;
 Matrix res;
 Kernel::getSeKernelFor(xA, xA, res, 5.0, 0.2);

 std::cout << "res: " << res << std::endl;
 Matrix inv = res.inverse();
 std::cout << "inv: " << inv << std::endl;
 */


//analytical solution not sure if needed here ...
void getSeKernelFor(const Matrix& xA, const Matrix& xB, Matrix& result,
		const Real sigma = 1.0, const Real l = 0.1){
	result = Matrix(xA.rows(), xB.rows());
	const Real sigmaSquared = sigma * sigma;
	const Real expFac = -1. / (2.0 * l * l);
	for(int i = 0; i < xA.rows(); ++i){
		const VectorX xARow = xA.row(i);
		for(int j = 0; j < xB.rows(); ++j){
			const VectorX diff = xARow - xB.row(j);
			result(i, j) = sigmaSquared * expReal(expFac * diff.dot(diff));
		}
	}
}

Matrix meanFun(const Matrix& x){
	Matrix mat;
	mat.Zero(x.rows(), 1);
	return mat;
}

struct MuSigma{
	Matrix mu;
	Matrix sigma;
};

// X = data
// f = resulting value
// xStar = new points
MuSigma predict(const Matrix& X, const Matrix& f, const Matrix& XStar,
		const Real noiseSigma = 0){
	Matrix K, Kinv, Kstar, Kstarstar, KstarTransposed;
	getSeKernelFor(X, X, K, 5, 0.1);
	K = noiseSigma * K;
	Kinv = K.inverse();
	getSeKernelFor(X, XStar, Kstar, 5., 0.1);
	getSeKernelFor(XStar, XStar, Kstarstar, 5., 0.1);
	MuSigma ret;
	KstarTransposed = Kstar.transpose();
	//ret.mu = meanFun(XStar) + KstarTransposed.dot(Kinv);// .dot(f - meanFun(X));
	//ret.sigma = Kstarstar - KstarTransposed.dot(Kinv); //.dot(Kstar);
	return ret;
}

} // close of namespace

#endif /* KERNELCALC_H_ */
