/*
 * kernelCalc.h
 *
 *  Created on: 18.05.2016
 *      Author: Max
 */

#ifndef KERNELCALC_H_
#define KERNELCALC_H_

#include <Eigen/Dense>
#include <math.h>

namespace Kernel {


/*
 Eigen::MatrixXd xA(10,1);
 Eigen::VectorXd col; // input data
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
 Eigen::MatrixXd res;
 Kernel::getSeKernelFor(xA, xA, res, 5.0, 0.2);

 std::cout << "res: " << res << std::endl;
 Eigen::MatrixXd inv = res.inverse();
 std::cout << "inv: " << inv << std::endl;
 */


//analytical solution not sure if needed here ...
void getSeKernelFor(const Eigen::MatrixXd& xA, const Eigen::MatrixXd& xB, Eigen::MatrixXd& result,
		const double sigma = 1.0, const double l = 0.1){
	result = Eigen::MatrixXd(xA.rows(), xB.rows());
	const double sigmaSquared = sigma * sigma;
	const double expFac = -1. / (2.0 * l * l);
	for(int i = 0; i < xA.rows(); ++i){
		const Eigen::VectorXd xARow = xA.row(i);
		for(int j = 0; j < xB.rows(); ++j){
			const Eigen::VectorXd diff = xARow - xB.row(j);
			result(i, j) = sigmaSquared * exp(expFac * diff.dot(diff));
		}
	}
}

Eigen::MatrixXd meanFun(const Eigen::MatrixXd& x){
	Eigen::MatrixXd mat;
	mat.Zero(x.rows(), 1);
	return mat;
}

struct MuSigma{
	Eigen::MatrixXd mu;
	Eigen::MatrixXd sigma;
};

// X = data
// f = resulting value
// xStar = new points
MuSigma predict(const Eigen::MatrixXd& X, const Eigen::MatrixXd& f, const Eigen::MatrixXd& XStar,
		const double noiseSigma = 0){
	Eigen::MatrixXd K, Kinv, Kstar, Kstarstar, KstarTransposed;
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
