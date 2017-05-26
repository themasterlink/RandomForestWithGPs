/*
 * GaussianProcessMultiClass.h
 *
 *  Created on: 15.06.2016
 *      Author: Max
 */

#ifndef GAUSSIANPROCESS_GAUSSIANPROCESSMULTICLASS_H_
#define GAUSSIANPROCESS_GAUSSIANPROCESSMULTICLASS_H_

#include <Eigen/Cholesky>
#include "../Utility/Util.h"


class GaussianProcessMultiClass{
public:

	static void calcPhiBasedOnF(const VectorX& f, VectorX& pi, const int amountOfClasses, const int dataPoints);

	static void magicFunc(const int amountOfClasses, const int dataPoints, const std::vector<Matrix>& K_c, const VectorX& y);

private:
	GaussianProcessMultiClass();
	virtual ~GaussianProcessMultiClass();
};

#endif /* GAUSSIANPROCESS_GAUSSIANPROCESSMULTICLASS_H_ */
