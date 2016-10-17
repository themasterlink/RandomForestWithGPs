/*
 * ConfusionMatrixPrinter.h
 *
 *  Created on: 23.07.2016
 *      Author: Max
 */

#ifndef UTILITY_CONFUSIONMATRIXPRINTER_H_
#define UTILITY_CONFUSIONMATRIXPRINTER_H_

#include "Util.h"
#include <Eigen/Dense>

class ConfusionMatrixPrinter {
public:
	static void print(const Eigen::MatrixXd& conv, std::ostream& stream = std::cout);

private:

	static int amountOfDigits(int number);

	ConfusionMatrixPrinter();
	virtual ~ConfusionMatrixPrinter();

};

#endif /* UTILITY_CONFUSIONMATRIXPRINTER_H_ */
