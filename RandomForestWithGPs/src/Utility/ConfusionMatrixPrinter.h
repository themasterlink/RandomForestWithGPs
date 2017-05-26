/*
 * ConfusionMatrixPrinter.h
 *
 *  Created on: 23.07.2016
 *      Author: Max
 */

#ifndef UTILITY_CONFUSIONMATRIXPRINTER_H_
#define UTILITY_CONFUSIONMATRIXPRINTER_H_

#include "Util.h"

class ConfusionMatrixPrinter {
public:
	static void print(const Matrix& conv);

private:

	static int amountOfDigits(int number);

	ConfusionMatrixPrinter();
	virtual ~ConfusionMatrixPrinter();

};

#endif /* UTILITY_CONFUSIONMATRIXPRINTER_H_ */
