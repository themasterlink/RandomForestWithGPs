/*
 * RFGPWriter.h
 *
 *  Created on: 14.07.2016
 *      Author: Max
 */

#ifndef RANDOMFORESTGAUSSIANPROCESS_RFGPWRITER_H_
#define RANDOMFORESTGAUSSIANPROCESS_RFGPWRITER_H_

#include "RandomForestGaussianProcess.h"

class RFGPWriter {
public:

	static void writeToFile(const std::string& filePath, RandomForestGaussianProcess& rfgp);

	static void readFromFile(const std::string& filePath, RandomForestGaussianProcess& rfgp);

private:
	RFGPWriter();
	virtual ~RFGPWriter();
};

#endif /* RANDOMFORESTGAUSSIANPROCESS_RFGPWRITER_H_ */
