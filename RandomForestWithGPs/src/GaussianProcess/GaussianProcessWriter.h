/*
 * GaussianProcessWriter.h
 *
 *  Created on: 13.07.2016
 *      Author: Max
 */

#ifndef GAUSSIANPROCESS_GAUSSIANPROCESSWRITER_H_
#define GAUSSIANPROCESS_GAUSSIANPROCESSWRITER_H_

#ifdef BUILD_OLD_CODE

#include "GaussianProcess.h"
#include <fstream>

class GaussianProcessWriter {
public:

	static void readFromFile(const std::string& filePath, GaussianProcess& gp);

	static void writeToFile(const std::string& filePath, GaussianProcess& gp);

	static void writeToStream(std::fstream& file, GaussianProcess& gp);

	static void readFromStream(std::fstream& file, GaussianProcess& gp);

private:

	GaussianProcessWriter();
	virtual ~GaussianProcessWriter();
};

#endif // BUILD_OLD_CODE

#endif /* GAUSSIANPROCESS_GAUSSIANPROCESSWRITER_H_ */
