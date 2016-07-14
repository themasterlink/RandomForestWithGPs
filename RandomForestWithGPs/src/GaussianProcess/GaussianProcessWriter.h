/*
 * GaussianProcessWriter.h
 *
 *  Created on: 13.07.2016
 *      Author: Max
 */

#ifndef GAUSSIANPROCESS_GAUSSIANPROCESSWRITER_H_
#define GAUSSIANPROCESS_GAUSSIANPROCESSWRITER_H_

#include "GaussianProcess.h"

class GaussianProcessWriter {
public:

	static void readFromFile(const std::string& filePath, GaussianProcess& gp);

	static void writeToFile(const std::string& filePath, GaussianProcess& gp);

private:

	static 	void writeMatrix(std::fstream& stream, const Eigen::MatrixXd& matrix);

	static 	void readMatrix(std::fstream& stream, Eigen::MatrixXd& matrix);

	static 	void readVector(std::fstream& stream, Eigen::VectorXd& vector);

	static void writeVector(std::fstream& stream, const Eigen::VectorXd& vector);

	GaussianProcessWriter();
	virtual ~GaussianProcessWriter();
};

#endif /* GAUSSIANPROCESS_GAUSSIANPROCESSWRITER_H_ */
