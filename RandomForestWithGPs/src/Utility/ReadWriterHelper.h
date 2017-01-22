/*
 * ReadWriterHelper.h
 *
 *  Created on: 14.07.2016
 *      Author: Max
 */

#ifndef UTILITY_READWRITERHELPER_H_
#define UTILITY_READWRITERHELPER_H_

#include <fstream>
#include <Eigen/Dense>
#include <vector>
#include <iostream>
#include "boost/filesystem.hpp"
#include "../Data/ClassPoint.h"
#include "../RandomForests/DynamicDecisionTree.h"
#include "../RandomForests/BigDynamicDecisionTree.h"

class ReadWriterHelper {
public:

	static 	void writeMatrix(std::fstream& stream, const Eigen::MatrixXd& matrix);

	static 	void readMatrix(std::fstream& stream, Eigen::MatrixXd& matrix);

	static 	void readVector(std::fstream& stream, Eigen::VectorXd& vector);

	static void writeVector(std::fstream& stream, const Eigen::VectorXd& vector);

	static 	void readPoint(std::fstream& stream, ClassPoint& vector);

	static void writePoint(std::fstream& stream, const ClassPoint& vector);

	template<class T>
	static void writeVector(std::fstream& stream, const std::vector<T>& vector);

	template<class T>
	static void readVector(std::fstream& stream, std::vector<T>& vector);

	static void writeDynamicTree(std::fstream& stream, const DynamicDecisionTree& tree);

	static void readDynamicTree(std::fstream& stream, DynamicDecisionTree& tree);

	static void writeBigDynamicTree(std::fstream& stream, const BigDynamicDecisionTree& tree);

	static void readBigDynamicTree(std::fstream& stream, BigDynamicDecisionTree& tree);

private:
	ReadWriterHelper();
	virtual ~ReadWriterHelper();
};

template<class T>
void ReadWriterHelper::writeVector(std::fstream& stream, const std::vector<T>& vector){
	long size = vector.size();
	stream.write((char*) &size, sizeof(long));
	stream.write((char*) &vector[0], sizeof(T) * size);
}

template<class T>
void ReadWriterHelper::readVector(std::fstream& stream, std::vector<T>& vector){
	long size;
	stream.read((char*) &size, sizeof(long));
	vector.resize(size);
	stream.read((char*) &vector[0], sizeof(T) * size);
}

#endif /* UTILITY_READWRITERHELPER_H_ */
