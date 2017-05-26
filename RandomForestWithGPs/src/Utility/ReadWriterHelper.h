/*
 * ReadWriterHelper.h
 *
 *  Created on: 14.07.2016
 *      Author: Max
 */

#ifndef UTILITY_READWRITERHELPER_H_
#define UTILITY_READWRITERHELPER_H_

#include "../Utility/Util.h"
#include "boost/filesystem.hpp"
#include "../Data/LabeledVectorX.h"
#include "../RandomForests/DynamicDecisionTree.h"
#include "../RandomForests/BigDynamicDecisionTree.h"

class ReadWriterHelper {
public:

	template<class T>
	static void writeMatrix(std::fstream& stream, const T& matrix);

	template<class T>
	static void readMatrix(std::fstream& stream, T& matrix);

	static void readVector(std::fstream& stream, VectorX& vector);

	static void writeVector(std::fstream& stream, const VectorX& vector);

	static void readPoint(std::fstream& stream, LabeledVectorX& vector);

	static void writePoint(std::fstream& stream, const LabeledVectorX& vector);

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
void ReadWriterHelper::writeMatrix(std::fstream& stream, const T& matrix){
	using Index = typename T::Index;
	Index rows = matrix.rows(), cols=matrix.cols();
	stream.write((char*) (&rows), sizeof(Index));
	stream.write((char*) (&cols), sizeof(Index));
	stream.write((char*) matrix.data(), rows*cols*sizeof(typename T::Scalar));
}

template<class T>
void ReadWriterHelper::readMatrix(std::fstream& stream, T& matrix){
	using Index = typename T::Index;
	Index rows=0, cols=0;
	stream.read((char*) (&rows),sizeof(Index));
	stream.read((char*) (&cols),sizeof(Index));
	matrix.resize(rows, cols);
	stream.read( (char *) matrix.data() , rows*cols*sizeof(typename T::Scalar));
}

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
