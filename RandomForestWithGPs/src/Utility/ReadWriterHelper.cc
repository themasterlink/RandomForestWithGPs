/*
 * ReadWriterHelper.cc
 *
 *  Created on: 14.07.2016
 *      Author: Max
 */

#include "ReadWriterHelper.h"

ReadWriterHelper::ReadWriterHelper() {
	// TODO Auto-generated constructor stub

}

ReadWriterHelper::~ReadWriterHelper() {
	// TODO Auto-generated destructor stub
}

void ReadWriterHelper::writeMatrix(std::fstream& stream, const Eigen::MatrixXd& matrix){
	Eigen::MatrixXd::Index rows = matrix.rows(), cols=matrix.cols();
	stream.write((char*) (&rows), sizeof(Eigen::MatrixXd::Index));
	stream.write((char*) (&cols), sizeof(Eigen::MatrixXd::Index));
	stream.write((char*) matrix.data(), rows*cols*sizeof(Eigen::MatrixXd::Scalar) );
}

void ReadWriterHelper::readMatrix(std::fstream& stream, Eigen::MatrixXd& matrix){
	Eigen::MatrixXd::Index rows=0, cols=0;
	stream.read((char*) (&rows),sizeof(Eigen::MatrixXd::Index));
	stream.read((char*) (&cols),sizeof(Eigen::MatrixXd::Index));
	matrix.resize(rows, cols);
	stream.read( (char *) matrix.data() , rows*cols*sizeof(Eigen::MatrixXd::Scalar) );
}

void ReadWriterHelper::writeVector(std::fstream& stream, const Eigen::VectorXd& vector){
	Eigen::MatrixXd::Index rows = vector.rows();
	stream.write((char*) (&rows), sizeof(Eigen::MatrixXd::Index));
	stream.write((char*) vector.data(), rows*sizeof(Eigen::MatrixXd::Scalar));
}

void ReadWriterHelper::readVector(std::fstream& stream, Eigen::VectorXd& vector){
	Eigen::MatrixXd::Index rows=0;
	stream.read((char*) (&rows),sizeof(Eigen::MatrixXd::Index));
	vector.resize(rows);
	stream.read( (char *) vector.data(), rows*sizeof(Eigen::MatrixXd::Scalar));
}

void ReadWriterHelper::readPoint(std::fstream& stream, ClassPoint& vector){
	Eigen::MatrixXd::Index rows=0;
	stream.read((char*) (&rows),sizeof(Eigen::MatrixXd::Index));
	vector.resize(rows);
	stream.read( (char *) vector.data(), rows*sizeof(Eigen::MatrixXd::Scalar));
	unsigned int label;
	stream.read( (char *) (&label), sizeof(unsigned int));
	vector.setLabel(label);
}

void ReadWriterHelper::writePoint(std::fstream& stream, const ClassPoint& vector){
	Eigen::MatrixXd::Index rows = vector.rows();
	stream.write((char*) (&rows), sizeof(Eigen::MatrixXd::Index));
	stream.write((char*) vector.data(), rows*sizeof(Eigen::MatrixXd::Scalar));
	unsigned int label = vector.getLabel();
	stream.write((char*) (&label), sizeof(unsigned int));
}
