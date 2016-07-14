/*
 * GaussianProcessWriter.cc
 *
 *  Created on: 13.07.2016
 *      Author: Max
 */

#include "GaussianProcessWriter.h"

GaussianProcessWriter::GaussianProcessWriter() {
	// TODO Auto-generated constructor stub

}

GaussianProcessWriter::~GaussianProcessWriter() {
	// TODO Auto-generated destructor stub
}



void GaussianProcessWriter::readFromFile(const std::string& filePath, GaussianProcess& gp){
	if(filePath.length() == 0){
		printError("File path is empty!");
		return;
	}
	std::fstream file(filePath,std::ios::binary| std::ios::in);
	if(file.is_open()){
		file >> gp.m_dataPoints;
		readMatrix(file, gp.m_dataMat);
		readVector(file, gp.m_a);
		readVector(file, gp.m_y);
		readVector(file, gp.m_f);
		readVector(file, gp.m_pi);
		readVector(file, gp.m_dLogPi);
		readVector(file, gp.m_ddLogPi);
		readVector(file, gp.m_sqrtDDLogPi);
		readVector(file, gp.m_sqrtDDLogPi);
		readMatrix(file, gp.m_innerOfLLT);
		readMatrix(file, gp.m_kernel.m_differences);
		file >> gp.m_kernel.m_hyperParams[0]; // order is len, sigmaF, sigmaN
		file >> gp.m_kernel.m_hyperParams[1];
		file >> gp.m_kernel.m_hyperParams[2];
		gp.m_kernel.init(gp.m_dataMat);
		double mean, sd;
		file >> mean;
		file >> sd;
		gp.m_kernel.m_randLen.reset(mean, sd);
		file >> mean;
		file >> sd;
		gp.m_kernel.m_randSigmaF.reset(mean, sd);
		file.close();
	}

}

void GaussianProcessWriter::writeMatrix(std::fstream& stream, const Eigen::MatrixXd& matrix){
	Eigen::MatrixXd::Index rows = matrix.rows(), cols=matrix.cols();
	stream.write((char*) (&rows), sizeof(Eigen::MatrixXd::Index));
	stream.write((char*) (&cols), sizeof(Eigen::MatrixXd::Index));
	stream.write((char*) matrix.data(), rows*cols*sizeof(Eigen::MatrixXd::Scalar) );
}

void GaussianProcessWriter::readMatrix(std::fstream& stream, Eigen::MatrixXd& matrix){
	Eigen::MatrixXd::Index rows=0, cols=0;
	stream.read((char*) (&rows),sizeof(Eigen::MatrixXd::Index));
	stream.read((char*) (&cols),sizeof(Eigen::MatrixXd::Index));
	matrix.resize(rows, cols);
	stream.read( (char *) matrix.data() , rows*cols*sizeof(Eigen::MatrixXd::Scalar) );
}

void GaussianProcessWriter::readVector(std::fstream& stream, Eigen::VectorXd& vector){
	Eigen::MatrixXd::Index rows=0;
	stream.read((char*) (&rows),sizeof(Eigen::MatrixXd::Index));
	vector.resize(rows);
	stream.read( (char *) vector.data() , rows*sizeof(Eigen::MatrixXd::Scalar) );
}

void GaussianProcessWriter::writeToFile(const std::string& filePath, GaussianProcess& gp){
	if(filePath.length() == 0){
		printError("File path is empty!");
		return;
	}else if(gp.m_dataPoints == 0){
		printError("Number of data points of gp is zero -> writing not possible!");
		return;
	}
	std::fstream file(filePath,std::ios::out|std::ios::binary);
	if(file.is_open()){
		file << (int) gp.m_dataPoints << "\n";
		writeMatrix(file, gp.m_dataMat);
		writeMatrix(file, gp.m_a);
		writeMatrix(file, gp.m_y);
		writeMatrix(file, gp.m_f);
		writeMatrix(file, gp.m_pi);
		writeMatrix(file, gp.m_dLogPi);
		writeMatrix(file, gp.m_ddLogPi);
		writeMatrix(file, gp.m_sqrtDDLogPi);
		writeMatrix(file, gp.m_sqrtDDLogPi);
		writeMatrix(file, gp.m_innerOfLLT);
		writeMatrix(file, gp.m_kernel.m_differences);
		file << gp.m_kernel.m_hyperParams[0]; // order is len, sigmaF, sigmaN
		file << gp.m_kernel.m_hyperParams[1];
		file << gp.m_kernel.m_hyperParams[2];
		file << gp.m_kernel.m_randLen.m_mean;
		file << gp.m_kernel.m_randLen.m_sd;
		file << gp.m_kernel.m_randSigmaF.m_mean;
		file << gp.m_kernel.m_randSigmaF.m_sd;
		file.close();
	}else{
		printError("The opening failed for: " << filePath);
		return;
	}
}
