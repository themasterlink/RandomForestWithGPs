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
		file.read((char*) &gp.m_dataPoints, sizeof(int));
		std::cout << "Amount of data points: " << gp.m_dataPoints << std::endl;
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
		gp.m_choleskyLLT.compute(gp.m_innerOfLLT);
		readMatrix(file, gp.m_kernel.m_differences);
		file.read((char*) &gp.m_kernel.m_hyperParams[0], sizeof(double)); // order is len, sigmaF, sigmaN
		file.read((char*) &gp.m_kernel.m_hyperParams[1], sizeof(double));
		file.read((char*) &gp.m_kernel.m_hyperParams[2], sizeof(double));
		std::cout << gp.m_kernel.prettyString() << std::endl;
		gp.m_kernel.init(gp.m_dataMat);
		double mean, sd;
		file.read((char*) &mean, sizeof(double));
		file.read((char*) &sd, sizeof(double));
		gp.m_kernel.m_randLen.reset(mean, sd);
		file.read((char*) &mean, sizeof(double));
		file.read((char*) &sd, sizeof(double));
		gp.m_kernel.m_randSigmaF.reset(mean, sd);
		std::cout << "gp. sd: " << gp.m_kernel.m_randSigmaF.m_sd << std::endl;
		gp.m_trained = true;
		gp.m_init = true,
		file.close();
	}

}

void GaussianProcessWriter::writeMatrix(std::fstream& stream, const Eigen::MatrixXd& matrix){
	Eigen::MatrixXd::Index rows = matrix.rows(), cols=matrix.cols();
	stream.write((char*) (&rows), sizeof(Eigen::MatrixXd::Index));
	stream.write((char*) (&cols), sizeof(Eigen::MatrixXd::Index));
	stream.write((char*) matrix.data(), rows*cols*sizeof(Eigen::MatrixXd::Scalar) );
}

void GaussianProcessWriter::writeVector(std::fstream& stream, const Eigen::VectorXd& vector){
	Eigen::MatrixXd::Index rows = vector.rows();
	stream.write((char*) (&rows), sizeof(Eigen::MatrixXd::Index));
	stream.write((char*) vector.data(), rows*sizeof(Eigen::MatrixXd::Scalar));
}

void GaussianProcessWriter::readMatrix(std::fstream& stream, Eigen::MatrixXd& matrix){
	Eigen::MatrixXd::Index rows=0, cols=0;
	stream.read((char*) (&rows),sizeof(Eigen::MatrixXd::Index));
	stream.read((char*) (&cols),sizeof(Eigen::MatrixXd::Index));
	std::cout << "rows: " << rows << ", cols: " << cols << std::endl;
	matrix.resize(rows, cols);
	stream.read( (char *) matrix.data() , rows*cols*sizeof(Eigen::MatrixXd::Scalar) );
}

void GaussianProcessWriter::readVector(std::fstream& stream, Eigen::VectorXd& vector){
	Eigen::MatrixXd::Index rows=0;
	stream.read((char*) (&rows),sizeof(Eigen::MatrixXd::Index));
	std::cout << "rows: " << rows << std::endl;
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
		file.write((char*) &gp.m_dataPoints, sizeof(int));
		writeMatrix(file, gp.m_dataMat);
		writeVector(file, gp.m_a);
		writeVector(file, gp.m_y);
		writeVector(file, gp.m_f);
		writeVector(file, gp.m_pi);
		writeVector(file, gp.m_dLogPi);
		writeVector(file, gp.m_ddLogPi);
		writeVector(file, gp.m_sqrtDDLogPi);
		writeVector(file, gp.m_sqrtDDLogPi);
		writeMatrix(file, gp.m_innerOfLLT);
		writeMatrix(file, gp.m_kernel.m_differences);
		file.write((char*) &gp.m_kernel.m_hyperParams[0], sizeof(double)); // order is len, sigmaF, sigmaN
		file.write((char*) &gp.m_kernel.m_hyperParams[1], sizeof(double));
		file.write((char*) &gp.m_kernel.m_hyperParams[2], sizeof(double));
		file.write((char*) &gp.m_kernel.m_randLen.m_mean, sizeof(double));
		file.write((char*) &gp.m_kernel.m_randLen.m_sd, sizeof(double));
		file.write((char*) &gp.m_kernel.m_randSigmaF.m_mean, sizeof(double));
		file.write((char*) &gp.m_kernel.m_randSigmaF.m_sd, sizeof(double));
		std::cout << "gp. sd: " << gp.m_kernel.m_randSigmaF.m_sd << std::endl;
		file.close();
	}else{
		printError("The opening failed for: " << filePath);
		return;
	}
}
