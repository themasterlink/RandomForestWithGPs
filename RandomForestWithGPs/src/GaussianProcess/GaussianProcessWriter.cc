/*
 * GaussianProcessWriter.cc
 *
 *  Created on: 13.07.2016
 *      Author: Max
 */

#include "GaussianProcessWriter.h"
#include "../Utility/Util.h"
#include "../Utility/ReadWriterHelper.h"

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
		readFromStream(file, gp);
		file.close();
	}
}

void GaussianProcessWriter::readFromStream(std::fstream& file, GaussianProcess& gp){
	file.read((char*) &gp.m_dataPoints, sizeof(int));
	std::cout << "Amount of points: " << gp.m_dataPoints << std::endl;
	ReadWriterHelper::readMatrix(file, gp.m_dataMat);
	ReadWriterHelper::readVector(file, gp.m_a);
	ReadWriterHelper::readVector(file, gp.m_y);
	ReadWriterHelper::readVector(file, gp.m_f);
	ReadWriterHelper::readVector(file, gp.m_pi);
	ReadWriterHelper::readVector(file, gp.m_dLogPi);
	ReadWriterHelper::readVector(file, gp.m_ddLogPi);
	ReadWriterHelper::readVector(file, gp.m_sqrtDDLogPi);
	ReadWriterHelper::readVector(file, gp.m_sqrtDDLogPi);
	ReadWriterHelper::readMatrix(file, gp.m_innerOfLLT);
	gp.m_choleskyLLT.compute(gp.m_innerOfLLT);
	ReadWriterHelper::readMatrix(file, gp.m_kernel.m_differences);
	file.read((char*) &gp.m_kernel.m_hyperParams[0], sizeof(double)); // order is len, sigmaF, sigmaN
	file.read((char*) &gp.m_kernel.m_hyperParams[1], sizeof(double));
	file.read((char*) &gp.m_kernel.m_hyperParams[2], sizeof(double));
	gp.m_kernel.init(gp.m_dataMat);
	double mean, sd;
	file.read((char*) &mean, sizeof(double));
	file.read((char*) &sd, sizeof(double));
	gp.m_kernel.m_randLen.reset(mean, sd);
	file.read((char*) &mean, sizeof(double));
	file.read((char*) &sd, sizeof(double));
	gp.m_kernel.m_randSigmaF.reset(mean, sd);
	gp.m_trained = true;
	gp.m_init = true;
}

void GaussianProcessWriter::writeToStream(std::fstream& file, GaussianProcess& gp){
	file.write((char*) &gp.m_dataPoints, sizeof(int));
	ReadWriterHelper::writeMatrix(file, gp.m_dataMat);
	ReadWriterHelper::writeVector(file, gp.m_a);
	ReadWriterHelper::writeVector(file, gp.m_y);
	ReadWriterHelper::writeVector(file, gp.m_f);
	ReadWriterHelper::writeVector(file, gp.m_pi);
	ReadWriterHelper::writeVector(file, gp.m_dLogPi);
	ReadWriterHelper::writeVector(file, gp.m_ddLogPi);
	ReadWriterHelper::writeVector(file, gp.m_sqrtDDLogPi);
	ReadWriterHelper::writeVector(file, gp.m_sqrtDDLogPi);
	ReadWriterHelper::writeMatrix(file, gp.m_innerOfLLT);
	ReadWriterHelper::writeMatrix(file, gp.m_kernel.m_differences);
	file.write((char*) &gp.m_kernel.m_hyperParams[0], sizeof(double)); // order is len, sigmaF, sigmaN
	file.write((char*) &gp.m_kernel.m_hyperParams[1], sizeof(double));
	file.write((char*) &gp.m_kernel.m_hyperParams[2], sizeof(double));
	file.write((char*) &gp.m_kernel.m_randLen.m_mean, sizeof(double));
	file.write((char*) &gp.m_kernel.m_randLen.m_sd, sizeof(double));
	file.write((char*) &gp.m_kernel.m_randSigmaF.m_mean, sizeof(double));
	file.write((char*) &gp.m_kernel.m_randSigmaF.m_sd, sizeof(double));
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
		writeToStream(file, gp);
		file.close();
	}else{
		printError("The opening failed for: " << filePath);
		return;
	}
}
