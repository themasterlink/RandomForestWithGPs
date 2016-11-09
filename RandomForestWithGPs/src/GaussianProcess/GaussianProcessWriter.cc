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
	bool hasMoreThanOneDim;
	file.read((char*) &hasMoreThanOneDim, sizeof(bool));
	if(hasMoreThanOneDim){
		double noiseF, noiseS;
		std::vector<double> length;
		ReadWriterHelper::readVector(file, length);
		file.read((char*) &noiseF, sizeof(double));
		file.read((char*) &noiseS, sizeof(double));
		gp.m_kernel.setHyperParams(length, noiseF , noiseS);
	}else{
		double len, noiseF, noiseS;
		file.read((char*) &len, sizeof(double)); // order is len, sigmaF, sigmaN
		file.read((char*) &noiseF, sizeof(double));
		file.read((char*) &noiseS, sizeof(double));
		gp.m_kernel.setHyperParams(len, noiseF, noiseS);
	}
	gp.m_kernel.init(gp.m_dataMat);
	/*double mean, sd;
	file.read((char*) &mean, sizeof(double));
	file.read((char*) &sd, sizeof(double));
	gp.m_kernel.m_randLen.reset(mean, sd);
	file.read((char*) &mean, sizeof(double));
	file.read((char*) &sd, sizeof(double));
	gp.m_kernel.m_randSigmaF.reset(mean, sd);*/
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
	bool hasMoreThanOneLength = gp.m_kernel.hasLengthMoreThanOneDim();
	file.write((char*) &hasMoreThanOneLength, sizeof(bool)); // order is len, sigmaF, sigmaN
	if(hasMoreThanOneLength){
		std::vector<double> length(ClassKnowledge::amountOfDims());
		for(unsigned int i = 0; i < length.size(); ++i){
			length[i] = gp.m_kernel.getHyperParams().m_length.getValues()[i];
		}
		ReadWriterHelper::writeVector(file, length);
	}else{
		double len = gp.m_kernel.getHyperParams().m_length.getValue();
		file.write((char*) &len, sizeof(double));
	}
	double noiseF = gp.m_kernel.getHyperParams().m_fNoise.getValue();
	double noiseS = gp.m_kernel.getHyperParams().m_sNoise.getValue();
	file.write((char*) &noiseF, sizeof(double));
	file.write((char*) &noiseS, sizeof(double));
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
