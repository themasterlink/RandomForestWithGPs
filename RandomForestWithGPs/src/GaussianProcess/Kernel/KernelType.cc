/*
 * KernelType.cc
 *
 *  Created on: 31.10.2016
 *      Author: Max
 */

#include "KernelType.h"
#include "../../Utility/Util.h"

const std::vector<unsigned int> GaussianKernelParams::usedParamTypes = {LengthParam, FNoiseParam, SNoiseParam};

KernelElement::KernelElement(unsigned int kernelNr): m_kernelNr(kernelNr), m_values(nullptr), m_hasMoreThanOneDim(true){
};


KernelElement::~KernelElement(){
	delete[] m_values;
};

void KernelElement::changeAmountOfDims(const bool newHasMoreThanOneDim){
	if(hasMoreThanOneDim() != newHasMoreThanOneDim){
		if(hasMoreThanOneDim()){
			delete[] m_values;
		}else{
			delete m_values;
		}
		m_hasMoreThanOneDim = newHasMoreThanOneDim;
		if(hasMoreThanOneDim()){
			m_values = new double[ClassKnowledge::amountOfDims()];
		}else{
			m_values = new double;
		}
	}
}

GaussianKernelElementLength::GaussianKernelElementLength(bool hasMoreThanOneDim): GaussianKernelElement(LengthParam){
	m_hasMoreThanOneDim = hasMoreThanOneDim;
	if(m_hasMoreThanOneDim){
		m_values = new double[ClassKnowledge::amountOfDims()];
	}else{
		m_values = new double;
	}
}

GaussianKernelElementFNoise::GaussianKernelElementFNoise(): GaussianKernelElement(FNoiseParam){
	m_values = new double;
}

GaussianKernelElementSNoise::GaussianKernelElementSNoise(): GaussianKernelElement(SNoiseParam){
	m_values = new double;
}

GaussianKernelParams::GaussianKernelParams(const bool simpleLength):
	m_length(!simpleLength) {
	m_params[0] = &m_length;
	m_params[1] = &m_fNoise;
	m_params[2] = &m_sNoise;
	setAllValuesTo(0);
}

GaussianKernelParams::GaussianKernelParams(const OwnKernelInitParams& initParams) : m_length(!initParams.m_simpleLength) {
	m_params[0] = &m_length;
	m_params[1] = &m_fNoise;
	m_params[2] = &m_sNoise;
	setAllValuesTo(0);
}

void GaussianKernelParams::setAllValuesTo(const double value){
	for(unsigned int i = 0; i < paramsAmount; ++i){
		m_params[i]->setAllValuesTo(value);
	}
}

void KernelElement::setAllValuesTo(const double value){
	if(hasMoreThanOneDim()){
		for(unsigned int i = 0; i < ClassKnowledge::amountOfDims(); ++i){
			m_values[i] = value;
		}
	}else{
		m_values[0] = value;
	}
}

void KernelElement::addToFirstValue(const double value){
	if(!hasMoreThanOneDim()){
		m_values[0] += value;
	}else{
		printError("This function should not be called if there is more than one length value!");
	}
}


KernelElement* KernelTypeGenerator::getKernelFor(unsigned int kernelNr){
	KernelElement* ret = nullptr;
	switch(kernelNr){
	case LengthParam:
		ret = new GaussianKernelElementLength(true); // just to get the type
		break;
	case FNoiseParam:
		ret = new GaussianKernelElementFNoise();
		break;
	case SNoiseParam:
		ret = new GaussianKernelElementSNoise();
		break;
	default:
		printError("This type is not defined here!");
		break;
	}
	return ret;
}

void GaussianKernelParams::writeToFile(const std::string& filePath){
	std::fstream file;
	file.open(filePath, std::fstream::out | std::fstream::trunc);
	if(!file.is_open()){
		printError("The file \"" << filePath << "\" could not be opened!"); return;
	}
	bool hasMoreThanOneDim = m_length.hasMoreThanOneDim();
	file.write((char*) &hasMoreThanOneDim, sizeof(bool));
	if(hasMoreThanOneDim){
		long size = ClassKnowledge::amountOfDims();
		file.write((char*) (&size), sizeof(long));
		file.write((char*) m_length.getValues(), size * sizeof(double));
	}else{
		double len = m_length.getValue();
		file.write((char*) (&len), sizeof(double));
	}
	double fNoise = m_fNoise.getValue();
	file.write((char*) (&fNoise), sizeof(double));
	double sNoise = m_sNoise.getValue();
	file.write((char*) (&sNoise), sizeof(double));
	file.close();
}

void GaussianKernelParams::readFromFile(const std::string& filePath){
	std::fstream file(filePath, std::fstream::in);
	if(!file.is_open()){
		printError("The file \"" << filePath << "\" could not be opened!"); return;
	}
	bool hasMoreThanOneDim;
	file.read((char*) &hasMoreThanOneDim, sizeof(bool));
	m_length.changeAmountOfDims(hasMoreThanOneDim);
	if(hasMoreThanOneDim){
		long size = 0;
		file.read((char*) (&size), sizeof(long));
		file.read((char*) m_length.getValues(), size * sizeof(double));
	}else{
		double len = 0;
		file.read((char*) (&len), sizeof(double));
		m_length.setAllValuesTo(len);
	}
	double fNoise;
	file.read((char*) (&fNoise), sizeof(double));
	m_fNoise.setAllValuesTo(fNoise);
	double sNoise;
	file.read((char*) (&sNoise), sizeof(double));
	m_sNoise.setAllValuesTo(sNoise);
	file.close();
}

std::ostream& operator<<(std::ostream& stream, const GaussianKernelParams& params){
	stream << "len: ";
	if(params.m_length.hasMoreThanOneDim()){
		for(unsigned int i = 0; i < ClassKnowledge::amountOfDims(); ++i){
			stream << params.m_length.getValues()[i] << " ";
		}
	}else{
		stream << params.m_length.getValue();
	}
	stream << ", fNoise: " << params.m_fNoise.getValue() << ", sNoise: " << params.m_sNoise.getValue();
	return stream;
}
