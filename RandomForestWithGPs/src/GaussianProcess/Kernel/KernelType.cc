/*
 * KernelType.cc
 *
 *  Created on: 31.10.2016
 *      Author: Max
 */

#include "KernelType.h"
#include "../../Utility/Util.h"

const std::vector<unsigned int> GaussianKernelParams::usedParamTypes = {LengthParam, FNoiseParam, SNoiseParam};

KernelElement::KernelElement(unsigned int kernelNr): m_kernelNr(kernelNr), m_values(nullptr), m_hasMoreThanOneDim(false){
};

KernelElement::KernelElement(const KernelElement& ele):
		m_kernelNr(ele.m_kernelNr), m_values(nullptr), m_hasMoreThanOneDim(ele.m_hasMoreThanOneDim){
	if(ele.m_hasMoreThanOneDim){
		const unsigned int dim = ClassKnowledge::amountOfDims();
		m_values = new Real[dim];
		for(unsigned int i = 0; i < dim; ++i){
			m_values[i] = ele.m_values[i];
		}
	}else{
		m_values = new Real[1]; // [1] to make the delete easier
		m_values[0] = ele.m_values[0];
	}
}

KernelElement::~KernelElement(){
	delete[] m_values;
	m_values = nullptr;
};

void KernelElement::changeAmountOfDims(const bool newHasMoreThanOneDim){
	if(hasMoreThanOneDim() != newHasMoreThanOneDim){
		delete[] m_values;
		m_values = nullptr;
		m_hasMoreThanOneDim = newHasMoreThanOneDim;
		if(hasMoreThanOneDim()){
			m_values = new Real[ClassKnowledge::amountOfDims()];
		}else{
			m_values = new Real[1]; // [1] to make the delete easier
		}
	}
}

GaussianKernelElementLength::GaussianKernelElementLength(bool hasMoreThanOneDim): GaussianKernelElement(LengthParam){
	m_hasMoreThanOneDim = hasMoreThanOneDim;
	if(m_hasMoreThanOneDim){
		m_values = new Real[ClassKnowledge::amountOfDims()];
	}else{
		m_values = new Real[1];
	}
}

GaussianKernelElementFNoise::GaussianKernelElementFNoise(): GaussianKernelElement(FNoiseParam){
	m_values = new Real[1];
}

GaussianKernelElementSNoise::GaussianKernelElementSNoise(): GaussianKernelElement(SNoiseParam){
	m_values = new Real[1];
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

void GaussianKernelParams::setAllValuesTo(const Real value){
	for(unsigned int i = 0; i < paramsAmount; ++i){
		m_params[i]->setAllValuesTo(value);
	}
}

void KernelElement::setAllValuesTo(const Real value){
	if(hasMoreThanOneDim()){
		for(unsigned int i = 0; i < ClassKnowledge::amountOfDims(); ++i){
			m_values[i] = value;
		}
	}else{
		m_values[0] = value;
	}
}

void KernelElement::addToFirstValue(const Real value){
	if(!hasMoreThanOneDim()){
		m_values[0] += value;
	}else{
		printError("This function should not be called if there is more than one length value!");
	}
}


UniquePtr<KernelElement> KernelTypeGenerator::createKernelFor(unsigned int kernelNr){
	UniquePtr<KernelElement> ret;
	ret.reset(nullptr);
	switch(kernelNr){
	case LengthParam:
		ret = std::make_unique<GaussianKernelElementLength>(true); // just to get the type
		break;
	case FNoiseParam:
		ret = std::make_unique<GaussianKernelElementFNoise>();
		break;
	case SNoiseParam:
		ret = std::make_unique<GaussianKernelElementSNoise>();
		break;
	default:
		printError("This type is not defined here!");
		break;
	}
	return ret;
}


GaussianKernelParams::GaussianKernelParams(const GaussianKernelParams& params):
		m_length(params.m_length), m_fNoise(params.m_fNoise), m_sNoise(params.m_sNoise) {
	m_params[0] = &m_length;
	m_params[1] = &m_fNoise;
	m_params[2] = &m_sNoise;
}

GaussianKernelParams& GaussianKernelParams::operator=(const GaussianKernelParams& params){
	const bool hasMoreThanOne = params.m_length.hasMoreThanOneDim();
    m_length.changeAmountOfDims(hasMoreThanOne);
	if(hasMoreThanOne){
		for(unsigned int i = 0; i < ClassKnowledge::amountOfDims(); ++i){
			m_length.getValues()[i] = params.m_length.getValues()[i];
		}
	}else{
		m_length.getValues()[0] = params.m_length.getValues()[0];
	}
	m_fNoise.setAllValuesTo(params.m_fNoise.getValue());
	m_sNoise.setAllValuesTo(params.m_sNoise.getValue());
	return *this;
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
		file.write((char*) m_length.getValues(), size * sizeof(Real));
	}else{
		Real len = m_length.getValue();
		file.write((char*) (&len), sizeof(Real));
	}
	Real fNoise = m_fNoise.getValue();
	file.write((char*) (&fNoise), sizeof(Real));
	Real sNoise = m_sNoise.getValue();
	file.write((char*) (&sNoise), sizeof(Real));
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
		file.read((char*) m_length.getValues(), size * sizeof(Real));
	}else{
		Real len = 0;
		file.read((char*) (&len), sizeof(Real));
		m_length.setAllValuesTo(len);
	}
	Real fNoise;
	file.read((char*) (&fNoise), sizeof(Real));
	m_fNoise.setAllValuesTo(fNoise);
	Real sNoise;
	file.read((char*) (&sNoise), sizeof(Real));
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

RandomForestKernelParams::RandomForestKernelParams(const OwnKernelInitParams& initParams){
	m_samplingAmount.setAllValuesTo(initParams.m_samplingAmount);
	m_maxDepth.setAllValuesTo(initParams.m_maxDepth);
	m_classAmount.setAllValuesTo(initParams.m_amountOfUsedClasses);
	m_params[0] = &m_samplingAmount;
	m_params[1] = &m_maxDepth;
	m_params[2] = &m_classAmount;
}

RandomForestKernelParams::RandomForestKernelParams(const RandomForestKernelParams& params){
	m_samplingAmount.setAllValuesTo(params.m_samplingAmount.getValue());
	m_maxDepth.setAllValuesTo(params.m_maxDepth.getValue());
	m_classAmount.setAllValuesTo(params.m_classAmount.getValue());
	m_params[0] = &m_samplingAmount;
	m_params[1] = &m_maxDepth;
	m_params[2] = &m_classAmount;
}

RandomForestKernelParams& RandomForestKernelParams::operator=(const RandomForestKernelParams& params){
	m_samplingAmount.setAllValuesTo(params.m_samplingAmount.getValue());
	m_maxDepth.setAllValuesTo(params.m_maxDepth.getValue());
	m_classAmount.setAllValuesTo(params.m_classAmount.getValue());
	return *this;
}

void RandomForestKernelParams::setAllValuesTo(const Real value){
	for(unsigned int i = 0; i < paramsAmount; ++i){
		m_params[i]->setAllValuesTo(value);
	}
}

std::ostream& operator<<(std::ostream& stream, const RandomForestKernelParams& params){
	stream << "sampling amount: " << (int) params.m_samplingAmount.getValue() << ", max depth: " << (int) params.m_maxDepth.getValue();
	return stream;
}


