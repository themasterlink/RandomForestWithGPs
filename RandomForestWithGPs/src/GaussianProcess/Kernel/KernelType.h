/*
 * KernelType.h
 *
 *  Created on: 31.10.2016
 *      Author: Max
 */

#ifndef GAUSSIANPROCESS_KERNEL_KERNELTYPE_H_
#define GAUSSIANPROCESS_KERNEL_KERNELTYPE_H_

#include "../../Data/ClassKnowledge.h"
#include "../../Utility/Util.h"
#include <vector>

class KernelElement {
public:
	KernelElement(unsigned int kernelNr);

	KernelElement(const KernelElement& ele);

	virtual ~KernelElement();

	virtual bool hasMoreThanOneDim() const = 0;

	virtual bool isDerivativeOnlyDiag() const = 0;

	unsigned int getKernelNr() const{ return m_kernelNr; };

	bool operator==(KernelElement& type){ return type.getKernelNr() == getKernelNr(); };

	Real* getValues(){ return m_values; };

	const Real* getValues() const{ return m_values; };

	Real getSquaredValue() const{ return getValue() * getValue(); };

	Real getSquaredInverseValue() const{ return 1.0 / getSquaredValue(); };

	void setAllValuesTo(const Real value);

	void addToFirstValue(const Real value);

	Real getValue() const { return m_values[0]; };

	void changeAmountOfDims(const bool hasMoreThanOneDim);

protected:
	unsigned int m_kernelNr;

	Real* m_values;

	bool m_hasMoreThanOneDim;
};

#define LengthParam 0
#define FNoiseParam 1
#define SNoiseParam 2
#define MaxDepthParam 3
#define AmountOfUsedClassesParam 4
#define AmountOfSamplingsParam 5

class GaussianKernelElement : public KernelElement { // just for nicer naming
public:
	GaussianKernelElement(unsigned int kernelNr): KernelElement(kernelNr){};
	virtual ~GaussianKernelElement(){};
};

class GaussianKernelElementLength : public GaussianKernelElement {
public:

	GaussianKernelElementLength(bool hasMoreThanOneDim);

	bool hasMoreThanOneDim() const{ return m_hasMoreThanOneDim; };

	bool isDerivativeOnlyDiag() const{ return false; };

};

class GaussianKernelElementFNoise : public GaussianKernelElement {
public:

	GaussianKernelElementFNoise();

	bool hasMoreThanOneDim() const{ return false; };

	bool isDerivativeOnlyDiag() const{ return false; };

};

class GaussianKernelElementSNoise : public GaussianKernelElement {
public:

	GaussianKernelElementSNoise();

	bool hasMoreThanOneDim() const{ return false; };

	bool isDerivativeOnlyDiag() const{ return true; };
};

class RandomForestKernelElement : public KernelElement {
public:
	RandomForestKernelElement(unsigned int kernelNr): KernelElement(kernelNr){};

	virtual ~RandomForestKernelElement(){};
};

class RandomForestKernelElementMaxDepth : public RandomForestKernelElement {
public:
	RandomForestKernelElementMaxDepth(): RandomForestKernelElement(MaxDepthParam){ m_values = new Real; };

	bool hasMoreThanOneDim() const{ return false; };

	bool isDerivativeOnlyDiag() const{ return true; };
};

class RandomForestKernelElementMaxAmountOfClasses : public RandomForestKernelElement {
public:
	RandomForestKernelElementMaxAmountOfClasses(): RandomForestKernelElement(AmountOfUsedClassesParam){ m_values = new Real; };

	bool hasMoreThanOneDim() const{ return false; };

	bool isDerivativeOnlyDiag() const{ return true; };
};

class RandomForestKernelElementAmountOfSamplings : public RandomForestKernelElement {
public:
	RandomForestKernelElementAmountOfSamplings(): RandomForestKernelElement(AmountOfSamplingsParam){ m_values = new Real; };

	bool hasMoreThanOneDim() const{ return false; };

	bool isDerivativeOnlyDiag() const{ return true; };
};

class GaussianKernelInitParams {
public:
	GaussianKernelInitParams(bool simpleLength): m_simpleLength(simpleLength){};
	bool m_simpleLength;
};

class RandomForestKernelInitParams {
public:
	RandomForestKernelInitParams(const int maxDepth, const int samplingAmount, const int amountOfUsedClasses):
		m_maxDepth(maxDepth), m_samplingAmount(samplingAmount), m_amountOfUsedClasses(amountOfUsedClasses){};
	const int m_maxDepth;
	const int m_samplingAmount;
	const int m_amountOfUsedClasses;
};

class KernelTypeGenerator {
public:
	static UniquePtr<KernelElement> createKernelFor(unsigned int kernelNr);

private:
	KernelTypeGenerator(){};
	~KernelTypeGenerator(){};
};

class GaussianKernelParams {
public:
	using OwnKernelElement = GaussianKernelElement;
	using OwnKernelInitParams = GaussianKernelInitParams;

	static const unsigned int paramsAmount = 3;

	static const std::vector<unsigned int> usedParamTypes;

	GaussianKernelParams(const OwnKernelInitParams& initParams);

	GaussianKernelParams(const GaussianKernelParams& params);

	GaussianKernelParams& operator=(const GaussianKernelParams& params);

	GaussianKernelParams(bool simpleLength = true);

	~GaussianKernelParams(){};

	GaussianKernelElementLength m_length;

	GaussianKernelElementFNoise m_fNoise;

	GaussianKernelElementSNoise m_sNoise;

	OwnKernelElement* m_params[paramsAmount];

	void setAllValuesTo(const Real value);

	void writeToFile(const std::string& file);

	void readFromFile(const std::string& file);
};

std::ostream& operator<<(std::ostream& stream, const GaussianKernelParams& params);

class RandomForestKernelParams {
public:
	using OwnKernelElement = RandomForestKernelElement;
	using OwnKernelInitParams = RandomForestKernelInitParams;

	static const unsigned int paramsAmount = 3;

	RandomForestKernelParams(const OwnKernelInitParams& initParams);

	RandomForestKernelParams(const RandomForestKernelParams& params);

	RandomForestKernelParams& operator=(const RandomForestKernelParams& params);

	OwnKernelElement* m_params[paramsAmount];

	RandomForestKernelElementAmountOfSamplings m_samplingAmount;

	RandomForestKernelElementMaxDepth m_maxDepth;

	RandomForestKernelElementMaxAmountOfClasses m_classAmount;

	void setAllValuesTo(const Real value);

	void writeToFile(const std::string& file){UNUSED(file);}; // read and write trees to the file

	void readFromFile(const std::string& file){UNUSED(file);};
};

std::ostream& operator<<(std::ostream& stream, const RandomForestKernelParams& params);


#endif /* GAUSSIANPROCESS_KERNEL_KERNELTYPE_H_ */
