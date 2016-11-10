/*
 * KernelType.h
 *
 *  Created on: 31.10.2016
 *      Author: Max
 */

#ifndef GAUSSIANPROCESS_KERNEL_KERNELTYPE_H_
#define GAUSSIANPROCESS_KERNEL_KERNELTYPE_H_

#include "../../Data/ClassKnowledge.h"

class KernelElement {
public:
	KernelElement(unsigned int kernelNr);

	KernelElement(const KernelElement& ele);

	virtual ~KernelElement();

	virtual bool hasMoreThanOneDim() const = 0;

	virtual bool isDerivativeOnlyDiag() const = 0;

	unsigned int getKernelNr() const{ return m_kernelNr; };

	bool operator==(KernelElement& type){ return type.getKernelNr() == getKernelNr(); };

	double* getValues(){ return m_values; };

	const double* getValues() const{ return m_values; };

	double getSquaredValue() const{ return getValue() * getValue(); };

	double getSquaredInverseValue() const{ return 1.0 / getSquaredValue(); };

	void setAllValuesTo(const double value);

	void addToFirstValue(const double value);

	const double getValue() const { return m_values[0]; };

	void changeAmountOfDims(const bool hasMoreThanOneDim);

protected:
	unsigned int m_kernelNr;

	double* m_values;

	bool m_hasMoreThanOneDim;
};

#define LengthParam 0
#define FNoiseParam 1
#define SNoiseParam 2

class GaussianKernelElement : public KernelElement { // just for nicer naming
public:
	GaussianKernelElement(unsigned int kernelNr): KernelElement(kernelNr){};
	virtual ~GaussianKernelElement(){};
};

class GaussianKernelElementLength : public GaussianKernelElement {
public:

	GaussianKernelElementLength(bool hasMoreThanOneDim);

	virtual ~GaussianKernelElementLength(){};

	bool hasMoreThanOneDim() const{ return m_hasMoreThanOneDim; };

	bool isDerivativeOnlyDiag() const{ return false; };

};

class GaussianKernelElementFNoise : public GaussianKernelElement {
public:

	GaussianKernelElementFNoise();

	virtual ~GaussianKernelElementFNoise(){m_hasMoreThanOneDim = false;};

	bool hasMoreThanOneDim() const{ return false; };

	bool isDerivativeOnlyDiag() const{ return false; };

};

class GaussianKernelElementSNoise : public GaussianKernelElement {
public:

	GaussianKernelElementSNoise();

	virtual ~GaussianKernelElementSNoise(){m_hasMoreThanOneDim = false;};

	bool hasMoreThanOneDim() const{ return false; };

	bool isDerivativeOnlyDiag() const{ return true; };
};

class GaussianKernelInitParams {
public:
	GaussianKernelInitParams(bool simpleLength): m_simpleLength(simpleLength){};
	bool m_simpleLength;
};

class KernelTypeGenerator {
public:
	static KernelElement* getKernelFor(unsigned int kernelNr);

private:
	KernelTypeGenerator(){};
	~KernelTypeGenerator(){};
};

class GaussianKernelParams {
public:
	typedef GaussianKernelElement OwnKernelElement;
	typedef GaussianKernelInitParams OwnKernelInitParams;

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

	void setAllValuesTo(const double value);

	void writeToFile(const std::string& file);

	void readFromFile(const std::string& file);
};

std::ostream& operator<<(std::ostream& stream, const GaussianKernelParams& params);

#endif /* GAUSSIANPROCESS_KERNEL_KERNELTYPE_H_ */
