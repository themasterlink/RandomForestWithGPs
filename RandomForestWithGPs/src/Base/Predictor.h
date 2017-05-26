/*
 * PredictorBinaryClass.h
 *
 *  Created on: 19.10.2016
 *      Author: Max
 */

#ifndef BASE_PREDICTOR_H_
#define BASE_PREDICTOR_H_

#include "Types.h"

class PredictorBinaryClass {
public:
	virtual ~PredictorBinaryClass(){};

	virtual real predict(const VectorX& point) const = 0;
};

class PredictorMultiClass {
public:
	virtual ~PredictorMultiClass(){};

	virtual unsigned int predict(const VectorX& point) const = 0;

	virtual void predictData(const Data& points, Labels& labels) const = 0;

	virtual void predictData(const Data& points, Labels& labels, std::vector< std::vector<real> >& probabilities) const = 0;

	virtual unsigned int amountOfClasses() const = 0;
};

#endif /* BASE_PREDICTOR_H_ */
