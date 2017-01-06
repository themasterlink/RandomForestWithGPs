/*
 * PredictorBinaryClass.h
 *
 *  Created on: 19.10.2016
 *      Author: Max
 */

#ifndef BASE_PREDICTOR_H_
#define BASE_PREDICTOR_H_

#include "../Data/Data.h"

class PredictorBinaryClass {
public:
	virtual ~PredictorBinaryClass(){};

	virtual double predict(const DataPoint& point) const = 0;
};

class PredictorMultiClass {
public:
	virtual ~PredictorMultiClass(){};

	virtual int predict(const DataPoint& point) const = 0;

	virtual void predictData(const Data& points, Labels& labels) const = 0;

	virtual void predictData(const Data& points, Labels& labels, std::vector< std::vector<double> >& probabilities) const = 0;

	virtual unsigned int amountOfClasses() const = 0;
};

#endif /* BASE_PREDICTOR_H_ */
