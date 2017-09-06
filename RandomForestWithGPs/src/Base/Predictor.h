/*
 * PredictorBinaryClass.h
 *
 *  Created on: 19.10.2016
 *      Author: Max
 */

#ifndef BASE_PREDICTOR_H_
#define BASE_PREDICTOR_H_

#include "Types.h"

/**
 * \brief
 */
class PredictorBinaryClass {
public:
	virtual ~PredictorBinaryClass(){};

	virtual Real predict(const VectorX& point) const = 0;
};

/**
 * \brief Predictor Multi Class pattern, defines the standard predict functions, for one point and a data set
 */
class PredictorMultiClass {
public:
	virtual ~PredictorMultiClass() = default;

	/**
	 * \brief Predict the class for a given point
	 * \param point for which the class is estimated
	 * \return the predicted class of the given point
	 */
	virtual unsigned int predict(const VectorX& point) const = 0;

	/**
	 * \brief Predict the classes for all points and save it in labels
	 * 		Labels is resized and should be empty at the beginning
	 * \param points which are used to predict the classes
	 * \param labels where the predicted labels are saved
	 */
	virtual void predictData(const Data& points, Labels& labels) const = 0;

	/**
	 * \brief Predict the classes for all points and save it in labels
	 * 		Labels is resized and should be empty at the beginning
	 * 		The probabilities are saved as followed:
	 * 			Each point gets its own vector with the length of amount of classes, in each the probability is saved
	 * 			for each class between the 0.0 and 1.0
	 * \param points which are used to predict the classes
	 * \param labels where the predicted labels are saved
	 * \param probabilities where the probabilities of each point are saved
	 */
	virtual void predictData(const Data& points, Labels& labels, std::vector< std::vector<Real> >& probabilities) const = 0;

	/**
	 * \brief returns the amount of classes, is necessary for the predictData(const Data& points, Labels& labels,
	 * std::vector< std::vector<Real> >& probabilities) function
	 * \return the amount of the classes
	 */
	virtual unsigned int amountOfClasses() const = 0;
};

#endif /* BASE_PREDICTOR_H_ */
