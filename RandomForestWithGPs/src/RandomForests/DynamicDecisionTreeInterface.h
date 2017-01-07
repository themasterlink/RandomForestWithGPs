/*
 * DynamicDecisionTreeInterface.h
 *
 *  Created on: 07.01.2017
 *      Author: Max
 */

#ifndef RANDOMFORESTS_DYNAMICDECISIONTREEINTERFACE_H_
#define RANDOMFORESTS_DYNAMICDECISIONTREEINTERFACE_H_

#include "../RandomNumberGenerator/RandomNumberGeneratorForDT.h"
#include "../Data/Data.h"

class DynamicDecisionTreeInterface : public PredictorMultiClass {
public:
	virtual ~DynamicDecisionTreeInterface(){};

	virtual void train(int amountOfUsedDims, RandomNumberGeneratorForDT& generator) = 0;

	virtual bool predictIfPointsShareSameLeaveWithHeight(const DataPoint& point1, const DataPoint& point2, const int usedHeight) const = 0;
};

#endif /* RANDOMFORESTS_DYNAMICDECISIONTREEINTERFACE_H_ */
