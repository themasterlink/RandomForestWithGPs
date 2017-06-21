/*
 * DynamicDecisionTreeInterface.h
 *
 *  Created on: 07.01.2017
 *      Author: Max
 */

#ifndef RANDOMFORESTS_DYNAMICDECISIONTREEINTERFACE_H_
#define RANDOMFORESTS_DYNAMICDECISIONTREEINTERFACE_H_

#include "../RandomNumberGenerator/RandomNumberGeneratorForDT.h"

class DynamicDecisionTreeInterface : public PredictorMultiClass {
public:
	virtual ~DynamicDecisionTreeInterface() = default;

	virtual void train(unsigned int amountOfUsedDims, RandomNumberGeneratorForDT& generator) = 0;

	virtual bool predictIfPointsShareSameLeaveWithHeight(const VectorX& point1, const VectorX& point2, const int usedHeight) const = 0;

	virtual MemoryType getMemSize() const = 0;
};

#endif /* RANDOMFORESTS_DYNAMICDECISIONTREEINTERFACE_H_ */
