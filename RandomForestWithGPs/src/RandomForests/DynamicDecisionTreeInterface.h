/*
 * DynamicDecisionTreeInterface.h
 *
 *  Created on: 07.01.2017
 *      Author: Max
 */

#ifndef RANDOMFORESTS_DYNAMICDECISIONTREEINTERFACE_H_
#define RANDOMFORESTS_DYNAMICDECISIONTREEINTERFACE_H_

#include "../RandomNumberGenerator/RandomNumberGeneratorForDT.h"

/**
 * \brief The interface used in the Online Random Forest, these function are provided by each Decision Tree implementation
 */
class DynamicDecisionTreeInterface : public PredictorMultiClass {
public:
	virtual ~DynamicDecisionTreeInterface() = default; // must be virtual

	/**
	 * \brief Train the tree
	 * \param amountOfUsedDims amount of use dims, which are used in the training
	 * \param generator the generator to generate the random numbers
	 */
	virtual void train(unsigned int amountOfUsedDims, RandomNumberGeneratorForDT& generator) = 0;

	/**
	 * \brief Predict if two points share the same leaf at a certain height
	 * \param point1 first point to compare
	 * \param point2 second point to compare
	 * \param usedHeight at which both points have to be in the same split
	 * \return if these two points share same node at the given height
	 */
	virtual bool predictIfPointsShareSameLeafWithHeight(const VectorX& point1, const VectorX& point2,
														const int usedHeight) const = 0;

	/**
	 * \brief Return the memory size of the tree
	 * \return the memory size of the tree
	 */
	virtual MemoryType getMemSize() const = 0;
};

#endif /* RANDOMFORESTS_DYNAMICDECISIONTREEINTERFACE_H_ */
