/*
 * DecisionTreeData.h
 *
 *  Created on: 06.06.2016
 *      Author: Max
 */

#ifndef RANDOMFORESTS_DECISIONTREEDATA_H_
#define RANDOMFORESTS_DECISIONTREEDATA_H_


struct DecisionTreeData{
	int height;
	int nrOfInternalNodes; // size of dim and split values
	int nrOfLeaves; // is equal to number of (internal nodes in tree + 1)
	int amountOfClasses;
	std::vector<Real> splitValues; // array of the splitValues
	std::vector<int> dimValues; // array of the dimension used for the split, if value is -1 -> not used!
	std::vector<int> labelsOfWinningClassInLeaves;
};

#endif /* RANDOMFORESTS_DECISIONTREEDATA_H_ */
