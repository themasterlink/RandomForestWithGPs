/*
 * BigDynamicDecisionTree.h
 *
 *  Created on: 07.01.2017
 *      Author: Max
 */

#ifndef RANDOMFORESTS_BIGDYNAMICDECISIONTREE_H_
#define RANDOMFORESTS_BIGDYNAMICDECISIONTREE_H_

#include "DynamicDecisionTree.h"

class BigDynamicDecisionTree : public DynamicDecisionTreeInterface {
public:
	BigDynamicDecisionTree(OnlineStorage<ClassPoint*>& storage, const int maxDepth, const int amountOfClasses);

	void train(int amountOfUsedDims, RandomNumberGeneratorForDT& generator);

	int predict(const DataPoint& point) const;

	bool predictIfPointsShareSameLeaveWithHeight(const DataPoint& point1, const DataPoint& point2, const int usedHeight) const{
		printError("This function is not implemented!");
		return false;
	}

	void predictData(const Data& data, Labels& labels) const{
		printError("This function is not implemented!");
	}

	void predictData(const Data& points, Labels& labels, std::vector< std::vector<double> >& probabilities) const{
		printError("Not implemented yet!");
	}

	unsigned int amountOfClasses() const;

	virtual ~BigDynamicDecisionTree();

private:
	OnlineStorage<ClassPoint*>& m_storage;
	// max depth allowed in this tree
	const int m_maxDepth;

	const int m_amountOfClasses;

	int m_depthPerLayer;

	typedef std::vector<DynamicDecisionTree*> TreeInnerStructure;
	typedef std::vector<TreeInnerStructure> TreeStructure;

	TreeStructure m_innerTrees;
};

#endif /* RANDOMFORESTS_BIGDYNAMICDECISIONTREE_H_ */
