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
	BigDynamicDecisionTree(OnlineStorage<ClassPoint*>& storage, const int maxDepth, const int amountOfClasses, const int layerAmount = -1);

	virtual ~BigDynamicDecisionTree();

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

private:
	OnlineStorage<ClassPoint*>& m_storage;
	// max depth allowed in this tree
	const int m_maxDepth;

	const int m_amountOfClasses;

	int m_depthPerLayer;

	typedef std::map<unsigned int, DynamicDecisionTree*> SmallTreeInnerStructure;
	typedef std::pair<unsigned int, DynamicDecisionTree*> SmallTreeInnerPair;
	typedef std::vector<SmallTreeInnerStructure> SmallTreeStructure;
	typedef std::vector<DynamicDecisionTree*> FastTreeInnerStructure;
	typedef std::vector<FastTreeInnerStructure> FastTreeStructure;

	FastTreeStructure m_fastInnerTrees;
	SmallTreeStructure m_smallInnerTrees;
};

#endif /* RANDOMFORESTS_BIGDYNAMICDECISIONTREE_H_ */
