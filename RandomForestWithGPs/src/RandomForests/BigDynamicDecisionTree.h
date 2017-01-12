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
	BigDynamicDecisionTree(OnlineStorage<ClassPoint*>& storage, const int maxDepth, const int amountOfClasses, const int layerAmount = -1, const int layerAmountForFast = -1);

	virtual ~BigDynamicDecisionTree();

	void train(const unsigned int amountOfUsedDims, RandomNumberGeneratorForDT& generator);

	unsigned int predict(const DataPoint& point) const;

	bool predictIfPointsShareSameLeaveWithHeight(const DataPoint& point1, const DataPoint& point2, const int usedHeight) const{
		UNUSED(point1); UNUSED(point2); UNUSED(usedHeight);
		printError("This function is not implemented!");
		return false;
	}

	void predictData(const Data& data, Labels& labels) const{
		UNUSED(data); UNUSED(labels);
		printError("This function is not implemented!");
	}

	void predictData(const Data& points, Labels& labels, std::vector< std::vector<double> >& probabilities) const{
		UNUSED(points); UNUSED(labels); UNUSED(probabilities);
		printError("Not implemented yet!");
	}

	unsigned int amountOfClasses() const;

private:

	typedef std::map<unsigned int, DynamicDecisionTree*> SmallTreeInnerStructure;
	typedef std::pair<unsigned int, DynamicDecisionTree*> SmallTreeInnerPair;
	typedef std::vector<SmallTreeInnerStructure> SmallTreeStructure;
	typedef std::vector<DynamicDecisionTree*> FastTreeInnerStructure;
	typedef std::vector<FastTreeInnerStructure> FastTreeStructure;


	void trainChildrenForRoot(DynamicDecisionTree* root, SmallTreeInnerStructure::iterator& it, SmallTreeInnerStructure& actSmallInnerTreeStructure,
			const unsigned int depthInThisLayer, const unsigned int iRootId,
			const unsigned int leavesForTreesInTheFatherLayer, const unsigned int neededPointsForNewTree,
			const int amountOfUsedDims, RandomNumberGeneratorForDT& generator, const bool saveDataPositions, bool& foundAtLeastOneChild);

	OnlineStorage<ClassPoint*>& m_storage;
	// max depth allowed in this tree
	const int m_maxDepth;

	const int m_amountOfClasses;

	int m_depthPerLayer;

	FastTreeStructure m_fastInnerTrees;
	SmallTreeStructure m_smallInnerTrees;
};

#endif /* RANDOMFORESTS_BIGDYNAMICDECISIONTREE_H_ */
