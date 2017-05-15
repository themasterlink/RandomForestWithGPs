/*
 * BigDynamicDecisionTree.h
 *
 *  Created on: 07.01.2017
 *      Author: Max
 */

#ifndef RANDOMFORESTS_BIGDYNAMICDECISIONTREE_H_
#define RANDOMFORESTS_BIGDYNAMICDECISIONTREE_H_

#include "DynamicDecisionTree.h"

class ReadWriterHelper;

class BigDynamicDecisionTree : public DynamicDecisionTreeInterface {

friend ReadWriterHelper;

public:
	BigDynamicDecisionTree(OnlineStorage<ClassPoint*>& storage, const unsigned int maxDepth, const unsigned int amountOfClasses, const int layerAmount = -1, const int layerAmountForFast = -1);

	BigDynamicDecisionTree(OnlineStorage<ClassPoint*>& storage);

	void prepareForSetting(const unsigned int maxDepth, const unsigned int amountOfClasses, const unsigned int amountOfLayers, const unsigned int amountForFast, const unsigned int amountForSmall);

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

	MemoryType getMemSize() const { return m_usedMemory; };

private:

	using SmallTreeInnerStructure = std::map<unsigned int, DynamicDecisionTree*>;
	using SmallTreeInnerPair = std::pair<unsigned int, DynamicDecisionTree*>;
	using SmallTreeStructure = std::vector<SmallTreeInnerStructure>;
	using FastTreeInnerStructure = std::vector<DynamicDecisionTree*>;
	using FastTreeStructure = std::vector<FastTreeInnerStructure>;

	bool shouldNewTreeBeCalculatedFor(std::vector<unsigned int>& dataPositions);

	void trainChildrenForRoot(DynamicDecisionTree* root, SmallTreeInnerStructure::iterator& it, SmallTreeInnerStructure& actSmallInnerTreeStructure,
			const unsigned int depthInThisLayer, const unsigned int iRootId,
			const unsigned int leavesForTreesInTheFatherLayer, const int amountOfUsedDims,
			RandomNumberGeneratorForDT& generator, const bool saveDataPositions, bool& foundAtLeastOneChild);

	OnlineStorage<ClassPoint*>& m_storage;
	// max depth allowed in this tree
	const unsigned int m_maxDepth;

	const unsigned int m_amountOfClasses;

	unsigned int m_depthPerLayer;

	MemoryType m_usedMemory;

	FastTreeStructure m_fastInnerTrees;
	SmallTreeStructure m_smallInnerTrees;
};

#endif /* RANDOMFORESTS_BIGDYNAMICDECISIONTREE_H_ */
