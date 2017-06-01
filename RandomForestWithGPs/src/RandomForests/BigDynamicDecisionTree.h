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
	BigDynamicDecisionTree(OnlineStorage<LabeledVectorX*>& storage, const unsigned int maxDepth,
						   const unsigned int amountOfClasses, const int layerAmount = -1,
						   const int layerAmountForFast = -1, const unsigned int amountOfPointsCheckedPerSplit = 100);

	BigDynamicDecisionTree(OnlineStorage<LabeledVectorX*>& storage);

	void prepareForSetting(const unsigned int maxDepth, const unsigned int amountOfClasses,
						   const unsigned int amountOfLayers, const unsigned int amountForFast,
						   const unsigned int amountForSmall);

	virtual ~BigDynamicDecisionTree();

	void train(const unsigned int amountOfUsedDims, RandomNumberGeneratorForDT& generator);

	unsigned int predict(const VectorX& point) const;

	bool predictIfPointsShareSameLeaveWithHeight(const VectorX& point1, const VectorX& point2,
												 const int usedHeight) const override {
		UNUSED(point1); UNUSED(point2); UNUSED(usedHeight);
		printErrorAndQuit("This function is not implemented!");
		return false;
	}

	void predictData(const Data& data, Labels& labels) const override {
		UNUSED(data); UNUSED(labels);
		printErrorAndQuit("This function is not implemented!");
	}

	void predictData(const Data& points, Labels& labels, std::vector< std::vector<Real> >& probabilities) const override {
		UNUSED(points); UNUSED(labels); UNUSED(probabilities);
		printErrorAndQuit("Not implemented yet!");
	}

	unsigned int amountOfClasses() const;

	MemoryType getMemSize() const override { return m_usedMemory; };

private:

	using PtrDynamicDecisionTree = DynamicDecisionTree<dimTypeForDDT>*;
	using SmallTreeInnerStructure = std::map<unsigned int, PtrDynamicDecisionTree>;
	using SmallTreeInnerStructureIterator = SmallTreeInnerStructure::iterator;
	using SmallTreeInnerPair = std::pair<unsigned int, PtrDynamicDecisionTree>;
	using SmallTreeStructure = std::vector<SmallTreeInnerStructure>;
	using FastTreeInnerStructure = std::vector<PtrDynamicDecisionTree>;
	using FastTreeStructure = std::vector<FastTreeInnerStructure>;

	bool shouldNewTreeBeCalculatedFor(std::vector<unsigned int>& dataPositions);

	void trainChildrenForRoot(PtrDynamicDecisionTree root, SmallTreeInnerStructureIterator& it,
							  SmallTreeInnerStructure& actSmallInnerTreeStructure,
			const unsigned int depthInThisLayer, const unsigned int iRootId,
			const unsigned int leavesForTreesInTheFatherLayer, const int amountOfUsedDims,
			RandomNumberGeneratorForDT& generator, const bool saveDataPositions, bool& foundAtLeastOneChild);

	OnlineStorage<LabeledVectorX*>& m_storage;
	// max depth allowed in this tree
	const unsigned int m_maxDepth;

	const unsigned int m_amountOfClasses;

	const unsigned int m_amountOfPointsCheckedPerSplit;

	unsigned int m_depthPerLayer;

	MemoryType m_usedMemory;

	FastTreeStructure m_fastInnerTrees;
	SmallTreeStructure m_smallInnerTrees;
};

#endif /* RANDOMFORESTS_BIGDYNAMICDECISIONTREE_H_ */
