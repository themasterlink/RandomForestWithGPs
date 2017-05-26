/*
 * DynamicDecisionTree.h
 *
 *  Created on: 17.10.2016
 *      Author: Max
 */

#ifndef RANDOMFORESTS_DYNAMICDECISIONTREE_H_
#define RANDOMFORESTS_DYNAMICDECISIONTREE_H_

#include "../Data/OnlineStorage.h"
#include "../Data/LabeledVectorX.h"
#include "../RandomNumberGenerator/RandomNumberGeneratorForDT.h"
#include "../RandomForests/DecisionTreeData.h"
#include "../Base/Predictor.h"
#include "DynamicDecisionTreeInterface.h"

class ReadWriterHelper;

class DynamicDecisionTree : public DynamicDecisionTreeInterface {

friend ReadWriterHelper;

public:
	enum NodeType : int { // saved in m_splitDim
		NODE_IS_NOT_USED = -1,
		NODE_CAN_BE_USED = -2,
	};

	DynamicDecisionTree(OnlineStorage<LabeledVectorX *> &storage, const unsigned int maxDepth,
							const unsigned int amountOfClasses, const unsigned int amountOfPointsPerSplit);

	// construct empty tree
	DynamicDecisionTree(OnlineStorage<LabeledVectorX*>& storage);

	// copy construct
	DynamicDecisionTree(const DynamicDecisionTree& tree);

	virtual ~DynamicDecisionTree();

	void train(unsigned int amountOfUsedDims, RandomNumberGeneratorForDT& generator){
		train(amountOfUsedDims, generator, 0, false);
	}

	bool train(unsigned int amountOfUsedDims, RandomNumberGeneratorForDT& generator, const unsigned int tryCounter, const bool saveDataPosition);

	Real trySplitFor(const Real usedSplitValue, const unsigned int usedDim,
			const std::vector<unsigned int>& dataInNode, std::vector<unsigned int>& leftHisto,
			std::vector<unsigned int>& rightHisto, RandomNumberGeneratorForDT& generator);

	void adjustToNewData();

	unsigned int predict(const VectorX& point) const;

	unsigned int predict(const VectorX& point, int& winningLeafNode) const;

	bool predictIfPointsShareSameLeaveWithHeight(const VectorX& point1, const VectorX& point2, const int usedHeight) const;

	void predictData(const Data& data, Labels& labels) const{
		UNUSED(data); UNUSED(labels);
		printError("This function is not implemented!");
	}

	void predictData(const Data& points, Labels& labels, std::vector< std::vector<Real> >& probabilities) const{
		UNUSED(points); UNUSED(labels); UNUSED(probabilities);
		printError("Not implemented yet!");
	}

	unsigned int getNrOfLeaves();

	unsigned int amountOfClasses() const;

	std::vector<std::vector<unsigned int> >* getDataPositions(){ return m_dataPositions; };

	void deleteDataPositions();

	void setUsedDataPositions(std::vector<unsigned int>* usedDataPositions){ m_useOnlyThisDataPositions = usedDataPositions; };

	MemoryType getMemSize() const;

//	void printStream(std::ostream &output, const Real precision = 3);

private:
	// this function is only called if the empty tree constructor was used!
	void prepareForSetting(const unsigned int maxDepth, const unsigned int amountOfClasses);

	OnlineStorage<LabeledVectorX*>& m_storage;
	// max depth allowed in this tree
	const unsigned int m_maxDepth;
	// max number of nodes possible in this tree
	const unsigned int m_maxNodeNr; // = pow2(m_maxDepth +1) - 1
	// max number of nodes, which have children
	const unsigned int m_maxInternalNodeNr; // = pow2(m_maxDepth) - 1

	const unsigned int m_amountOfClasses;

	const std::vector<unsigned int>::size_type m_amountOfPointsCheckedPerSplit;
	// contains the split values for the nodes:
	// the order of the nodes is like that:
	// !!!! first element is not used !!!!
	// 			1
	// 		2		3
	//  4	   5  6 	7
	// 8 9 	10 11 12 13  14 15
	std::vector<Real> m_splitValues;
	// order is like with split values
	std::vector<int> m_splitDim;

	std::vector<unsigned int> m_labelsOfWinningClassesInLeaves;

	// is used in the BigDynamicDecisionTree
	// fill contain the data for each node ->
	// 	in the end only the last pow2(m_maxDepth) will have values, the rest will be empty
	std::vector<std::vector<unsigned int> >* m_dataPositions;

	std::vector<unsigned int>* m_useOnlyThisDataPositions;

};

#endif /* RANDOMFORESTS_DYNAMICDECISIONTREE_H_ */
