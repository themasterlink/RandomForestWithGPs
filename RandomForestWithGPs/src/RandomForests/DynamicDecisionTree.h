/*
 * DynamicDecisionTree.h
 *
 *  Created on: 17.10.2016
 *      Author: Max
 */

#ifndef RANDOMFORESTS_DYNAMICDECISIONTREE_H_
#define RANDOMFORESTS_DYNAMICDECISIONTREE_H_

#include "../Data/OnlineStorage.h"
#include "../Data/ClassPoint.h"
#include "../RandomNumberGenerator/RandomNumberGeneratorForDT.h"
#include "../RandomForests/DecisionTreeData.h"

class DynamicDecisionTree {
public:
	enum NodeType{ // saved in m_splitDim
		NODE_IS_NOT_USED = -1,
		NODE_CAN_BE_USED = -2,
	};

	DynamicDecisionTree(OnlineStorage<ClassPoint*>& storage, const int maxDepth, const int amountOfClasses);

	// copy construct
	DynamicDecisionTree(const DynamicDecisionTree& tree);

	virtual ~DynamicDecisionTree();

	void train(const int amountOfUsedDims, RandomNumberGeneratorForDT& generator);

	double trySplitFor(const int actNode, const int usedNode, const int usedDim,
			const std::vector<int>& dataInNode, std::vector<int>& leftHisto,
			std::vector<int>& rightHisto, RandomNumberGeneratorForDT& generator);

	void adjustToNewData();

	int predict(const DataPoint& point) const;

	int getNrOfLeaves();

private:
	OnlineStorage<ClassPoint*>& m_storage;
	// max depth allowed in this tree
	const int m_maxDepth;
	// max number of nodes possible in this tree
	const int m_maxNodeNr; // = pow(2, m_maxDepth +1) - 1
	// max number of nodes, which have children
	const int m_maxInternalNodeNr; // = pow(2, m_maxDepth) - 1

	const int m_amountOfClasses;
	// contains the split values for the nodes:
	// the order of the nodes is like that:
	// !!!! first element is not used !!!!
	// 			1
	// 		2		3
	//  4	   5  6 	7
	// 8 9 	10 11 12 13  14 15
	std::vector<double> m_splitValues;
	// order is like with split values
	std::vector<int> m_splitDim;

	std::vector<int> m_labelsOfWinningClassesInLeaves;
};

#endif /* RANDOMFORESTS_DYNAMICDECISIONTREE_H_ */
