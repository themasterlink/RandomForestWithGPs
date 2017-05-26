/*
 * OtherDecisionTree.h
 *
 *  Created on: 03.06.2016
 *      Author: Max
 */

#ifndef OTHERRANDOMFORESTS_OTHERDECISIONTREE_H_
#define OTHERRANDOMFORESTS_OTHERDECISIONTREE_H_

#include "../Data/LabeledVectorX.h"
#include "../RandomNumberGenerator/RandomNumberGeneratorForDT.h"
#include "DecisionTreeData.h"

class DecisionTree{
public:
	enum NodeType { // saved in m_splitDim, no class used, because the value should be interchangeable with -1 and -2
		NODE_IS_NOT_USED = -1,
		NODE_CAN_BE_USED = -2,
	};

	DecisionTree(const int maxDepth, const int amountOfClasses);

	// copy construct
	DecisionTree(const DecisionTree& tree);

	virtual ~DecisionTree();

	void train(const LabeledData& data, const int amountOfUsedDims,
			RandomNumberGeneratorForDT& generator);

	double trySplitFor(const int usedNode, const int usedDim, const LabeledData& data,
			const std::vector<int>& dataInNode, std::vector<int>& leftHisto,
			std::vector<int>& rightHisto, RandomNumberGeneratorForDT& generator);

	int predict(const VectorX& point) const;

	void writeToData(DecisionTreeData& data) const;

	void initFromData(const DecisionTreeData& data);

	DecisionTree& operator=(const DecisionTree& tree);

	int getNrOfLeaves(){return pow2(m_maxDepth);};

private:
	// max depth allowed in this tree
	const int m_maxDepth;
	// max number of nodes possible in this tree
	const int m_maxNodeNr; // = pow2(m_maxDepth +1) - 1
	// max number of nodes, which have children
	const int m_maxInternalNodeNr; // = pow2(m_maxDepth) - 1

	const int m_amountOfClasses;
	// contains the split values for the nodes:
	// the order of the nodes is like that:
	// !!!! first element is not used !!!!
	// 			1
	// 		2		3
	//  4	   5  6 	7
	// 8 9 	10 11 12 13  14 15
	std::vector<real> m_splitValues;
	// order is like with split values
	std::vector<int> m_splitDim;

	std::vector<int> m_labelsOfWinningClassesInLeaves;

};

#endif /* OTHERRANDOMFORESTS_OTHERDECISIONTREE_H_ */
