/*
 * OtherDecisionTree.h
 *
 *  Created on: 03.06.2016
 *      Author: Max
 */

#ifndef OTHERRANDOMFORESTS_OTHERDECISIONTREE_H_
#define OTHERRANDOMFORESTS_OTHERDECISIONTREE_H_

#include "../Data/Data.h"
#include "../RandomNumberGenerator/RandomNumberGeneratorForDT.h"

class OtherDecisionTree{
public:
	OtherDecisionTree(const int maxDepth, const int amountOfClasses);

	virtual ~OtherDecisionTree();

	void train(const Data& data, const Labels& labels, const int amountOfUsedDims,
			RandomNumberGeneratorForDT& generator);

	double trySplitFor(const int actNode, const int usedNode, const int usedDim, const Data& data,
			const Labels& labels, const std::vector<int>& dataInNode, std::vector<int>& leftHisto,
			std::vector<int>& rightHisto, RandomNumberGeneratorForDT& generator);

	int predict(const DataElement& point) const;

private:
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
	// order is like with split values, is this node used -> (does it contain any values)
	std::vector<bool> m_isUsed;

	std::vector<int> m_labelsOfWinningClassesInLeaves;

};

#endif /* OTHERRANDOMFORESTS_OTHERDECISIONTREE_H_ */
