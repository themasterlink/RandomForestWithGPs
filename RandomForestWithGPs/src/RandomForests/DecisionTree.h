/*
 * DecisionTree.h
 *
 *  Created on: 31.05.2016
 *      Author: Max
 */

#ifndef RANDOMFORESTS_DECISIONTREE_H_
#define RANDOMFORESTS_DECISIONTREE_H_

#include "Data.h"

class DecisionTree {
public:
	DecisionTree(const int maxDepth);

	virtual ~DecisionTree();

	void train(const Data& data, const Labels& labels, const int amountOfUsedDims, const Eigen::Vector2i minMaxUsedData);


private:
	// max depth allowed in this tree
	const int m_maxDepth;
	// max number of nodes possible in this tree
	const int m_maxNodeNr; // = pow(2, m_maxDepth +1) - 1
	// max number of nodes, which have children
	const int m_maxInternalNodeNr; // = pow(2, m_maxDepth) - 1
	// contains the split values for the nodes:
	// the order of the nodes is like that:
	// !!!! first element is not used !!!!
	// 			1
	// 		2		3
	//  4	   5  6 	7
	// 8 9 	10 11 12 13  14 15
	std::vector<double> m_splitValues;

};

#endif /* RANDOMFORESTS_DECISIONTREE_H_ */
