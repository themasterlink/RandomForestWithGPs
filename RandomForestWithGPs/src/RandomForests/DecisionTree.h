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

	void train(const Data& data, const Labels& labels);


private:
	// max depth allowed in this tree
	const int m_maxDepth;
	// max number of nodes possible in this tree
	const int m_maxNodeNr; // = pow(2, m_maxDepth +1) - 1
	// max number of nodes, which have children
	const int m_maxInternalNodeNr; // = pow(2, m_maxDepth) - 1


};

#endif /* RANDOMFORESTS_DECISIONTREE_H_ */
