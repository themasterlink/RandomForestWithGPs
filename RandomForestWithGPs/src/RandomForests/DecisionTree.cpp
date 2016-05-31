/*
 * DecisionTree.cpp
 *
 *  Created on: 31.05.2016
 *      Author: Max
 */

#include "DecisionTree.h"
#include <cmath>
#include "../Utility/Util.h"

DecisionTree::DecisionTree(const int maxDepth) :
		m_maxDepth(maxDepth), m_maxNodeNr(pow(2, maxDepth + 1) - 1),
		m_maxInternalNodeNr(pow(2, maxDepth) - 1) {
}

DecisionTree::~DecisionTree() {
	// TODO Auto-generated destructor stub
}

void DecisionTree::train(const Data& data, const Labels& labels){
	if(data.size() != labels.size()){
		printError("Label and data size are not equal!");
		return;
	}


}
