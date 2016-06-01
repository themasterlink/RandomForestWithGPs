/*
 * DecisionTree.cpp
 *
 *  Created on: 31.05.2016
 *      Author: Max
 */

#include "DecisionTree.h"
#include <cmath>
#include <random>
#include <algorithm>
#include <limits>

#define MIN_NR_TO_SPLIT 2

DecisionTree::DecisionTree(const int maxDepth, const int amountOfClasses) :
m_maxDepth(maxDepth), m_maxNodeNr(pow(2, maxDepth + 1) - 1),
m_maxInternalNodeNr(pow(2, maxDepth) - 1),
m_amountOfClasses(amountOfClasses),
m_splitValues(m_maxInternalNodeNr + 1),  // + 1 -> no use of the first element
m_splitDim(m_maxInternalNodeNr + 1),
m_isUsed(m_maxNodeNr + 1, false),
m_labelsOfWinningClassesInLeaves(pow(2, maxDepth)){
}

DecisionTree::~DecisionTree() {
}

void DecisionTree::train(const Data& data, const Labels& labels, const int amountOfUsedDims, const Eigen::Vector2i minMaxUsedData){
	if(data.size() != labels.size()){
		printError("Label and data size are not equal!"); return;
	}else if(data.size() < 2){
		printError("There must be at least two points!"); return;
	}else if(data[0].rows() < 2){
		printError("There should be at least 2 dimensions in the data");
	}else if(amountOfUsedDims > data[0].rows()){
		printError("Amount of dims can't be bigger than the dimension size!"); return;
	}

	// order is the same as in data, value specifies the node in which it is saved at the moment
	std::vector<int> nodesContent = std::vector<int>(data.size(), 1); // 1 = root node

	std::default_random_engine generator;
	std::uniform_int_distribution<int> uniformDist_Dimension(0,data[0].rows() - 1); // 0 ... (dimension of data - 1)
	std::uniform_int_distribution<int> uniformDist_usedData(minMaxUsedData[0], minMaxUsedData[1]); // TODO add check!
	std::uniform_int_distribution<int> uniformDist_Data(0, data.size() - 1); // 0 ... (dimension of data - 1)

	std::vector<int> usedDims(amountOfUsedDims);
	if(amountOfUsedDims == m_amountOfClasses){
		for(int i = 0; i < amountOfUsedDims; ++i){
			usedDims[i] = i;
		}
	}else{
		for(int i = 0; i < amountOfUsedDims; ++i){
			bool doAgain = false;
			do{
				const int randNr = uniformDist_Data(generator);  // generates number in the range 0...data.rows() - 1;
				for(int j = 0; j < i; ++j){
					if(randNr == usedDims[j]){
						doAgain = true; break;
					}
				}
				if(!doAgain){
					usedDims[i] = randNr;
				}
			}while(doAgain);
		}
	}
	m_isUsed[1] = true; // init root
	for(int iActNode = 1; iActNode < m_maxInternalNodeNr + 1; ++iActNode){ // first element is not used!
		if(!m_isUsed[iActNode]){ // checks if node contains data or not
			continue;
		}
		// calc split value for each node
		// choose dimension for split
		const int randDim = uniformDist_Dimension(generator);  // generates number in the range 0...amountOfUsedDims - 1

		const int amountOfUsedData = uniformDist_usedData(generator);
		int maxScoreElement = -1;
		double actScore = -1000; // TODO check magic number
		for(int j = 0; j < amountOfUsedData; ++j){ // amount of checks for a specified split
			const int randElementId = uniformDist_Data(generator);
			const double score = trySplitFor(iActNode, randElementId, randDim, data, labels, nodesContent);
			if(score > actScore){
				actScore = score;
				maxScoreElement = randElementId;
			}
		}
		// save actual split
		m_splitValues[iActNode] = (double) data[maxScoreElement][randDim];
		m_splitDim[iActNode] = randDim;
		// apply split to data
		int foundDataLeft = 0, foundDataRight = 0;
		for(int j = 0; j < data.size(); ++j){
			if(nodesContent[j] == iActNode){ // move all elements from this node one down
				nodesContent[j] = iActNode * 2; // move in left child
				if(data[j][randDim] >= m_splitValues[iActNode]){ // TODO check >= like below  or only >
					++nodesContent[j]; // change to right child
					++foundDataRight;
				}else{
					++foundDataLeft;
				}
			}
		}
		if(foundDataLeft == 0 || foundDataRight == 0){
			// split is not needed
			// move data back up!
			for(int j = 0; j < data.size(); ++j){
				if(nodesContent[j] / 2 == iActNode){
					nodesContent[j] /= 2;
				}
			}
		}else{
			// set the use flag for children:
			m_isUsed[iActNode * 2] = foundDataLeft > 0;
			m_isUsed[iActNode * 2 + 1] = foundDataRight > 0;
		}

	}
	const int leafAmount = pow(2, m_maxDepth);
	const int offset = leafAmount; // pow(2, maxDepth - 1)
	for(int i = 0; i < leafAmount; ++i){
		std::vector<int> histo(m_amountOfClasses, 0);
		int amount = 0;
		int actNode = i + offset;
		while(!m_isUsed[actNode]){
			actNode /= 2;
		}
		for(int j = 0; j < data.size(); ++j){
			const int nodeOfDataPoint = nodesContent[j];
			if(nodeOfDataPoint == actNode){ // check offset right?
				++histo[labels[j]];
				++amount;
			}
		}

		int maxEle = 0, labelWithHighestOcc = 0;
		for(int k = 0; k < m_amountOfClasses; ++k){
			if(histo[k] > maxEle){
				maxEle = histo[k];
				labelWithHighestOcc = k;
			}
		}
		m_labelsOfWinningClassesInLeaves[i] = labelWithHighestOcc;
	}
}

double DecisionTree::trySplitFor(const int actNode, const int usedNode, const int usedDim, const Data& data, const Labels& labels, const std::vector<int>& nodesContent){
	std::vector<int> leftHisto(m_amountOfClasses), rightHisto(m_amountOfClasses);
	const double usedValue = data[usedNode][usedDim];
	double leftAmount = 0, rightAmount = 0;
	for(int i = 0; i < data.size(); ++i){
		if(actNode == nodesContent[i]){
			if(usedValue < data[i][usedDim]){ // TODO check < or <=
				++leftAmount;
				++leftHisto[labels[i]];
			}else{
				++rightAmount;
				++rightHisto[labels[i]];
			}
		}
	}

	// Entropy -> TODO maybe Gini
	double leftCost = 0, rightCost = 0;
	for(int i = 0; i < m_amountOfClasses; ++i){
		const double normalizer = leftHisto[i] + rightHisto[i];
		if(normalizer > 0){
			const double leftClassProb = leftHisto[i] / normalizer;
			if(leftClassProb > 0){
				leftCost -= leftClassProb * log(leftClassProb);
			}
			if(leftClassProb < 1.0){
				rightCost -= (1.-leftClassProb) * log((1.-leftClassProb));
			}
		}
	}

	return rightAmount * rightCost + leftAmount * leftCost;
}

int DecisionTree::predict(const DataElement& point) const{
	int iActNode = 1; // start in root
	while(iActNode <= m_maxInternalNodeNr){
		bool right = m_splitValues[iActNode] < point[m_splitDim[iActNode]];
		iActNode *= 2; // get to next level
		if(right){// point is on right side of split
			++iActNode; // go to right node
		}
		if(!m_isUsed[iActNode]){
			while(iActNode <= m_maxInternalNodeNr){
				iActNode *= 2;
			}
			break;
		}
	}
	return m_labelsOfWinningClassesInLeaves[iActNode - pow(2, m_maxDepth)];
}


