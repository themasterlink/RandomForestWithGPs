/*
 * OtherDecisionTree.cc
 *
 *  Created on: 03.06.2016
 *      Author: Max
 */

#include <cmath>
#include <random>
#include <algorithm>
#include <limits>
#include "../Utility/Util.h"
#include "DecisionTree.h"
#include <boost/thread.hpp> // Boost threads

#define MIN_NR_TO_SPLIT 2

DecisionTree::DecisionTree(const int maxDepth,
	const int amountOfClasses)
		: m_maxDepth(maxDepth),
			m_maxNodeNr(pow2(maxDepth + 1) - 1),
			m_maxInternalNodeNr(pow2(maxDepth) - 1),
			m_amountOfClasses(amountOfClasses),
			m_splitValues(m_maxInternalNodeNr + 1), // + 1 -> no use of the first element
			m_splitDim(m_maxInternalNodeNr + 1, NodeType::NODE_IS_NOT_USED),
			m_labelsOfWinningClassesInLeaves(pow2(maxDepth), UNDEF_CLASS_LABEL){
}

DecisionTree::DecisionTree(const DecisionTree& tree):
		m_maxDepth(tree.m_maxDepth),
		m_maxNodeNr(tree.m_maxNodeNr),
		m_maxInternalNodeNr(tree.m_maxInternalNodeNr),
		m_amountOfClasses(tree.m_amountOfClasses){
	m_splitValues = tree.m_splitValues;
	m_splitDim = tree.m_splitDim;
	m_labelsOfWinningClassesInLeaves = tree.m_labelsOfWinningClassesInLeaves;
}

DecisionTree& DecisionTree::operator=(const DecisionTree& tree){
	*(const_cast<int*>(&m_maxDepth)) = tree.m_maxDepth; // change of const value
	*(const_cast<int*>(&m_maxNodeNr)) = tree.m_maxNodeNr;
	*(const_cast<int*>(&m_maxInternalNodeNr)) = tree.m_maxInternalNodeNr;
	*(const_cast<int*>(&m_amountOfClasses)) = tree.m_amountOfClasses;
	m_splitValues = tree.m_splitValues;
	m_splitDim = tree.m_splitDim;
	m_labelsOfWinningClassesInLeaves = tree.m_labelsOfWinningClassesInLeaves;
	return *this;
}

DecisionTree::~DecisionTree(){
}

void DecisionTree::train(const LabeledData& data,
	const int amountOfUsedDims,
	RandomNumberGeneratorForDT& generator){
	std::vector<int> usedDims(amountOfUsedDims,-1);
	if(amountOfUsedDims == data[0]->rows()){
		for(int i = 0; i < amountOfUsedDims; ++i){
			usedDims[i] = i;
		}
	}else{
		for(int i = 0; i < amountOfUsedDims; ++i){
			bool doAgain = false;
			do{
				doAgain = false;
				const int randNr = generator.getRandDim(); // generates number in the range 0...data.rows() - 1;
				for(int j = 0; j < i; ++j){
					if(randNr == usedDims[j]){
						doAgain = true;
						break;
					}
				}
				if(!doAgain){
					usedDims[i] = randNr;
				}
 			}while(doAgain);
		}
	}
	m_splitDim[1] = NodeType::NODE_CAN_BE_USED; // init the root value
	std::vector<int> leftHisto(m_amountOfClasses), rightHisto(m_amountOfClasses);
	std::vector<std::vector<int> > dataPosition(m_maxNodeNr + 1, std::vector<int>());
	for(int iActNode = 1; iActNode < m_maxInternalNodeNr + 1; ++iActNode){ // first element is not used!
		if(m_splitDim[iActNode] == NodeType::NODE_IS_NOT_USED){ // checks if node contains data or not
			continue; // if node is not used, go to next node, if node can be used process it
		}
		// calc actual nodes
		// calc split value for each node
		// choose dimension for split

		const int randDim = usedDims[generator.getRandDim()]; // generates number in the range 0...amountOfUsedDims - 1
		const int amountOfUsedData = generator.getRandAmountOfUsedData();
		int maxScoreElement = -1;
		double actScore = -1000; // TODO check magic number
		for(int j = 0; j < amountOfUsedData; ++j){ // amount of checks for a specified split
			const int randElementId = generator.getRandNextDataEle();
			const double score = trySplitFor(randElementId, randDim, data,
					dataPosition[iActNode], leftHisto, rightHisto, generator);
			if(score > actScore){
				actScore = score;
				maxScoreElement = randElementId;
			}
		}
		// save actual split
		m_splitValues[iActNode] = (double) (*data[maxScoreElement])[randDim];
		m_splitDim[iActNode] = randDim;
		// apply split to data
		int foundDataLeft = 0, foundDataRight = 0;
		const int leftPos = iActNode * 2, rightPos = iActNode * 2 + 1;
		if(iActNode == 1){ // splitting like this avoids copying the whole stuff into the dataPosition[1]
			dataPosition[leftPos].reserve(dataPosition[iActNode].size());
			dataPosition[rightPos].reserve(dataPosition[iActNode].size());
			for(unsigned int i = 0; i < (unsigned int) data.size(); ++i){
				if((*data[i])[randDim] >= m_splitValues[iActNode]){ // TODO check >= like below  or only >
					dataPosition[rightPos].push_back(i);
					++foundDataRight;
				}else{
					dataPosition[leftPos].push_back(i);
					++foundDataLeft;
				}
			}
		}else{
			dataPosition[leftPos].reserve(dataPosition[iActNode].size());
			dataPosition[rightPos].reserve(dataPosition[iActNode].size());
			for(std::vector<int>::const_iterator it = dataPosition[iActNode].cbegin();
					it != dataPosition[iActNode].cend(); ++it){
				if((*data[*it])[randDim] >= m_splitValues[iActNode]){ // TODO check >= like below  or only >
					dataPosition[rightPos].push_back(*it);
					++foundDataRight;
				}else{
					dataPosition[leftPos].push_back(*it);
					++foundDataLeft;
				}
			}
		}
		/*std::cout << "i: " << iActNode << std::endl;
		 std::cout << "length: " << dataPosition[iActNode].size() << std::endl;
		 std::cout << "Found data left  " << foundDataLeft << std::endl;
		 std::cout << "Found data right " << foundDataRight << std::endl;*/
		if(foundDataLeft == 0 || foundDataRight == 0){
			// split is not needed
			dataPosition[leftPos].clear();
			dataPosition[rightPos].clear();
			m_splitDim[iActNode] = NodeType::NODE_IS_NOT_USED; // do not split here
			if(iActNode == 1){
				m_splitDim[iActNode] = NodeType::NODE_CAN_BE_USED; // there should be a split
				// todo maybe avoid endless loop
			}
		}else{
			dataPosition[iActNode].clear();
			// set the use flag for children:
			if(rightPos < m_maxInternalNodeNr + 1){ // if right is leave, than left is too -> just control one
				m_splitDim[leftPos] = foundDataLeft > 0 ? NodeType::NODE_CAN_BE_USED : NodeType::NODE_IS_NOT_USED;
				m_splitDim[rightPos] = foundDataRight > 0 ? NodeType::NODE_CAN_BE_USED : NodeType::NODE_IS_NOT_USED;
			}
		}
	}
	const int leafAmount = pow2(m_maxDepth);
	const int offset = leafAmount; // pow2(maxDepth - 1)
	for(int i = 0; i < leafAmount; ++i){
		std::vector<int> histo(m_amountOfClasses, 0);
		int lastValue = i + offset;
		int actNode = lastValue / 2;
		while(m_splitDim[actNode] == NodeType::NODE_IS_NOT_USED && actNode > 1){
			lastValue = actNode; // save correct child
			actNode /= 2; // if node is not take parent and try again
		}

		for(std::vector<int>::const_iterator it = dataPosition[lastValue].cbegin();
				it != dataPosition[lastValue].cend(); ++it){
			++histo[data[*it]->getLabel()];
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
	if(m_splitDim[1] == NodeType::NODE_CAN_BE_USED){
		// try again!
		train(data, amountOfUsedDims,generator);
	}
}

double DecisionTree::trySplitFor(const int usedNode, const int usedDim,
		const LabeledData& data,
		const std::vector<int>& dataInNode, std::vector<int>& leftHisto,
		std::vector<int>& rightHisto,
		RandomNumberGeneratorForDT& generator){
	const double usedValue = (*data[usedNode])[usedDim];
	double leftAmount = 0, rightAmount = 0;
	if(dataInNode.size() < 100){ // under 100 take each value
		for(std::vector<int>::const_iterator it = dataInNode.cbegin(); it != dataInNode.cend();
				++it){
			if(usedValue < (*data[*it])[usedDim]){ // TODO check < or <=
				++leftAmount;
				++leftHisto[data[*it]->getLabel()];
			}else{
				++rightAmount;
				++rightHisto[data[*it]->getLabel()];
			}
		}
	}else{
		const int stepSize = dataInNode.size() / 100;
		generator.setRandFromRange(1, stepSize);
		const int dataSize = dataInNode.size();
		for(int i = 0; i < dataSize; i += generator.getRandFromRange()){
			if(i < dataSize){
				const int val = dataInNode[i];
				if(usedValue < (*data[val])[usedDim]){ // TODO check < or <=
					++leftAmount;
					++leftHisto[data[val]->getLabel()];
				}else{
					++rightAmount;
					++rightHisto[data[val]->getLabel()];
				}
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
				rightCost -= (1. - leftClassProb) * log((1. - leftClassProb));
			}
		}
		leftHisto[i] = 0;
		rightHisto[i] = 0;
	}
	return rightAmount * rightCost + leftAmount * leftCost;
}

int DecisionTree::predict(const VectorX& point) const{
	int iActNode = 1; // start in root
	if(m_splitDim[1] != NodeType::NODE_IS_NOT_USED && m_splitDim[1] != NodeType::NODE_CAN_BE_USED){
		while(iActNode <= m_maxInternalNodeNr){
			const bool right = m_splitValues[iActNode] < point[m_splitDim[iActNode]];
			iActNode *= 2; // get to next level
			if(right){ // point is on right side of split
				++iActNode; // go to right node
			}
			if(m_splitDim[iActNode] == NodeType::NODE_IS_NOT_USED){
				// if there is a node which isn't used on the way down to the leave
				while(iActNode <= m_maxInternalNodeNr){ // go down always on the left side (it doesn't really matter)
					iActNode *= 2;
				}
				break;
			}else if(m_splitDim[iActNode] == NodeType::NODE_CAN_BE_USED){
				// if there is a node which isn't used on the way down to the leave
				while(iActNode <= m_maxInternalNodeNr){ // go down always on the left side (it doesn't really matter)
					iActNode *= 2;
				}
				break;
			}
		}
		return m_labelsOfWinningClassesInLeaves[iActNode - pow2(m_maxDepth)];
	}else{
		printError("A tree must be trained before it can predict anything!");
		return UNDEF_CLASS_LABEL;
	}
}

void DecisionTree::writeToData(DecisionTreeData& data) const{
	data.height = m_maxDepth;
	data.nrOfLeaves = m_labelsOfWinningClassesInLeaves.size();
	data.nrOfInternalNodes = m_maxInternalNodeNr; // size of splitDim and splitValues
	data.amountOfClasses = m_amountOfClasses;
	data.splitValues = m_splitValues;
	data.dimValues = m_splitDim;
	data.labelsOfWinningClassInLeaves = m_labelsOfWinningClassesInLeaves;
}

void DecisionTree::initFromData(const DecisionTreeData& data){
	*(const_cast<int*>(&m_maxDepth)) = data.height; // change of const value
	*(const_cast<int*>(&m_maxNodeNr)) = pow2(m_maxDepth + 1) - 1;
	*(const_cast<int*>(&m_maxInternalNodeNr)) = data.nrOfInternalNodes;
	*(const_cast<int*>(&m_amountOfClasses)) = data.amountOfClasses;
	m_splitValues = data.splitValues;
	m_splitDim = data.dimValues;
	m_labelsOfWinningClassesInLeaves = data.labelsOfWinningClassInLeaves;
}
