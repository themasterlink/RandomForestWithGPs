/*
 * DynamicDecisionTree.cc
 *
 *  Created on: 17.10.2016
 *      Author: Max
 */

#include "DynamicDecisionTree.h"
#include "../Utility/Util.h"
#include <algorithm>

DynamicDecisionTree::DynamicDecisionTree(OnlineStorage<ClassPoint*>& storage, const int maxDepth, const int amountOfClasses):
		m_storage(storage),
		m_maxDepth(maxDepth),
		m_maxNodeNr(pow(2, maxDepth + 1) - 1),
		m_maxInternalNodeNr(pow(2, maxDepth) - 1),
		m_amountOfClasses(amountOfClasses),
		m_splitValues(m_maxInternalNodeNr + 1), // + 1 -> no use of the first element
		m_splitDim(m_maxInternalNodeNr + 1, NODE_IS_NOT_USED),
		m_labelsOfWinningClassesInLeaves(pow(2, maxDepth), -1){
}

// copy construct
DynamicDecisionTree::DynamicDecisionTree(const DynamicDecisionTree& tree):
		m_storage(tree.m_storage),
		m_maxDepth(tree.m_maxDepth),
		m_maxNodeNr(tree.m_maxNodeNr),
		m_maxInternalNodeNr(tree.m_maxInternalNodeNr),
		m_amountOfClasses(tree.m_amountOfClasses),
		m_splitValues(tree.m_splitValues),
		m_splitDim(tree.m_splitDim),
		m_labelsOfWinningClassesInLeaves(tree.m_labelsOfWinningClassesInLeaves){
}

DynamicDecisionTree::~DynamicDecisionTree(){
}

void DynamicDecisionTree::DynamicDecisionTree::train(const int amountOfUsedDims,
		RandomNumberGeneratorForDT& generator){
	if(m_splitDim[1] != NODE_IS_NOT_USED || m_splitDim[1] != NODE_CAN_BE_USED){
		// reset training
		std::fill(m_splitDim.begin(), m_splitDim.end(), NODE_IS_NOT_USED);
		std::fill(m_labelsOfWinningClassesInLeaves.begin(), m_labelsOfWinningClassesInLeaves.end(), -1);
	}
	std::vector<int> usedDims(amountOfUsedDims,-1);
	if(amountOfUsedDims == m_storage.dim()){
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
	m_splitDim[1] = NODE_CAN_BE_USED; // init the root value
	std::vector<int> leftHisto(m_amountOfClasses), rightHisto(m_amountOfClasses);
	std::vector<std::vector<int> > dataPosition(m_maxNodeNr + 1, std::vector<int>());
	for(int iActNode = 1; iActNode < m_maxInternalNodeNr + 1; ++iActNode){ // first element is not used!
		if(m_splitDim[iActNode] == NODE_IS_NOT_USED){ // checks if node contains data or not
			continue; // if node is not used, go to next node, if node can be used process it
		}
		// calc actual nodes
		// calc split value for each node
		// choose dimension for split
		const int randDim = usedDims[generator.getRandDim()]; // generates number in the range 0...amountOfUsedDims - 1
		const int amountOfUsedData = generator.getRandAmountOfUsedData();
		double maxScoreElementValue = 0;
		double actScore = -DBL_MAX; // TODO check magic number
		for(int j = 0; j < amountOfUsedData; ++j){ // amount of checks for a specified split
			//const int randElementId = generator.getRandNextDataEle();
			//const double usedValue = (*m_storage[usedNode])[usedDim];
			const double usedValue = generator.getRandSplitValueInDim(randDim);
			const double score = trySplitFor(iActNode, usedValue, randDim,
					dataPosition[iActNode], leftHisto, rightHisto, generator);
			if(score > actScore){
				actScore = score;
				maxScoreElementValue = usedValue;
			}
		}
		// save actual split
		m_splitValues[iActNode] = maxScoreElementValue;//(double) (*m_storage[maxScoreElement])[randDim];
		m_splitDim[iActNode] = randDim;
		// apply split to data
		int foundDataLeft = 0, foundDataRight = 0;
		const int leftPos = iActNode * 2, rightPos = iActNode * 2 + 1;
		if(iActNode == 1){ // splitting like this avoids copying the whole stuff into the dataPosition[1]
			dataPosition[leftPos].reserve(dataPosition[iActNode].size());
			dataPosition[rightPos].reserve(dataPosition[iActNode].size());
			for(int i = 0; i < m_storage.size(); ++i){
				if((*m_storage[i])[randDim] >= m_splitValues[iActNode]){ // TODO check >= like below  or only >
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
				if((*m_storage[*it])[randDim] >= m_splitValues[iActNode]){ // TODO check >= like below  or only >
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
			m_splitDim[iActNode] = NODE_IS_NOT_USED; // do not split here
			if(iActNode == 1){
				m_splitDim[iActNode] = NODE_CAN_BE_USED; // there should be a split
				// todo maybe avoid endless loop
			}
		}else{
			dataPosition[iActNode].clear();
			// set the use flag for children:
			if(rightPos < m_maxInternalNodeNr + 1){ // if right is leave, than left is too -> just control one
				m_splitDim[leftPos] = foundDataLeft > 0 ? NODE_CAN_BE_USED : NODE_IS_NOT_USED;
				m_splitDim[rightPos] = foundDataRight > 0 ? NODE_CAN_BE_USED : NODE_IS_NOT_USED;
			}
		}
	}
	const int leafAmount = pow(2, m_maxDepth);
	const int offset = leafAmount; // pow(2, maxDepth - 1)
	for(int i = 0; i < leafAmount; ++i){
		std::vector<int> histo(m_amountOfClasses, 0);
		int lastValue = i + offset;
		int actNode = lastValue / 2;
		while(m_splitDim[actNode] == NODE_IS_NOT_USED && actNode > 1){
			lastValue = actNode; // save correct child
			actNode /= 2; // if node is not take parent and try again
		}

		for(std::vector<int>::const_iterator it = dataPosition[lastValue].cbegin();
				it != dataPosition[lastValue].cend(); ++it){
			++histo[m_storage[*it]->getLabel()];
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
	if(m_splitDim[1] == NODE_CAN_BE_USED){
		// try again!
		train(amountOfUsedDims,generator);
	}
}

double DynamicDecisionTree::trySplitFor(const int actNode, const double usedValue, const int usedDim,
		const std::vector<int>& dataInNode, std::vector<int>& leftHisto,
		std::vector<int>& rightHisto, RandomNumberGeneratorForDT& generator){
	double leftAmount = 0, rightAmount = 0;
	if(dataInNode.size() < 100){ // under 100 take each value
		for(std::vector<int>::const_iterator it = dataInNode.cbegin(); it != dataInNode.cend();
				++it){
			if(usedValue < (*m_storage[*it])[usedDim]){ // TODO check < or <=
				++leftAmount;
				++leftHisto[m_storage[*it]->getLabel()];
			}else{
				++rightAmount;
				++rightHisto[m_storage[*it]->getLabel()];
			}
		}
	}else{
		const int stepSize = dataInNode.size() / 100;
		generator.setRandFromRange(1, stepSize);
		const int dataSize = dataInNode.size();
		for(int i = 0; i < dataSize; i += generator.getRandFromRange()){
			if(i < dataSize){
				const int val = dataInNode[i];
				if(usedValue < (*m_storage[val])[usedDim]){ // TODO check < or <=
					++leftAmount;
					++leftHisto[m_storage[val]->getLabel()];
				}else{
					++rightAmount;
					++rightHisto[m_storage[val]->getLabel()];
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

int DynamicDecisionTree::predict(const DataPoint& point) const{
	int iActNode = 1; // start in root
	if(m_splitDim[1] != NODE_IS_NOT_USED && m_splitDim[1] != NODE_CAN_BE_USED){
		while(iActNode <= m_maxInternalNodeNr){
			if(m_splitDim[iActNode] == NODE_IS_NOT_USED){
				// if there is a node which isn't used on the way down to the leave
				while(iActNode <= m_maxInternalNodeNr){ // go down always on the left side (it doesn't really matter)
					iActNode *= 2;
				}
				break;
			}else if(m_splitDim[iActNode] == NODE_CAN_BE_USED){
				// if there is a node which isn't used on the way down to the leave
				while(iActNode <= m_maxInternalNodeNr){ // go down always on the left side (it doesn't really matter)
					iActNode *= 2;
				}
				break;
			}
			const bool right = m_splitValues[iActNode] < point[m_splitDim[iActNode]];
			iActNode *= 2; // get to next level
			if(right){ // point is on right side of split
				++iActNode; // go to right node
			}
		}
		return m_labelsOfWinningClassesInLeaves[iActNode - pow(2, m_maxDepth)];
	}else{
		printError("A tree must be trained before it can predict anything!");
		return -1;
	}
}

bool DynamicDecisionTree::predictIfPointsShareSameLeaveWithHeight(const DataPoint& point1, const DataPoint& point2, const int usedHeight) const {
	int iActNode = 1; // start in root
	int actLevel = 1;
	if(m_splitDim[1] != NODE_IS_NOT_USED && m_splitDim[1] != NODE_CAN_BE_USED){
		while(iActNode <= m_maxInternalNodeNr){
			if(m_splitDim[iActNode] == NODE_IS_NOT_USED){
				// if there is a node which isn't used on the way down to the leave
				while(iActNode <= m_maxInternalNodeNr){ // go down always on the left side (it doesn't really matter)
					iActNode *= 2;
				}
				break;
			}else if(m_splitDim[iActNode] == NODE_CAN_BE_USED){
				// if there is a node which isn't used on the way down to the leave
				while(iActNode <= m_maxInternalNodeNr){ // go down always on the left side (it doesn't really matter)
					iActNode *= 2;
				}
				break;
			}
			const bool firstPointRight = m_splitValues[iActNode] < point1[m_splitDim[iActNode]];
			const bool secondPointRight = m_splitValues[iActNode] < point2[m_splitDim[iActNode]];
			if(firstPointRight != secondPointRight){ // walk in different directions
				return false;
			}
			iActNode *= 2; // get to next level
			if(firstPointRight){ // point is on right side of split
				++iActNode; // go to right node
			}
			if(actLevel == usedHeight){
				return true; // reached height
			}
			++actLevel;
		}
		return true; // share the same node
	}else{
		printError("A tree must be trained before it can predict anything!");
		return false;
	}
}

void DynamicDecisionTree::adjustToNewData(){
	for(std::vector<int>::iterator it = m_labelsOfWinningClassesInLeaves.begin(); it != m_labelsOfWinningClassesInLeaves.end(); ++it){
		*it = 0;
	}
	const int leafAmount = pow(2, m_maxDepth);
	std::vector< std::vector<int> > histo(leafAmount, std::vector<int>(m_amountOfClasses, 0));
	for(OnlineStorage<ClassPoint*>::ConstIterator it = m_storage.begin(); it != m_storage.end(); ++it){
		int iActNode = 1; // start in root
		while(iActNode <= m_maxInternalNodeNr){
			if(m_splitDim[iActNode] == NODE_IS_NOT_USED){
				// if there is a node which isn't used on the way down to the leave
				while(iActNode <= m_maxInternalNodeNr){ // go down always on the left side (it doesn't really matter)
					iActNode *= 2;
				}
				break;
			}else if(m_splitDim[iActNode] == NODE_CAN_BE_USED){
				// if there is a node which isn't used on the way down to the leave
				while(iActNode <= m_maxInternalNodeNr){ // go down always on the left side (it doesn't really matter)
					iActNode *= 2;
				}
				break;
			}
			const bool right = m_splitValues[iActNode] < (**it)[m_splitDim[iActNode]];
			iActNode *= 2; // get to next level
			if(right){ // point is on right side of split
				++iActNode; // go to right node
			}

		}
		++histo[iActNode - leafAmount][(*it)->getLabel()];
	}
	for(unsigned int i = 0; i < leafAmount; ++i){
		int max = -1;
		int ele = -1;
		for(unsigned int k = 0; k < m_amountOfClasses; ++k){
			if(histo[i][k] > max){
				max = histo[i][k];
				ele = k;
			}
		}
		m_labelsOfWinningClassesInLeaves[i] = ele;
	}
}

int DynamicDecisionTree::getNrOfLeaves(){
	return pow(2, m_maxDepth);
}

int DynamicDecisionTree::amountOfClasses() const{
	return m_amountOfClasses;
}

