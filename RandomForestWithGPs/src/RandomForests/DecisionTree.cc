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
#include "DecisionTree.h"

#define MIN_NR_TO_SPLIT 2

DecisionTree::DecisionTree(const int maxDepth,
	const int amountOfClasses)
		: m_maxDepth(maxDepth),
			m_maxNodeNr(pow(2, maxDepth + 1) - 1),
			m_maxInternalNodeNr(pow(2, maxDepth) - 1),
			m_amountOfClasses(amountOfClasses),
			m_splitValues(m_maxInternalNodeNr + 1), // + 1 -> no use of the first element
			m_splitDim(m_maxInternalNodeNr + 1),
			m_isUsed(m_maxNodeNr + 1, false),
			m_labelsOfWinningClassesInLeaves(pow(2, maxDepth), -1){
}

DecisionTree::DecisionTree(const DecisionTree& tree):
		m_maxDepth(tree.m_maxDepth),
		m_maxNodeNr(tree.m_maxNodeNr),
		m_maxInternalNodeNr(tree.m_maxInternalNodeNr),
		m_amountOfClasses(tree.m_amountOfClasses){
	m_splitValues = tree.m_splitValues;
	m_splitDim = tree.m_splitDim;
	m_isUsed = tree.m_isUsed;
	m_labelsOfWinningClassesInLeaves = tree.m_labelsOfWinningClassesInLeaves;
}

DecisionTree::~DecisionTree(){
}

void DecisionTree::train(const Data& data,
	const Labels& labels,
	const int amountOfUsedDims,
	RandomNumberGeneratorForDT& generator){

	std::vector<int> usedDims(amountOfUsedDims);
	if(amountOfUsedDims == m_amountOfClasses){
		for(int i = 0; i < amountOfUsedDims; ++i){
			usedDims[i] = i;
		}
	}else{
		for(int i = 0; i < amountOfUsedDims; ++i){
			bool doAgain = false;
			do{
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
	m_isUsed[1] = true; // init root
	std::vector<int> leftHisto(m_amountOfClasses), rightHisto(m_amountOfClasses);
	std::vector<std::vector<int> > dataPosition(m_maxNodeNr + 1, std::vector<int>());
	for(int iActNode = 1; iActNode < m_maxInternalNodeNr + 1; ++iActNode){ // first element is not used!
		if(!m_isUsed[iActNode]){ // checks if node contains data or not
			continue;
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
			const double score = trySplitFor(iActNode, randElementId, randDim, data, labels,
					dataPosition[iActNode], leftHisto, rightHisto, generator);
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
		const int leftPos = iActNode * 2, rightPos = iActNode * 2 + 1;
		if(iActNode == 1){ // splitting like this avoids copying the whole stuff into the dataPosition[1]
			dataPosition[leftPos].reserve(dataPosition[iActNode].size());
			dataPosition[rightPos].reserve(dataPosition[iActNode].size());
			for(int i = 0; i < data.size(); ++i){
				if(data[i][randDim] >= m_splitValues[iActNode]){ // TODO check >= like below  or only >
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
				if(data[*it][randDim] >= m_splitValues[iActNode]){ // TODO check >= like below  or only >
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
		}else{
			dataPosition[iActNode].clear();
			// set the use flag for children:
			m_isUsed[leftPos] = foundDataLeft > 0;
			m_isUsed[rightPos] = foundDataRight > 0;
		}
	}
	const int leafAmount = pow(2, m_maxDepth);
	const int offset = leafAmount; // pow(2, maxDepth - 1)
	for(int i = 0; i < leafAmount; ++i){
		std::vector<int> histo(m_amountOfClasses, 0);
		int actNode = i + offset;
		while(!m_isUsed[actNode]){
			actNode /= 2;
		}
		for(std::vector<int>::const_iterator it = dataPosition[actNode].cbegin();
				it != dataPosition[actNode].cend(); ++it){
			++histo[labels[*it]];
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

double DecisionTree::trySplitFor(const int actNode,
		const int usedNode, const int usedDim,
		const Data& data, const Labels& labels,
		const std::vector<int>& dataInNode, std::vector<int>& leftHisto,
		std::vector<int>& rightHisto,
		RandomNumberGeneratorForDT& generator){
	const double usedValue = data[usedNode][usedDim];
	double leftAmount = 0, rightAmount = 0;
	if(dataInNode.size() < 100){ // under 100 take each value
		for(std::vector<int>::const_iterator it = dataInNode.cbegin(); it != dataInNode.cend();
				++it){
			if(usedValue < data[*it][usedDim]){ // TODO check < or <=
				++leftAmount;
				++leftHisto[labels[*it]];
			}else{
				++rightAmount;
				++rightHisto[labels[*it]];
			}
		}
	}else{
		const int stepSize = dataInNode.size() / 100;
		generator.setRandFromRange(1, stepSize);
		for(int i = 0; i < dataInNode.size(); i += generator.getRandFromRange()){
			const int val = dataInNode[i];
			if(usedValue < data[val][usedDim]){ // TODO check < or <=
				++leftAmount;
				++leftHisto[labels[val]];
			}else{
				++rightAmount;
				++rightHisto[labels[val]];
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

int DecisionTree::predict(const DataElement& point) const{
	int iActNode = 1; // start in root
	while(iActNode <= m_maxInternalNodeNr){
		bool right = m_splitValues[iActNode] < point[m_splitDim[iActNode]];
		iActNode *= 2; // get to next level
		if(right){ // point is on right side of split
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
	*(const_cast<int*>(&m_maxNodeNr)) = pow(2, m_maxDepth + 1) - 1;
	*(const_cast<int*>(&m_maxInternalNodeNr)) = data.nrOfInternalNodes;
	*(const_cast<int*>(&m_amountOfClasses)) = data.amountOfClasses;
	m_splitValues = data.splitValues;
	m_splitDim = data.dimValues;
	m_labelsOfWinningClassesInLeaves = data.labelsOfWinningClassInLeaves;
	m_isUsed.resize(m_maxNodeNr + 1);
	for(int i = 0; i < m_maxInternalNodeNr + 1; ++i){ // for internal nodes
		if(m_splitDim[i] == -1){
			m_isUsed[i] = false;
		}else{
			m_isUsed[i] = true;
		}
	}
	for(int i = 0; i < m_labelsOfWinningClassesInLeaves.size(); ++i){ // for leaves
		m_isUsed[m_maxInternalNodeNr + i + 1] = false; // set all to false
	}

}
