/*
 * BigDynamicDecisionTree.cc
 *
 *  Created on: 07.01.2017
 *      Author: Max
 */

#include "BigDynamicDecisionTree.h"

BigDynamicDecisionTree::BigDynamicDecisionTree(OnlineStorage<ClassPoint*>& storage, const int maxDepth, const int amountOfClasses):
	m_storage(storage),
	m_maxDepth(maxDepth),
	m_amountOfClasses(amountOfClasses){
	const int amountOfLayers = 3;
	m_depthPerLayer = m_maxDepth / (double) amountOfLayers;
	m_innerTrees.resize(amountOfLayers - 1); // the last layer is not be placed with trees
}

BigDynamicDecisionTree::~BigDynamicDecisionTree() {
	// TODO Auto-generated destructor stub
}



void BigDynamicDecisionTree::train(int amountOfUsedDims,
		RandomNumberGeneratorForDT& generator){
	const bool saveDataPositions = true;
	const unsigned int leavesPerTree = pow(2, m_depthPerLayer);
	for(unsigned int iTreeLayer = 0; iTreeLayer < m_innerTrees.size(); ++iTreeLayer){
		const int depthForThisLayer = iTreeLayer + 1 == m_innerTrees.size() ? m_maxDepth - m_innerTrees.size() * m_depthPerLayer : m_depthPerLayer ;
		const unsigned int sizeOfActLayer = pow(2, m_depthPerLayer * iTreeLayer);
		m_innerTrees[iTreeLayer].resize(sizeOfActLayer);
		if(iTreeLayer == 0){
			// first tree
			m_innerTrees[iTreeLayer][0] = new DynamicDecisionTree(m_storage, depthForThisLayer, m_amountOfClasses);
			const bool ret = m_innerTrees[iTreeLayer][0]->train(amountOfUsedDims, generator, 0, saveDataPositions);
			if(!ret){
				printError("The first split could not be performed!");
				return;
			}
		}else{
			std::fill(m_innerTrees[iTreeLayer].begin(), m_innerTrees[iTreeLayer].end(), nullptr);
			const unsigned innerAmountOfNodesInEachTree = pow(2, m_depthPerLayer);  //  (- 1 + 1) == 0 -> +1 because the 0 element in each tree is not used
			const unsigned int amountOfFathers = sizeOfActLayer / leavesPerTree;
			for(unsigned int iFatherId = 0; iFatherId < amountOfFathers; ++iFatherId){
				if(m_innerTrees[iTreeLayer - 1][iFatherId] != nullptr){
					std::vector<std::vector<int> >& dataPositions = *m_innerTrees[iTreeLayer - 1][iFatherId]->getDataPositions();
					for(unsigned int iChild = 0; iChild < leavesPerTree; ++iChild){
						if(dataPositions[innerAmountOfNodesInEachTree + iChild].size() > leavesPerTree / 2){ // min amount is that half of the leaves are filled which at least one point!
							const unsigned int iChildIdInLayer = iChild + leavesPerTree * iFatherId;
							m_innerTrees[iTreeLayer][iChildIdInLayer] = new DynamicDecisionTree(m_storage, depthForThisLayer, m_amountOfClasses);
							m_innerTrees[iTreeLayer][iChildIdInLayer]->setUsedDataPositions(&dataPositions[innerAmountOfNodesInEachTree + iChild]); // set the values of the storage which should be used in this tree
							const bool trained = m_innerTrees[iTreeLayer][iChildIdInLayer]->train(amountOfUsedDims, generator, 0, saveDataPositions);
							if(!trained){
								delete m_innerTrees[iTreeLayer][iChildIdInLayer];
								m_innerTrees[iTreeLayer][iChildIdInLayer] = nullptr;
							}
						}
					}
				}
			}
		}
	}
}


int BigDynamicDecisionTree::predict(const DataPoint& point) const{
	const unsigned int leavesPerTree = pow(2, m_depthPerLayer);
	if(m_innerTrees[0][0] != nullptr){
		int winningNode = 1;
		unsigned int result = m_innerTrees[0][0]->predict(point, winningNode);
		winningNode -= leavesPerTree;
		for(unsigned int iTreeLayer = 1; iTreeLayer < m_innerTrees.size(); ++iTreeLayer){
			if(m_innerTrees[iTreeLayer][winningNode] != nullptr){
				result = m_innerTrees[iTreeLayer][winningNode]->predict(point, winningNode);
				winningNode -= leavesPerTree;
			}else{
				return result;
			}
		}
		return result;
	}
	return UNDEF_CLASS_LABEL;
}

unsigned int BigDynamicDecisionTree::amountOfClasses() const{
	return m_amountOfClasses;
}
