/*
 * BigDynamicDecisionTree.cc
 *
 *  Created on: 07.01.2017
 *      Author: Max
 */

#include "BigDynamicDecisionTree.h"
#include "../Base/Settings.h"

BigDynamicDecisionTree::BigDynamicDecisionTree(OnlineStorage<ClassPoint*>& storage, const int maxDepth, const int amountOfClasses, const int layerAmount):
	m_storage(storage),
	m_maxDepth(maxDepth),
	m_amountOfClasses(amountOfClasses){
	int amountOfLayers = layerAmount;
	if(layerAmount < 1){ // 0 and -1, ...
		Settings::getValue("OnlineRandomForest.layerAmountOfBigDDT", amountOfLayers);
	}
	m_depthPerLayer = m_maxDepth / (double) amountOfLayers;
	m_innerTrees.resize(amountOfLayers); // the last layer is not be placed with trees
	if(amountOfLayers > 0){
		m_innerTrees[0].resize(1); // start node
		m_innerTrees[0][0] = nullptr;
	}
}

BigDynamicDecisionTree::~BigDynamicDecisionTree() {
//	for(TreeStructure::iterator it = m_innerTrees.begin(); it != m_innerTrees.end(); ++it){
//		for(TreeInnerStructure::iterator itInner = it->begin(); itInner != it->end(); ++itInner){
//			SAVE_DELETE(*itInner); // nullptr can also be deleted
//		}
//	}
}

void BigDynamicDecisionTree::train(int amountOfUsedDims,
		RandomNumberGeneratorForDT& generator){
	if(m_innerTrees.size() > 0 && m_innerTrees[0][0] != nullptr){ // in the case of a retraining that all trees are removed
		for(TreeStructure::iterator it = m_innerTrees.begin(); it != m_innerTrees.end(); ++it){
			for(TreeInnerStructure::iterator itInner = it->begin(); itInner != it->end(); ++itInner){
				SAVE_DELETE(*itInner); // nullptr can also be deleted
			}
		}
	}
//	const unsigned int leavesPerTree = pow(2, m_depthPerLayer);
	unsigned int rootsForTreesInThisLayer = 1;
	unsigned int depthInTheFatherLayer = m_depthPerLayer;
	for(unsigned int iTreeLayer = 0; iTreeLayer < m_innerTrees.size(); ++iTreeLayer){
//		printOnScreen("layer: " << iTreeLayer);
		bool saveDataPositions = true;  // only in the last layer the data positions don't need to be saved
		unsigned int depthInThisLayer = m_depthPerLayer;
		if(iTreeLayer + 1 == m_innerTrees.size()){ // is the last layer
			saveDataPositions = false;
			depthInThisLayer = m_maxDepth - m_innerTrees.size() * m_depthPerLayer;
		}
		const unsigned int leavesForTreesInThisLayer = pow(2, depthInThisLayer); // amount of leaves of one of the layertrees
		if(iTreeLayer == 0){
			// first tree
			m_innerTrees[iTreeLayer][0] = new DynamicDecisionTree(m_storage, m_depthPerLayer, m_amountOfClasses);
			const bool ret = m_innerTrees[iTreeLayer][0]->train(amountOfUsedDims, generator, 0, saveDataPositions);
			if(!ret){
				printError("The first split could not be performed!");
				return;
			}
		}else{
			const unsigned int leavesForTreesInTheFatherLayer = pow(2, depthInTheFatherLayer); // amount of leaves of one of the layertrees
			const unsigned int amountOfChildrenInThisLayer = rootsForTreesInThisLayer * leavesForTreesInThisLayer;
			m_innerTrees[iTreeLayer].resize(amountOfChildrenInThisLayer);
			std::fill(m_innerTrees[iTreeLayer].begin(), m_innerTrees[iTreeLayer].end(), nullptr); // set all values to null
			for(unsigned int iRootId = 0; iRootId < rootsForTreesInThisLayer; ++iRootId){
//				printOnScreen("Rootid: " << iRootId);
				if(m_innerTrees[iTreeLayer - 1][iRootId] != nullptr){ // if the father is not a nullpointer
					std::vector<std::vector<int> >& dataPositions = *m_innerTrees[iTreeLayer - 1][iRootId]->getDataPositions();
					for(unsigned int iChild = 0; iChild < leavesForTreesInThisLayer; ++iChild){
//						printOnScreen("size: " << dataPositions.size() << ", leaves: " << leavesForTreesInTheFatherLayer
//									<< ", ichild: " << iChild << ", desired amount: " << leavesForTreesInThisLayer << ", root: " << iRootId
//									<< "in data: " << dataPositions[leavesForTreesInTheFatherLayer + iChild].size());
						if(dataPositions[leavesForTreesInTheFatherLayer + iChild].size() > 5){ // min amount is that half of the leaves are filled which at least one point!
							const unsigned int iChildIdInLayer = iChild + leavesForTreesInThisLayer * iRootId;
//							printOnScreen("Child: " << iChildIdInLayer);
							m_innerTrees[iTreeLayer][iChildIdInLayer] = new DynamicDecisionTree(m_storage, depthInThisLayer, m_amountOfClasses);
							m_innerTrees[iTreeLayer][iChildIdInLayer]->setUsedDataPositions(&dataPositions[leavesForTreesInTheFatherLayer + iChild]); // set the values of the storage which should be used in this tree
							const bool trained = m_innerTrees[iTreeLayer][iChildIdInLayer]->train(amountOfUsedDims, generator, 0, saveDataPositions);
							//							m_innerTrees[iTreeLayer][iChildIdInLayer]->setUsedDataPositions(nullptr); // erase pointer to used dataPositions
							if(!trained){
								SAVE_DELETE(m_innerTrees[iTreeLayer][iChildIdInLayer]);
							}
						}
					}
				}
			}
			rootsForTreesInThisLayer = amountOfChildrenInThisLayer; // amount of children nodes in this level
			depthInTheFatherLayer = depthInThisLayer;
		}
	}
//	int amountOfUsedTrees = 0;
//	for(unsigned int iTreeLayer = 0; iTreeLayer < m_innerTrees.size(); ++iTreeLayer){
//		for(unsigned int iChild = 0; iChild < m_innerTrees[iTreeLayer].size(); ++iChild){
//			if(m_innerTrees[iTreeLayer][iChild] != nullptr){
//				++amountOfUsedTrees;
//			}
//		}
//	}
//	printOnScreen("Needed " << amountOfUsedTrees);
//	for(unsigned int iTreeLayer = 0; iTreeLayer < m_innerTrees.size(); ++iTreeLayer){
//		for(unsigned int iChild = 0; iChild < m_innerTrees[iTreeLayer].size(); ++iChild){
//			if(m_innerTrees[iTreeLayer][iChild] != nullptr){
//				m_innerTrees[iTreeLayer][iChild]->deleteDataPositions();
//			}
//		}
//	}
}


int BigDynamicDecisionTree::predict(const DataPoint& point) const{
	if(m_innerTrees[0][0] != nullptr){
		int winningNode = 1;
		unsigned int result = m_innerTrees[0][0]->predict(point, winningNode);
		winningNode -= m_depthPerLayer; // the first layer always is m_depthPerLayer high
		int iFatherId = winningNode;
		for(unsigned int iTreeLayer = 1; iTreeLayer < m_innerTrees.size(); ++iTreeLayer){
			const unsigned int depthForThisLayer = iTreeLayer + 1 == m_innerTrees.size() ? m_maxDepth - m_innerTrees.size() * m_depthPerLayer : m_depthPerLayer;
			const unsigned int iChildInLayer = winningNode + depthForThisLayer * iFatherId;
			if(iChildInLayer < m_innerTrees[iTreeLayer].size()){
				if(m_innerTrees[iTreeLayer][iChildInLayer] != nullptr){
					result = m_innerTrees[iTreeLayer][iChildInLayer]->predict(point, winningNode);
					winningNode -= depthForThisLayer; // on the same level as the calc of iChildInLayer
				}else{
					return result;
				}
				iFatherId = iChildInLayer;
			}else{
				break;
			}
		}
		return result;
	}
	return UNDEF_CLASS_LABEL;
}

unsigned int BigDynamicDecisionTree::amountOfClasses() const{
	return m_amountOfClasses;
}
