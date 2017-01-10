/*
 * BigDynamicDecisionTree.cc
 *
 *  Created on: 07.01.2017
 *      Author: Max
 */

#include "BigDynamicDecisionTree.h"
#include "../Base/Settings.h"

BigDynamicDecisionTree::BigDynamicDecisionTree(OnlineStorage<ClassPoint*>& storage, const int maxDepth, const int amountOfClasses, const int layerAmount, const int layerAmountForFast):
	m_storage(storage),
	m_maxDepth(maxDepth),
	m_amountOfClasses(amountOfClasses){
	int amountOfLayers = layerAmount;
	if(layerAmount < 1){ // 0 and -1, ...
		Settings::getValue("OnlineRandomForest.layerAmountOfBigDDT", amountOfLayers);
	}
	const int amountForFast = std::min(std::max(layerAmountForFast, 1), amountOfLayers);
	const int amountForSmall = std::max(amountOfLayers - amountForFast, 0);
	m_depthPerLayer = m_maxDepth / (double) amountOfLayers;
	m_fastInnerTrees.resize(amountForFast); // the last layer is not be placed with trees
	m_smallInnerTrees.resize(amountForSmall);
	if(amountForFast > 0){
		m_fastInnerTrees[0].resize(1); // start node
		m_fastInnerTrees[0][0] = nullptr;
	}
}

BigDynamicDecisionTree::~BigDynamicDecisionTree() {
	for(FastTreeStructure::iterator it = m_fastInnerTrees.begin(); it != m_fastInnerTrees.end(); ++it){
		for(FastTreeInnerStructure::iterator itInner = it->begin(); itInner != it->end(); ++itInner){
			SAVE_DELETE(*itInner); // nullptr can also be deleted
		}
	}
	for(SmallTreeStructure::iterator it = m_smallInnerTrees.begin(); it != m_smallInnerTrees.end(); ++it){
		for(SmallTreeInnerStructure::iterator itInner = it->begin(); itInner != it->end(); ++itInner){
			SAVE_DELETE(itInner->second); 
		}
	}
}

void BigDynamicDecisionTree::train(int amountOfUsedDims,
		RandomNumberGeneratorForDT& generator){
	if(m_fastInnerTrees.size() > 0 && m_fastInnerTrees[0][0] != nullptr){ // in the case of a retraining that all trees are removed
		for(FastTreeStructure::iterator it = m_fastInnerTrees.begin(); it != m_fastInnerTrees.end(); ++it){
			for(FastTreeInnerStructure::iterator itInner = it->begin(); itInner != it->end(); ++itInner){
				SAVE_DELETE(*itInner); // nullptr can also be deleted
			}
		}
		for(SmallTreeStructure::iterator it = m_smallInnerTrees.begin(); it != m_smallInnerTrees.end(); ++it){
			for(SmallTreeInnerStructure::iterator itInner = it->begin(); itInner != it->end(); ++itInner){
				SAVE_DELETE(itInner->second);
			}
			it->clear(); // clear map
		}
	}
	const unsigned int neededPointsForNewTree = 2;
//	const unsigned int leavesPerTree = pow(2, m_depthPerLayer);
	unsigned int rootsForTreesInThisLayer = 1;
	unsigned int depthInTheFatherLayer = m_depthPerLayer;
	for(unsigned int iTreeLayer = 0; iTreeLayer < m_fastInnerTrees.size(); ++iTreeLayer){
//		printOnScreen("layer: " << iTreeLayer);
		bool saveDataPositions = true;  // only in the last layer the data positions don't need to be saved
		unsigned int depthInThisLayer = m_depthPerLayer;
		if(iTreeLayer + 1 == m_fastInnerTrees.size() && m_smallInnerTrees.size() == 0){ // is the last layer
			saveDataPositions = false;
			depthInThisLayer = m_maxDepth - m_fastInnerTrees.size() * m_depthPerLayer;
		}
		printOnScreen("iTreeLayer: " << iTreeLayer);
		const unsigned int leavesForTreesInThisLayer = pow(2, depthInThisLayer); // amount of leaves of one of the layertrees
		if(iTreeLayer == 0){
			// first tree
			m_fastInnerTrees[iTreeLayer][0] = new DynamicDecisionTree(m_storage, m_depthPerLayer, m_amountOfClasses);
			const bool ret = m_fastInnerTrees[iTreeLayer][0]->train(amountOfUsedDims, generator, 0, saveDataPositions);
			if(!ret){
				printError("The first split could not be performed!");
				return;
			}
		}else{
			bool foundAtLeastOneChild = false;
			const unsigned int leavesForTreesInTheFatherLayer = pow(2, depthInTheFatherLayer); // amount of leaves of one of the layer trees
			const unsigned int amountOfChildrenInThisLayer = rootsForTreesInThisLayer * leavesForTreesInThisLayer;
			m_fastInnerTrees[iTreeLayer].resize(amountOfChildrenInThisLayer);
			std::fill(m_fastInnerTrees[iTreeLayer].begin(), m_fastInnerTrees[iTreeLayer].end(), nullptr); // set all values to null
			for(unsigned int iRootId = 0; iRootId < rootsForTreesInThisLayer; ++iRootId){
//				printOnScreen("Rootid: " << iRootId);
				if(m_fastInnerTrees[iTreeLayer - 1][iRootId] != nullptr){ // if the father is not a nullpointer
					std::vector<std::vector<int> >& dataPositions = *m_fastInnerTrees[iTreeLayer - 1][iRootId]->getDataPositions();
					for(unsigned int iChild = 0; iChild < leavesForTreesInThisLayer; ++iChild){
//						printOnScreen("size: " << dataPositions.size() << ", leaves: " << leavesForTreesInTheFatherLayer
//									<< ", ichild: " << iChild << ", desired amount: " << leavesForTreesInThisLayer << ", root: " << iRootId
//									<< "in data: " << dataPositions[leavesForTreesInTheFatherLayer + iChild].size());
						if(dataPositions[leavesForTreesInTheFatherLayer + iChild].size() > neededPointsForNewTree){ // min amount is that half of the leaves are filled which at least one point!
							foundAtLeastOneChild = true;
							const unsigned int iChildIdInLayer = iChild + leavesForTreesInThisLayer * iRootId;
//							printOnScreen("Child: " << iChildIdInLayer);
							m_fastInnerTrees[iTreeLayer][iChildIdInLayer] = new DynamicDecisionTree(m_storage, depthInThisLayer, m_amountOfClasses);
							m_fastInnerTrees[iTreeLayer][iChildIdInLayer]->setUsedDataPositions(&dataPositions[leavesForTreesInTheFatherLayer + iChild]); // set the values of the storage which should be used in this tree
							const bool trained = m_fastInnerTrees[iTreeLayer][iChildIdInLayer]->train(amountOfUsedDims, generator, 0, saveDataPositions);
							m_fastInnerTrees[iTreeLayer][iChildIdInLayer]->setUsedDataPositions(nullptr); // erase pointer to used dataPositions
							if(!trained){
								SAVE_DELETE(m_fastInnerTrees[iTreeLayer][iChildIdInLayer]);
							}
						}
					}
				}
			}
			if(!foundAtLeastOneChild){
				return; // no need for inserting anything in the m_smallerInnerTrees
			}
			rootsForTreesInThisLayer = amountOfChildrenInThisLayer; // amount of children nodes in this level
			depthInTheFatherLayer = depthInThisLayer;
		}
	}
//	printOnScreen("AmountOfFathernodes: " << rootsForTreesInThisLayer);
	for(unsigned int iTreeSmallLayer = 0; iTreeSmallLayer < m_smallInnerTrees.size(); ++iTreeSmallLayer){
		bool saveDataPositions = true;  // only in the last layer the data positions don't need to be saved
		unsigned int depthInThisLayer = m_depthPerLayer;
		if(iTreeSmallLayer + 1 == m_smallInnerTrees.size()){ // is the last layer
			saveDataPositions = false;
			depthInThisLayer = m_maxDepth - (m_fastInnerTrees.size() + m_smallInnerTrees.size()) * m_depthPerLayer;
		}
		const unsigned int leavesForTreesInThisLayer = pow(2, depthInThisLayer); // amount of leaves of one of the layertrees
		bool foundAtLeastOneChild = false;
		const unsigned int leavesForTreesInTheFatherLayer = pow(2, depthInTheFatherLayer); // amount of leaves of one of the father layertrees
		const unsigned int amountOfChildrenInThisLayer = rootsForTreesInThisLayer * leavesForTreesInThisLayer;
		SmallTreeInnerStructure& actSmallInnerTreeStructure = m_smallInnerTrees[iTreeSmallLayer];
		SmallTreeInnerStructure::iterator it = actSmallInnerTreeStructure.end();
		if(iTreeSmallLayer == 0){
			// go over all of the last layer of the fast tree
			for(unsigned int iRootId = 0; iRootId < rootsForTreesInThisLayer; ++iRootId){
				trainChildrenForRoot(m_fastInnerTrees.back()[iRootId], it, actSmallInnerTreeStructure,
						depthInThisLayer, leavesForTreesInThisLayer, iRootId,
						leavesForTreesInTheFatherLayer, neededPointsForNewTree,
						amountOfUsedDims, generator, saveDataPositions, foundAtLeastOneChild);
			}

		}else{
			// just iterate over the used parents, avoid all other
			for(SmallTreeInnerStructure::const_iterator itRoot = m_smallInnerTrees[iTreeSmallLayer - 1].begin(); itRoot != m_smallInnerTrees[iTreeSmallLayer - 1].end(); ++itRoot){
				trainChildrenForRoot(itRoot->second, it, actSmallInnerTreeStructure,
						depthInThisLayer, leavesForTreesInThisLayer, itRoot->first,
						leavesForTreesInTheFatherLayer, neededPointsForNewTree,
						amountOfUsedDims, generator, saveDataPositions, foundAtLeastOneChild);
			}
		}
		if(!foundAtLeastOneChild){
			break;
		}
		rootsForTreesInThisLayer = amountOfChildrenInThisLayer; // amount of children nodes in this level
		depthInTheFatherLayer = depthInThisLayer;
	}
//	int amountOfUsedTrees = 0;
//	for(unsigned int iTreeLayer = 0; iTreeLayer < m_fastInnerTrees.size(); ++iTreeLayer){
//		for(unsigned int iChild = 0; iChild < m_fastInnerTrees[iTreeLayer].size(); ++iChild){
//			if(m_fastInnerTrees[iTreeLayer][iChild] != nullptr){
//				++amountOfUsedTrees;
//			}
//		}
//	}
//	for(SmallTreeStructure::iterator it = m_smallInnerTrees.begin(); it != m_smallInnerTrees.end(); ++it){
//		for(SmallTreeInnerStructure::iterator itInner = it->begin(); itInner != it->end(); ++itInner){
//			if(itInner->second != nullptr){
//				++amountOfUsedTrees;
//			}
//		}
//	}
//	printOnScreen(amountOfUsedTrees);
	for(unsigned int iTreeLayer = 0; iTreeLayer < m_fastInnerTrees.size(); ++iTreeLayer){
		for(unsigned int iChild = 0; iChild < m_fastInnerTrees[iTreeLayer].size(); ++iChild){
			if(m_fastInnerTrees[iTreeLayer][iChild] != nullptr){
				m_fastInnerTrees[iTreeLayer][iChild]->deleteDataPositions();
			}
		}
	}
}

void BigDynamicDecisionTree::trainChildrenForRoot(DynamicDecisionTree* root, SmallTreeInnerStructure::iterator& it, SmallTreeInnerStructure& actSmallInnerTreeStructure,
		const unsigned int depthInThisLayer, const unsigned int leavesForTreesInThisLayer, const unsigned int iRootId,
		const unsigned int leavesForTreesInTheFatherLayer, const unsigned int neededPointsForNewTree,
		const int amountOfUsedDims, RandomNumberGeneratorForDT& generator, const bool saveDataPositions, bool& foundAtLeastOneChild){
	if(root != nullptr){ // if the father is not a nullpointer
		std::vector<std::vector<int> >& dataPositions = *root->getDataPositions();
		for(unsigned int iChild = 0; iChild < leavesForTreesInThisLayer; ++iChild){
			if(dataPositions[leavesForTreesInTheFatherLayer + iChild].size() > neededPointsForNewTree){ // min amount is that half of the leaves are filled which at least one point!
				const unsigned int iChildIdInLayer = iChild + leavesForTreesInThisLayer * iRootId;
				if(it != actSmallInnerTreeStructure.end() && false){	// it is given here to hint the position were it should be added
					it = actSmallInnerTreeStructure.insert(it, SmallTreeInnerPair(iChildIdInLayer, new DynamicDecisionTree(m_storage, depthInThisLayer, m_amountOfClasses)));
				}else{
					actSmallInnerTreeStructure.insert(SmallTreeInnerPair(iChildIdInLayer, new DynamicDecisionTree(m_storage, depthInThisLayer, m_amountOfClasses)));
					it = actSmallInnerTreeStructure.find(iChildIdInLayer);
				}
				if(it != actSmallInnerTreeStructure.end()){
					it->second->setUsedDataPositions(&dataPositions[leavesForTreesInTheFatherLayer + iChild]); // set the values of the storage which should be used in this tree
					const bool trained = it->second->train(amountOfUsedDims, generator, 0, saveDataPositions);
					if(!trained){
						SAVE_DELETE(it->second);
						actSmallInnerTreeStructure.erase(it);
					}else{
						foundAtLeastOneChild = true;
					}
				}else{
					printError("This should never happen!");
				}
			}
		}
	}
}


int BigDynamicDecisionTree::predict(const DataPoint& point) const{
	if(m_fastInnerTrees[0][0] != nullptr){
		int winningNode = 1;
		int iFatherId = 0; // in the first layer the used node is always the first one
		unsigned int result = m_fastInnerTrees[0][iFatherId]->predict(point, winningNode);
		unsigned int amountOfInternalNodes = pow(2, m_depthPerLayer);
		winningNode -= amountOfInternalNodes; // the first layer always is m_depthPerLayer high
		unsigned int iChildInLayer = 1;
		for(unsigned int iTreeLayer = 1; iTreeLayer < m_fastInnerTrees.size(); ++iTreeLayer){
			unsigned int depthForThisLayer = m_depthPerLayer;
			if(iTreeLayer +1 == m_fastInnerTrees.size() && m_smallInnerTrees.size() == 0){
				depthForThisLayer = m_maxDepth - m_fastInnerTrees.size() * m_depthPerLayer;
			}
			amountOfInternalNodes = pow(2, depthForThisLayer);
			iChildInLayer = winningNode + depthForThisLayer * iFatherId;
			iFatherId = iChildInLayer; // update early is not used anymore
//			printOnScreen("iChild: " << iChildInLayer << ", size: " << m_fastInnerTrees[iTreeLayer].size() << " in layer: " << iTreeLayer);
			if(iChildInLayer < m_fastInnerTrees[iTreeLayer].size()){
				if(m_fastInnerTrees[iTreeLayer][iChildInLayer] != nullptr){
					result = m_fastInnerTrees[iTreeLayer][iChildInLayer]->predict(point, winningNode);
					winningNode -= amountOfInternalNodes; // on the same level as the calc of iChildInLayer
				}else{
//					printError("should not happen");
					return result;
				}
			}else{
				return result;
			}
		}
		for(unsigned int iTreeSmallLayer = 0; iTreeSmallLayer < m_smallInnerTrees.size(); ++iTreeSmallLayer){
			unsigned int depthForThisLayer = m_depthPerLayer;
			if(iTreeSmallLayer + 1 == m_smallInnerTrees.size()){
				depthForThisLayer = m_maxDepth - (m_fastInnerTrees.size() + m_smallInnerTrees.size()) * m_depthPerLayer;
			}
			amountOfInternalNodes = pow(2, depthForThisLayer);
			iChildInLayer = winningNode + depthForThisLayer * iFatherId;
			iFatherId = iChildInLayer; // update early is not used anymore
			SmallTreeInnerStructure::const_iterator itChild = m_smallInnerTrees[iTreeSmallLayer].find(iChildInLayer);
			if(itChild != m_smallInnerTrees[iTreeSmallLayer].end()){
				result = itChild->second->predict(point, winningNode);
				winningNode -= amountOfInternalNodes; // on the same level as the calc of iChildInLayer
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
