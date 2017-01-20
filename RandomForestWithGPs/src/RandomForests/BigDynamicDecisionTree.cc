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
	m_amountOfClasses(amountOfClasses),
	m_usedMemory(0){ // 16 for ints, 24 for the pointer
	int amountOfLayers = layerAmount;
	if(layerAmount < 1){ // 0 and -1, ...
		Settings::getValue("OnlineRandomForest.layerAmountOfBigDDT", amountOfLayers);
	}
	const int amountForFast = std::min(std::max(layerAmountForFast, 2), amountOfLayers);
	const int amountForSmall = std::max(amountOfLayers - amountForFast, 0);
	if(m_maxDepth % amountOfLayers == 0){ // that is the easy case
		m_depthPerLayer = m_maxDepth / amountOfLayers;
	}else{
		// we can adjust the height of the last layer
		const unsigned int higherHeight = std::ceil(m_maxDepth/ (double) amountOfLayers); // use bigger depth for all layers
		const unsigned int lowerHeight = m_maxDepth / (double) amountOfLayers; // truncs by its own
		// now take the values which minimizes the height of the last layer:
		if(m_maxDepth - (higherHeight * (amountOfLayers - 1)) < m_maxDepth - (lowerHeight * (amountOfLayers - 1))){
			// take the higher height: example:
			// 		amount of layers = 8 and m_maxDepth = 30 -> the height of the layer here is: 2, instead of 6
			m_depthPerLayer = higherHeight;
		}else if(m_maxDepth - (higherHeight * (amountOfLayers - 1)) > m_maxDepth - (lowerHeight * (amountOfLayers - 1))){
			// take the lower height: example:
			//		amount of layer = 6
			m_depthPerLayer = lowerHeight;
		}else if(higherHeight > lowerHeight){
			m_depthPerLayer = higherHeight;
		}else{
			m_depthPerLayer = lowerHeight;
		}
//		printOnScreen("For depth: " << m_maxDepth << " choosen " << amountOfLayers << " layers with depth of " << m_depthPerLayer);
	}
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

void BigDynamicDecisionTree::train(const unsigned int amountOfUsedDims,
		RandomNumberGeneratorForDT& generator){
	m_usedMemory = 40 + (m_fastInnerTrees.size() + m_smallInnerTrees.size()) * 8; // 40 fix values, 8 for the first pointer of the layer
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
//	const unsigned int leavesPerTree = pow2(m_depthPerLayer);
	unsigned int amountOfRootsInTheFatherLayer = 1;
	unsigned int depthInTheFatherLayer = m_depthPerLayer;
	for(unsigned int iTreeLayer = 0; iTreeLayer < m_fastInnerTrees.size(); ++iTreeLayer){
//		printOnScreen("layer: " << iTreeLayer);
		bool saveDataPositions = true;  // only in the last layer the data positions don't need to be saved
		unsigned int depthInThisLayer = m_depthPerLayer;
		if(iTreeLayer + 1 == m_fastInnerTrees.size() && m_smallInnerTrees.size() == 0){ // is the last layer
//			saveDataPositions = false;
			depthInThisLayer = m_maxDepth - (m_fastInnerTrees.size() - 1) * m_depthPerLayer;
		}
//		printOnScreen("iTreeLayer: " << iTreeLayer);
//		const unsigned int leavesForTreesInThisLayer = pow2(depthInThisLayer); // amount of leaves of one of the layertrees
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
			const unsigned int leavesForTreesInTheFatherLayer = pow2(depthInTheFatherLayer); // amount of leaves of one of the layer trees
			const unsigned int amountOfRootsInThisLayer = amountOfRootsInTheFatherLayer * leavesForTreesInTheFatherLayer;
			m_fastInnerTrees[iTreeLayer].resize(amountOfRootsInThisLayer);
			m_usedMemory += (MemoryType) amountOfRootsInThisLayer * 8;
			std::fill(m_fastInnerTrees[iTreeLayer].begin(), m_fastInnerTrees[iTreeLayer].end(), nullptr); // set all values to null
			for(unsigned int iRootOfFatherLayerId = 0; iRootOfFatherLayerId < amountOfRootsInTheFatherLayer; ++iRootOfFatherLayerId){
//				printOnScreen("Rootid: " << iRootId);
				if(m_fastInnerTrees[iTreeLayer - 1][iRootOfFatherLayerId] != nullptr){ // if the father is not a nullpointer
					std::vector<std::vector<unsigned int> >& dataPositions = *m_fastInnerTrees[iTreeLayer - 1][iRootOfFatherLayerId]->getDataPositions();
					for(unsigned int iChildId = 0; iChildId < leavesForTreesInTheFatherLayer; ++iChildId){
//						printOnScreen("size: " << dataPositions.size() << ", leaves: " << leavesForTreesInTheFatherLayer
//									<< ", ichild: " << iChild << ", desired amount: " << leavesForTreesInThisLayer << ", root: " << iRootId
//									<< "in data: " << dataPositions[leavesForTreesInTheFatherLayer + iChild].size());
						if(dataPositions[leavesForTreesInTheFatherLayer + iChildId].size() > neededPointsForNewTree){ // min amount is that half of the leaves are filled which at least one point!
							foundAtLeastOneChild = true;
							const unsigned int iChildIdInLayer = iChildId + leavesForTreesInTheFatherLayer * iRootOfFatherLayerId;
//							printOnScreen("Child: " << iChildIdInLayer);
							m_fastInnerTrees[iTreeLayer][iChildIdInLayer] = new DynamicDecisionTree(m_storage, depthInThisLayer, m_amountOfClasses);
							m_fastInnerTrees[iTreeLayer][iChildIdInLayer]->setUsedDataPositions(&dataPositions[leavesForTreesInTheFatherLayer + iChildId]); // set the values of the storage which should be used in this tree
							const bool trained = m_fastInnerTrees[iTreeLayer][iChildIdInLayer]->train(amountOfUsedDims, generator, 0, saveDataPositions);
							m_fastInnerTrees[iTreeLayer][iChildIdInLayer]->setUsedDataPositions(nullptr); // erase pointer to used dataPositions
							if(!trained){
								SAVE_DELETE(m_fastInnerTrees[iTreeLayer][iChildIdInLayer]);
							}else{
								m_usedMemory += m_fastInnerTrees[iTreeLayer][iChildIdInLayer]->getMemSize();
							}
						}
					}
				}
			}
			if(!foundAtLeastOneChild){
				return; // no need for inserting anything in the m_smallerInnerTrees
			}
			amountOfRootsInTheFatherLayer = amountOfRootsInThisLayer; // amount of children nodes in this level
			depthInTheFatherLayer = depthInThisLayer;
		}
	}
//	printOnScreen("AmountOfFathernodes: " << rootsForTreesInThisLayer);
	for(unsigned int iTreeSmallLayer = 0; iTreeSmallLayer < m_smallInnerTrees.size(); ++iTreeSmallLayer){
		bool saveDataPositions = true;  // only in the last layer the data positions don't need to be saved
		unsigned int depthInThisLayer = m_depthPerLayer;
		if(iTreeSmallLayer + 1 == m_smallInnerTrees.size()){ // is the last layer
//			saveDataPositions = false;
			depthInThisLayer = m_maxDepth - (m_fastInnerTrees.size() + m_smallInnerTrees.size() - 1) * m_depthPerLayer;
		}
//		const unsigned int leavesForTreesInThisLayer = pow2(depthInThisLayer); // amount of leaves of one of the layertrees
		bool foundAtLeastOneChild = false;
		const unsigned int leavesForTreesInTheFatherLayer = pow2(depthInTheFatherLayer); // amount of leaves of one of the father layertrees
		const unsigned int amountOfRootsInThisLayer = amountOfRootsInTheFatherLayer * leavesForTreesInTheFatherLayer;
		SmallTreeInnerStructure& actSmallInnerTreeStructure = m_smallInnerTrees[iTreeSmallLayer];
		SmallTreeInnerStructure::iterator it = actSmallInnerTreeStructure.end();
//		printOnScreen(iTreeSmallLayer << ", leavesForTreesInTheFatherLayer: " << leavesForTreesInTheFatherLayer
//				<< ", depthInThisLayer: " << depthInThisLayer << ", amountOfChildrenInThisLayer: " << amountOfRootsInThisLayer << ", rootsForTreesInThisLayer: " << amountOfRootsInTheFatherLayer
//				<< ", m_depthPerLayer: " << m_depthPerLayer << ", fast: " << m_fastInnerTrees.size() << ", small: " << m_smallInnerTrees.size());
		if(iTreeSmallLayer == 0){
			// go over all of the last layer of the fast tree
			for(unsigned int iRootId = 0; iRootId < amountOfRootsInTheFatherLayer; ++iRootId){
//				printOnScreen("iRootId: " << iRootId);
				trainChildrenForRoot(m_fastInnerTrees.back()[iRootId], it, actSmallInnerTreeStructure,
						depthInThisLayer, iRootId, leavesForTreesInTheFatherLayer,
						neededPointsForNewTree, amountOfUsedDims, generator, saveDataPositions,
						foundAtLeastOneChild);
			}

		}else{
			// just iterate over the used parents, avoid all other
			for(SmallTreeInnerStructure::const_iterator itRoot = m_smallInnerTrees[iTreeSmallLayer - 1].begin(); itRoot != m_smallInnerTrees[iTreeSmallLayer - 1].end(); ++itRoot){
//				printOnScreen(iTreeSmallLayer << " in iRootId: " << itRoot->first << ", " << m_smallInnerTrees[iTreeSmallLayer - 1].size() << ", " << actSmallInnerTreeStructure.size());
				trainChildrenForRoot(itRoot->second, it, actSmallInnerTreeStructure,
						depthInThisLayer, itRoot->first, leavesForTreesInTheFatherLayer,
						neededPointsForNewTree, amountOfUsedDims, generator, saveDataPositions,
						foundAtLeastOneChild);
			}
		}
		if(!foundAtLeastOneChild){
			break;
		}
		amountOfRootsInTheFatherLayer = amountOfRootsInThisLayer; // amount of children nodes in this level
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
		const unsigned int depthInThisLayer, const unsigned int iRootId, const unsigned int leavesForTreesInTheFatherLayer,
		const unsigned int neededPointsForNewTree, const int amountOfUsedDims, RandomNumberGeneratorForDT& generator,
		const bool saveDataPositions, bool& foundAtLeastOneChild){
	if(root != nullptr){ // if the father is not a nullpointer
		std::vector<std::vector<unsigned int> >& dataPositions = *root->getDataPositions();
		for(unsigned int iChild = 0; iChild < leavesForTreesInTheFatherLayer; ++iChild){ // there can only be so many children, how the father had children
//			printOnScreen("leavesForTreesInTheFatherLayer: " << leavesForTreesInTheFatherLayer << ", depth for trees in the father layer: " << leavesForTreesInTheFatherLayer
//					<< ", iChild: " << iChild << ", leaves for this layer: " << leavesForTreesInThisLayer << ", size: " << dataPositions.size());
//			printOnScreen("dataPositions[leavesForTreesInTheFatherLayer + iChild].size(): "
//					<< dataPositions[leavesForTreesInTheFatherLayer + iChild].size());
			if(dataPositions[leavesForTreesInTheFatherLayer + iChild].size() > neededPointsForNewTree){ // min amount is that half of the leaves are filled which at least one point!
				const unsigned int iChildIdInLayer = iChild + leavesForTreesInTheFatherLayer * iRootId;
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
						m_usedMemory += it->second->getMemSize() + 16; // 16 for each node
						foundAtLeastOneChild = true;
					}
				}else{
					printError("This should never happen!");
				}
			}
		}
	}
}

unsigned int BigDynamicDecisionTree::predict(const DataPoint& point) const{
	if(m_fastInnerTrees[0][0] != nullptr){
		const unsigned int depthForFatherLayer = m_depthPerLayer; // the father layer always has the same height, because only the last layer can change the height
		int winningNode = 1;
		int iFatherId = 0; // in the first layer the used node is always the first one
		unsigned int result = m_fastInnerTrees[0][iFatherId]->predict(point, winningNode);
		unsigned int amountOfInternalNodes = pow2(m_depthPerLayer);
		// convert tree node id into leave id
		winningNode -= amountOfInternalNodes; // the first layer always is m_depthPerLayer high
		unsigned int iChildInLayer = 1;
		for(unsigned int iTreeLayer = 1; iTreeLayer < m_fastInnerTrees.size(); ++iTreeLayer){
			unsigned int depthForThisLayer = m_depthPerLayer;
			if(iTreeLayer + 1 == m_fastInnerTrees.size() && m_smallInnerTrees.size() == 0){
				depthForThisLayer = m_maxDepth - (m_fastInnerTrees.size() - 1) * m_depthPerLayer;
			}
			amountOfInternalNodes = pow2(depthForThisLayer);
			iChildInLayer = winningNode + depthForFatherLayer * iFatherId;
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
			if(iTreeSmallLayer + 1 == m_fastInnerTrees.size()){
				depthForThisLayer = m_maxDepth - (m_fastInnerTrees.size() + m_smallInnerTrees.size() - 1) * m_depthPerLayer;
			}
			amountOfInternalNodes = pow2(depthForThisLayer);
			iChildInLayer = winningNode + depthForFatherLayer * iFatherId;
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
