/*
 * BigDynamicDecisionTree.cc
 *
 *  Created on: 07.01.2017
 *      Author: Max
 */

#include "BigDynamicDecisionTree.h"
#include "../Base/Settings.h"
#include<memory>

BigDynamicDecisionTree::BigDynamicDecisionTree(OnlineStorage<ClassPoint*>& storage, const unsigned int maxDepth, const unsigned int amountOfClasses, const int layerAmount, const int layerAmountForFast):
	m_storage(storage),
	m_maxDepth(maxDepth),
	m_amountOfClasses(amountOfClasses),
	m_depthPerLayer(0),
	m_usedMemory(0){ // 16 for ints, 24 for the pointer
	int amountOfLayers = layerAmount;
	if(layerAmount < 1){ // 0 and -1, ...
		Settings::getValue("OnlineRandomForest.layerAmountOfBigDDT", amountOfLayers);
	}
	const unsigned int amountForFast = std::min(std::max(layerAmountForFast, 2), amountOfLayers);
	const unsigned int amountForSmall = std::max(amountOfLayers - amountForFast, 0u);
	prepareForSetting(maxDepth, m_amountOfClasses, amountOfLayers, amountForFast, amountForSmall);
}

BigDynamicDecisionTree::BigDynamicDecisionTree(OnlineStorage<ClassPoint*>& storage):
		m_storage(storage), m_maxDepth(0), m_amountOfClasses(0), m_depthPerLayer(0), m_usedMemory(0){
}


BigDynamicDecisionTree::~BigDynamicDecisionTree(){
}

// fill empty tree
void BigDynamicDecisionTree::prepareForSetting(const unsigned int maxDepth, const unsigned int amountOfClasses, const unsigned int amountOfLayers, const unsigned int amountForFast, const unsigned int amountForSmall){
	if(maxDepth > 0 && maxDepth < 50 && maxDepth / amountOfLayers <= 20){
		overwriteConst(m_maxDepth, maxDepth);
		if(m_maxDepth % amountOfLayers == 0){ // that is the easy case
			m_depthPerLayer = m_maxDepth / amountOfLayers;
		}else{
			// we can adjust the height of the last layer
			const unsigned int higherHeight = std::ceil(m_maxDepth/ (double) amountOfLayers); // use bigger depth for all layers
			const unsigned int lowerHeight = m_maxDepth / (double) amountOfLayers; // truncs by its own
			// now take the values which minimizes the height of the last layer:
			if(m_maxDepth - (higherHeight * (amountOfLayers - 1)) == 0){
				m_depthPerLayer = lowerHeight;
			}else if(m_maxDepth - (higherHeight * (amountOfLayers - 1)) < m_maxDepth - (lowerHeight * (amountOfLayers - 1))){
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
		overwriteConst(m_amountOfClasses, amountOfClasses);
	}else{
		printError("The empty tree constructor was not called before!");
	}
}

void BigDynamicDecisionTree::train(const unsigned int amountOfUsedDims,
		RandomNumberGeneratorForDT& generator){
	m_usedMemory = 40 + (m_fastInnerTrees.size() + m_smallInnerTrees.size()) * 8; // 40 fix values, 8 for the first pointer of the layer
	if(m_fastInnerTrees.size() > 0 && m_fastInnerTrees[0][0] != nullptr){ // in the case of a retraining that all trees are removed
		for(FastTreeStructure::iterator it = m_fastInnerTrees.begin(); it != m_fastInnerTrees.end(); ++it){
			it->clear(); // clear map
		}
		for(SmallTreeStructure::iterator it = m_smallInnerTrees.begin(); it != m_smallInnerTrees.end(); ++it){
			it->clear(); // clear map
		}
	}
//	const unsigned int leavesPerTree = pow2(m_depthPerLayer);
	unsigned int amountOfRootsInTheFatherLayer = 1;
	unsigned int depthInTheFatherLayer = m_depthPerLayer;
	for(unsigned int iTreeLayer = 0; iTreeLayer < m_fastInnerTrees.size(); ++iTreeLayer){
		bool saveDataPositions = true;  // only in the last layer the data positions don't need to be saved
		unsigned int depthInThisLayer = m_depthPerLayer;
		if(iTreeLayer + 1 == m_fastInnerTrees.size() && m_smallInnerTrees.size() == 0){ // is the last layer
			printOnScreen("last layer: " << iTreeLayer);
			saveDataPositions = false;
			depthInThisLayer = m_maxDepth - (m_fastInnerTrees.size() - 1) * m_depthPerLayer;
		}

		printOnScreen("layer: " << iTreeLayer << ", has depth: " << depthInThisLayer << ", depth in father layer: " << depthInTheFatherLayer);
		if(depthInThisLayer == 0){
			printError("This should not happen!");
		}
//		printOnScreen("iTreeLayer: " << iTreeLayer);
//		const unsigned int leavesForTreesInThisLayer = pow2(depthInThisLayer); // amount of leaves of one of the layertrees
		if(iTreeLayer == 0){
			// first tree
//			printOnScreen("iTreeLayer: " << iTreeLayer << ", depthInThisLayer: " << depthInThisLayer);
			m_fastInnerTrees[0][0] = std::make_unique<DynamicDecisionTree>(m_storage, depthInThisLayer, m_amountOfClasses);
			const bool ret = m_fastInnerTrees[0][0]->train(amountOfUsedDims, generator, 0, saveDataPositions);
			if(!ret){
				printError("The first split could not be performed!");
				return;
			}
//			auto firstTree = m_fastInnerTrees[0][0];
//			printOnScreen("Amount of leaves: " << firstTree->getNrOfLeaves() << ", height: " << depthInThisLayer);
//			auto& leaveData = *firstTree->getDataPositions();
//			for(unsigned int iVal = firstTree->getNrOfLeaves(); iVal < leaveData.size(); ++iVal){
//				printOnScreen("Amount of data: " << leaveData[iVal].size() << ", in: " << iVal);
//			}
		}else{
			bool foundAtLeastOneChild = false;
			const unsigned int leavesForTreesInTheFatherLayer = pow2(depthInTheFatherLayer); // amount of leaves of one of the layer trees
			const unsigned int amountOfRootsInThisLayer = amountOfRootsInTheFatherLayer * leavesForTreesInTheFatherLayer;
			m_fastInnerTrees[iTreeLayer].resize(amountOfRootsInThisLayer);
			m_usedMemory += (MemoryType) amountOfRootsInThisLayer * 8;
			std::fill(m_fastInnerTrees[iTreeLayer].begin(), m_fastInnerTrees[iTreeLayer].end(), nullptr); // set all values to null
			unsigned int counter = 0;
			// walk over all roots in the father layer (in the second there is just one)
			for(unsigned int iRootOfFatherLayerId = 0; iRootOfFatherLayerId < amountOfRootsInTheFatherLayer; ++iRootOfFatherLayerId){
				auto& currentFather = m_fastInnerTrees[iTreeLayer - 1][iRootOfFatherLayerId];
				if(currentFather != nullptr){ // if the father is not a nullpointer
					++counter;
					auto& dataPositions = *currentFather->getDataPositions();
					for(unsigned int iChildId = 0; iChildId < leavesForTreesInTheFatherLayer; ++iChildId){
//						printOnScreen("size: " << dataPositions.size() << ", leaves: " << leavesForTreesInTheFatherLayer
//									<< ", ichild: " << iChild << ", desired amount: " << leavesForTreesInThisLayer << ", root: " << iRootId
//									<< "in data: " << dataPositions[leavesForTreesInTheFatherLayer + iChild].size());
						auto& dataForThisChild = dataPositions[leavesForTreesInTheFatherLayer + iChildId];
						if(shouldNewTreeBeCalculatedFor(dataForThisChild)){ // min amount is that half of the leaves are filled which at least one point!
							foundAtLeastOneChild = true;
							const auto iChildIdInLayer = iChildId + leavesForTreesInTheFatherLayer * iRootOfFatherLayerId;
							auto& currentTree = m_fastInnerTrees[iTreeLayer][iChildIdInLayer];
//							printOnScreen("Child: " << iChildIdInLayer);
							currentTree = std::make_unique<DynamicDecisionTree>(m_storage, depthInThisLayer, m_amountOfClasses);
							currentTree->setUsedDataPositions(&dataForThisChild); // set the values of the storage which should be used in this tree
							const bool trained = currentTree->train(amountOfUsedDims, generator, 0, saveDataPositions);
							currentTree->setUsedDataPositions(nullptr); // erase pointer to used dataPositions
							if(!trained){
								currentTree.release(nullptr);
							}else{
								m_usedMemory += currentTree->getMemSize();
							}
						}
					}
				}
			}
//			printOnScreen("iTreeLayer: " << iTreeLayer << ", depthInThisLayer: " << depthInThisLayer << ", amountOfRootsInThisLayer: " << amountOfRootsInThisLayer << ", leavesForTreesInTheFatherLayer: " << leavesForTreesInTheFatherLayer << ", correct: " << counter / (double) amountOfRootsInTheFatherLayer);
			if(!foundAtLeastOneChild){
				break; // no need for inserting anything in the m_smallerInnerTrees
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
		if(depthInThisLayer == 0){
//			25, 2, 4, 5, 5, 3
			printError("This should not happen: " << m_maxDepth << ", " << m_fastInnerTrees.size() << ", " << m_smallInnerTrees.size() << ", " << m_depthPerLayer
					<< ", " << (m_fastInnerTrees.size() + m_smallInnerTrees.size() - 1) << ", " << iTreeSmallLayer);
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
						depthInThisLayer, iRootId, leavesForTreesInTheFatherLayer, amountOfUsedDims, generator, saveDataPositions,
						foundAtLeastOneChild);
			}

		}else{
			// just iterate over the used parents, avoid all other
			for(SmallTreeInnerStructure::const_iterator itRoot = m_smallInnerTrees[iTreeSmallLayer - 1].begin(); itRoot != m_smallInnerTrees[iTreeSmallLayer - 1].end(); ++itRoot){
//				printOnScreen(iTreeSmallLayer << " in iRootId: " << itRoot->first << ", " << m_smallInnerTrees[iTreeSmallLayer - 1].size() << ", " << actSmallInnerTreeStructure.size());
				trainChildrenForRoot(itRoot->second, it, actSmallInnerTreeStructure,
						depthInThisLayer, itRoot->first, leavesForTreesInTheFatherLayer, amountOfUsedDims, generator, saveDataPositions,
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

void BigDynamicDecisionTree::trainChildrenForRoot(PtrDynamicDecisionTree root, SmallTreeInnerStructure::iterator& it, SmallTreeInnerStructure& actSmallInnerTreeStructure,
		const unsigned int depthInThisLayer, const unsigned int iRootId, const unsigned int leavesForTreesInTheFatherLayer, const int amountOfUsedDims, RandomNumberGeneratorForDT& generator,
		const bool saveDataPositions, bool& foundAtLeastOneChild){
	if(root != nullptr){ // if the father is not a nullpointer
		std::vector<std::vector<unsigned int> >& dataPositions = *root->getDataPositions();
		for(unsigned int iChild = 0; iChild < leavesForTreesInTheFatherLayer; ++iChild){ // there can only be so many children, how the father had children
//			printOnScreen("leavesForTreesInTheFatherLayer: " << leavesForTreesInTheFatherLayer << ", depth for trees in the father layer: " << leavesForTreesInTheFatherLayer
//					<< ", iChild: " << iChild << ", leaves for this layer: " << leavesForTreesInThisLayer << ", size: " << dataPositions.size());
//			printOnScreen("dataPositions[leavesForTreesInTheFatherLayer + iChild].size(): "
//					<< dataPositions[leavesForTreesInTheFatherLayer + iChild].size());
			if(shouldNewTreeBeCalculatedFor(dataPositions[leavesForTreesInTheFatherLayer + iChild])){ // min amount is that half of the leaves are filled which at least one point!
				const unsigned int iChildIdInLayer = iChild + leavesForTreesInTheFatherLayer * iRootId;
				if(it != actSmallInnerTreeStructure.end() && false){	// it is given here to hint the position were it should be added
					it = actSmallInnerTreeStructure.insert(it, SmallTreeInnerPair(iChildIdInLayer, std::make_unique<DynamicDecisionTree>(m_storage, depthInThisLayer, m_amountOfClasses)));
				}else{
					actSmallInnerTreeStructure.insert(SmallTreeInnerPair(iChildIdInLayer, std::make_unique<DynamicDecisionTree>(m_storage, depthInThisLayer, m_amountOfClasses)));
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

bool BigDynamicDecisionTree::shouldNewTreeBeCalculatedFor(std::vector<unsigned int>& dataPositions){
//	const unsigned int neededPointsForNewTree = 2;
//	return dataPositions.size() > neededPointsForNewTree;
	if(dataPositions.size() > 2){
		auto itPos = dataPositions.begin();
		const auto firstClass = m_storage[*itPos]->getLabel();
		++itPos;
		for(; itPos != dataPositions.end(); ++itPos){
			if(firstClass != m_storage[*itPos]->getLabel()){ // check if one of the elements is not equal the first class
				return true;
			}
		}
		return false;
	}
	return false;
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
					return result;
				}
			}else{
				printError("Should not happen");
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
