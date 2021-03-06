/*
 * BigDynamicDecisionTree.cc
 *
 *  Created on: 07.01.2017
 *      Author: Max
 */

#include "BigDynamicDecisionTree.h"
#include "../Base/Settings.h"
#include "GlobalLifeTimeMeasurement.h"

BigDynamicDecisionTree::BigDynamicDecisionTree(OnlineStorage<LabeledVectorX*>& storage, const unsigned int maxDepth,
											   const unsigned int amountOfClasses, const int layerAmount,
											   const int layerAmountForFast,
											   const unsigned int amountOfPointsCheckedPerSplit) :
		m_storage(storage),
		m_maxDepth(maxDepth),
		m_amountOfClasses(amountOfClasses),
		m_amountOfPointsCheckedPerSplit(amountOfPointsCheckedPerSplit),
		m_depthPerLayer(0),
		m_personalId(-1),
		m_usedMemory(0){ // 16 for ints, 24 for the pointer
	auto amountOfLayers = layerAmount;
	if(layerAmount < 1){ // 0 and -1, ...
		Settings::instance().getValue("OnlineRandomForest.layerAmountOfBigDDT", amountOfLayers);
	}
	if(amountOfLayers > 0 && maxDepth > amountOfLayers){
		const auto amountForFast = (unsigned int) std::min(std::max(layerAmountForFast, 2), amountOfLayers);
		const auto amountForSmall = (unsigned int) std::max(amountOfLayers - amountForFast, 0u);
		prepareForSetting(maxDepth, m_amountOfClasses, (const unsigned int) amountOfLayers, amountForFast,
						  amountForSmall);
	}else{
		printError("The amount of layers is: " << amountOfLayers << ", can not be smaller than 1!");
	}
}

BigDynamicDecisionTree::BigDynamicDecisionTree(OnlineStorage<LabeledVectorX*>& storage) :
		m_storage(storage), m_maxDepth(0), m_amountOfClasses(0), m_amountOfPointsCheckedPerSplit(100),
		m_depthPerLayer(0), m_personalId(-1), m_usedMemory(0){
}


// fill empty tree
void BigDynamicDecisionTree::prepareForSetting(const unsigned int maxDepth, const unsigned int amountOfClasses,
											   const unsigned int amountOfLayers, const unsigned int amountForFast,
											   const unsigned int amountForSmall){
	if(maxDepth > 0 && maxDepth < 50 && maxDepth / amountOfLayers <= 20){
		overwriteConst(m_maxDepth, maxDepth);
		if(m_maxDepth % amountOfLayers == 0){ // that is the easy case
			m_depthPerLayer = m_maxDepth / amountOfLayers;
		}else{
			// we can adjust the height of the last layer
			const auto higherHeight = (unsigned int) std::ceil(
					m_maxDepth / (Real) amountOfLayers); // use bigger depth for all layers
			const auto lowerHeight = (unsigned int) (m_maxDepth / (Real) amountOfLayers); // truncs by its own
			// now take the values which minimizes the height of the last layer:
			if(m_maxDepth - (higherHeight * (amountOfLayers - 1)) == 0){
				m_depthPerLayer = lowerHeight;
			}else if(m_maxDepth - (higherHeight * (amountOfLayers - 1)) <
					 m_maxDepth - (lowerHeight * (amountOfLayers - 1))){
				// take the higher height: example:
				// 		amount of layers = 8 and m_maxDepth = 30 -> the height of the layer here is: 2, instead of 6
				m_depthPerLayer = higherHeight;
			}else if(m_maxDepth - (higherHeight * (amountOfLayers - 1)) >
					 m_maxDepth - (lowerHeight * (amountOfLayers - 1))){
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
		}
		overwriteConst(m_amountOfClasses, amountOfClasses);
	}else{
		printError("The empty tree constructor was not called before!");
	}
}

void BigDynamicDecisionTree::train(const unsigned int amountOfUsedDims,
								   RandomNumberGeneratorForDT& generator){
	if(VerboseMode::instance().isVerboseLevelHigher()){
		printLine();
	}
	if(GlobalLifeTimeMeasurement::instance().isUsed()){
		if(m_personalId == -1){
			m_personalId = GlobalLifeTimeMeasurement::instance().addNewTreeId();
		}else{
			GlobalLifeTimeMeasurement::instance().dieTreeId(m_personalId);
			m_personalId = GlobalLifeTimeMeasurement::instance().addNewTreeId();
		}
	}
	StopWatch bigDTTrain;
	// unsigned ints, the Memory Type and the two vectors and the size for the layers
	m_usedMemory = sizeof(unsigned int) * 4 + sizeof(MemoryType) + sizeof(FastTreeStructure) * 2
				   + m_fastInnerTrees.size() * sizeof(FastTreeInnerStructure)
				   + m_smallInnerTrees.size() * sizeof(SmallTreeInnerStructure);
	if(!m_fastInnerTrees.empty() && m_fastInnerTrees[0][0].get() != nullptr){
		// in the case of a retraining that all trees are removed
		for(auto& fastLayer : m_fastInnerTrees){
			fastLayer.erase(fastLayer.begin(), fastLayer.end()); // clear map
		}
		m_fastInnerTrees[0].resize(1);
		for(auto& smallLayer : m_smallInnerTrees){
			smallLayer.erase(smallLayer.begin(), smallLayer.end()); // clear map
		}
	}
	if(VerboseMode::instance().isVerboseLevelHigher()){
		printLine();
	}
//	const unsigned int leavesPerTree = pow2(m_depthPerLayer);
	unsigned int amountOfRootsInTheFatherLayer = 1;
	unsigned int depthInTheFatherLayer = m_depthPerLayer;
	for(unsigned int iTreeLayer = 0; iTreeLayer < m_fastInnerTrees.size(); ++iTreeLayer){
		bool saveDataPositions = true;  // only in the last layer the data positions don't need to be saved
		unsigned int depthInThisLayer = m_depthPerLayer;
		if(iTreeLayer + 1 == m_fastInnerTrees.size() && m_smallInnerTrees.empty()){ // is the last layer
			saveDataPositions = false;
			depthInThisLayer = m_maxDepth - ((unsigned int) m_fastInnerTrees.size() - 1) * m_depthPerLayer;
		}
		if(depthInThisLayer == 0){
			printError("This should not happen!");
		}
		if(iTreeLayer == 0){
			// first tree
			bool wholeStorageIsOneClass = false;
			if(VerboseMode::instance().isVerboseLevelHigher()){
				printLine();
				wholeStorageIsOneClass = true;
				auto itStorage = m_storage.begin();
				const auto firstClass = (*itStorage)->getLabel();
				++itStorage;
				for(; itStorage != m_storage.end(); ++itStorage){
					if(firstClass !=
					   (*itStorage)->getLabel()){ // check if one of the elements is not equal the first class
						wholeStorageIsOneClass = false;
					}
				}
			}
			if(!wholeStorageIsOneClass){
				m_fastInnerTrees[0][0] = std::make_unique<UsedDecisionTree>(m_storage, depthInThisLayer,
																				m_amountOfClasses,
																				m_amountOfPointsCheckedPerSplit);
				const bool ret = m_fastInnerTrees[0][0]->train((dimTypeForDDT) amountOfUsedDims, generator, 0, saveDataPositions);
				m_usedMemory += m_fastInnerTrees[0][0]->getMemSize();
				if(!ret){
					VerboseMode::instance().setVerboseLevel(1);
					printError("The first split could not be performed!");
					m_fastInnerTrees[0][0]->train((dimTypeForDDT) amountOfUsedDims, generator, 0, saveDataPositions);
					return;
				}
			}else{
				printError("The storage does only contain one class no ");
				return;
			}
		}else{
			bool foundAtLeastOneChild = false;
			const unsigned int leavesForTreesInTheFatherLayer = pow2(
					depthInTheFatherLayer); // amount of leaves of one of the layer trees
			const unsigned int amountOfRootsInThisLayer =
					amountOfRootsInTheFatherLayer * leavesForTreesInTheFatherLayer;
			m_fastInnerTrees[iTreeLayer].resize(amountOfRootsInThisLayer);
			m_usedMemory += (MemoryType) amountOfRootsInThisLayer * sizeof(PtrDynamicDecisionTree);
			// set all values to null
			std::fill(m_fastInnerTrees[iTreeLayer].begin(), m_fastInnerTrees[iTreeLayer].end(), nullptr);
			unsigned int counter = 0;
			// walk over all roots in the father layer (in the second there is just one)
			for(unsigned int iRootOfFatherLayerId = 0;
				iRootOfFatherLayerId < amountOfRootsInTheFatherLayer; ++iRootOfFatherLayerId){
				auto& currentFather = m_fastInnerTrees[iTreeLayer - 1][iRootOfFatherLayerId];
				if(currentFather != nullptr){ // if the father is not a nullpointer
					++counter;
					auto& dataPositions = *currentFather->getDataPositions();
					for(unsigned int iChildId = 0; iChildId < leavesForTreesInTheFatherLayer; ++iChildId){
						auto& dataForThisChild = dataPositions[leavesForTreesInTheFatherLayer + iChildId];
						if(shouldNewTreeBeCalculatedFor(dataForThisChild)){
							// min amount is that half of the leaves are filled which at least one point!
							foundAtLeastOneChild = true;
							const auto iChildIdInLayer = iChildId +
									leavesForTreesInTheFatherLayer * iRootOfFatherLayerId;
							auto currentTree = std::make_unique<UsedDecisionTree>(m_storage, depthInThisLayer,
																				 m_amountOfClasses,
																				 m_amountOfPointsCheckedPerSplit);
							// set the values of the storage which should be used in this tree
							currentTree->setUsedDataPositions(&dataForThisChild);
							const bool trained = currentTree->train((dimTypeForDDT) amountOfUsedDims, generator, 0, saveDataPositions);
							currentTree->setUsedDataPositions(nullptr); // erase pointer to used dataPositions
							if(trained){
								m_usedMemory += currentTree->getMemSize();
								m_fastInnerTrees[iTreeLayer][iChildIdInLayer] = std::move(currentTree);
							}
						}
					}
				}
			}
			if(!foundAtLeastOneChild){
				break; // no need for inserting anything in the m_smallerInnerTrees
			}
			amountOfRootsInTheFatherLayer = amountOfRootsInThisLayer; // amount of children nodes in this level
			depthInTheFatherLayer = depthInThisLayer;
		}
	}
	for(unsigned int iTreeSmallLayer = 0; iTreeSmallLayer < m_smallInnerTrees.size(); ++iTreeSmallLayer){
		bool saveDataPositions = true;  // only in the last layer the data positions don't need to be saved
		unsigned int depthInThisLayer = m_depthPerLayer;
		if(iTreeSmallLayer + 1 == m_smallInnerTrees.size()){ // is the last layer
			saveDataPositions = false;
			depthInThisLayer = m_maxDepth - (unsigned int) (m_fastInnerTrees.size() + m_smallInnerTrees.size() - 1) *
											m_depthPerLayer;
		}
		if(depthInThisLayer == 0){
			printError("This should not happen: " << m_maxDepth << ", " << m_fastInnerTrees.size() << ", "
												  << m_smallInnerTrees.size() << ", " << m_depthPerLayer
												  << ", " << (m_fastInnerTrees.size() + m_smallInnerTrees.size() - 1)
												  << ", " << iTreeSmallLayer);
		}
		bool foundAtLeastOneChild = false;
		const auto leavesForTreesInTheFatherLayer = pow2(
				depthInTheFatherLayer); // amount of leaves of one of the father layertrees
		const auto amountOfRootsInThisLayer = amountOfRootsInTheFatherLayer * leavesForTreesInTheFatherLayer;
		auto& actSmallInnerTreeStructure = m_smallInnerTrees[iTreeSmallLayer];
		auto it = actSmallInnerTreeStructure.end();
		if(iTreeSmallLayer == 0){
			// go over all of the last layer of the fast tree
			for(unsigned int iRootId = 0; iRootId < amountOfRootsInTheFatherLayer; ++iRootId){
//				printOnScreen("iRootId: " << iRootId);
				trainChildrenForRoot(m_fastInnerTrees.back()[iRootId].get(), it, actSmallInnerTreeStructure,
									 depthInThisLayer, iRootId, leavesForTreesInTheFatherLayer, amountOfUsedDims,
									 generator, saveDataPositions,
									 foundAtLeastOneChild);
			}

		}else{
			// just iterate over the used parents, avoid all other
			for(auto& root : m_smallInnerTrees[iTreeSmallLayer - 1]){
//			for(auto itRoot = m_smallInnerTrees[iTreeSmallLayer - 1].begin();
//				itRoot != m_smallInnerTrees[iTreeSmallLayer - 1].end(); ++itRoot){
//				printOnScreen(iTreeSmallLayer << " in iRootId: " << root.first << ", " << m_smallInnerTrees[iTreeSmallLayer - 1].size() << ", " << actSmallInnerTreeStructure.size());
				trainChildrenForRoot(root.second.get(), it, actSmallInnerTreeStructure,
									 depthInThisLayer, root.first, leavesForTreesInTheFatherLayer, amountOfUsedDims,
									 generator, saveDataPositions,
									 foundAtLeastOneChild);
			}
		}
		if(!foundAtLeastOneChild){
			break;
		}
		amountOfRootsInTheFatherLayer = amountOfRootsInThisLayer; // amount of children nodes in this level
		depthInTheFatherLayer = depthInThisLayer;
	}
	for(auto& layer : m_fastInnerTrees){
		for(auto& element : layer){
			if(element != nullptr){
				element->deleteDataPositions();
			}
		}
	}
	for(auto& layer : m_smallInnerTrees){
		for(auto& element : layer){
			if(element.second != nullptr){
				element.second->deleteDataPositions();
			}
		}
	}
	GlobalStopWatch<BigDecisionTreeTrain>::instance().recordActTime(bigDTTrain.elapsedSeconds());
}

void BigDynamicDecisionTree::trainChildrenForRoot(UsedDecisionTree* root, SmallTreeInnerStructureIterator& it,
												  SmallTreeInnerStructure& actSmallInnerTreeStructure,
												  const unsigned int depthInThisLayer, const unsigned int iRootId,
												  const unsigned int leavesForTreesInTheFatherLayer,
												  const int amountOfUsedDims, RandomNumberGeneratorForDT& generator,
												  const bool saveDataPositions, bool& foundAtLeastOneChild){
	if(root != nullptr){ // if the father is not a nullpointer
		auto& dataPositions = *root->getDataPositions();
		for(unsigned int iChild = 0; iChild < leavesForTreesInTheFatherLayer; ++iChild){
			// there can only be so many children, how the father had children
			if(shouldNewTreeBeCalculatedFor(dataPositions[leavesForTreesInTheFatherLayer + iChild])){
				// min amount is that half of the leaves are filled which at least one point!
				const unsigned int iChildIdInLayer = iChild + leavesForTreesInTheFatherLayer * iRootId;
				auto currentTree = std::make_unique<UsedDecisionTree>(m_storage, depthInThisLayer, m_amountOfClasses,
																	  m_amountOfPointsCheckedPerSplit);
				currentTree->setUsedDataPositions(&dataPositions[leavesForTreesInTheFatherLayer + iChild]);
				// set the values of the storage which should be used in this tree
				const bool trained = currentTree->train((dimTypeForDDT) amountOfUsedDims, generator, 0,
													   saveDataPositions);
				currentTree->setUsedDataPositions(nullptr);
				if(trained){
					m_usedMemory += currentTree->getMemSize() + sizeof(SmallTreeInnerPair) + sizeof(UsedDecisionTree);
					actSmallInnerTreeStructure.emplace(iChildIdInLayer, std::move(currentTree));
					foundAtLeastOneChild = true;
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
			if(firstClass != m_storage[*itPos]->getLabel()){
				// check if one of the elements is not equal the first class
				return true;
			}
		}
		return false;
	}
	return false;
}

unsigned int BigDynamicDecisionTree::predict(const VectorX& point) const{
	if(m_fastInnerTrees[0][0] != nullptr){
		// the father layer always has the same height, because only the last layer can change the height
		const auto childrenInFatherLayer = pow2(m_depthPerLayer);
		int winningNode = 1;
		unsigned int iChildInLayer(0u);
		auto result = m_fastInnerTrees[0][iChildInLayer]->predict(point, winningNode);
		// convert tree node id into leave id
		auto offsetForChildren = 0u;
		for(unsigned int iTreeLayer = 1; iTreeLayer < m_fastInnerTrees.size(); ++iTreeLayer){
			iChildInLayer = winningNode + offsetForChildren;
			offsetForChildren = iChildInLayer * childrenInFatherLayer;
			if(iChildInLayer < m_fastInnerTrees[iTreeLayer].size()){
				if(m_fastInnerTrees[iTreeLayer][iChildInLayer] != nullptr){
					result = m_fastInnerTrees[iTreeLayer][iChildInLayer]->predict(point, winningNode);
				}else{
					return result;
				}
			}else{
				printError("Should not happen, child: " << iChildInLayer << ", size: "
														<< m_fastInnerTrees[iTreeLayer].size());
				return result;
			}
		}
		for(const auto& smallInnerTree : m_smallInnerTrees){
			iChildInLayer = winningNode + offsetForChildren;
			offsetForChildren = iChildInLayer * childrenInFatherLayer;
			auto itChild = smallInnerTree.find(iChildInLayer);
			if(itChild != smallInnerTree.end()){
				result = itChild->second->predict(point, winningNode);
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
