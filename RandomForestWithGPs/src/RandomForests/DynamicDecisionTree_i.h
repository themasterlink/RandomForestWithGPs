/*
 * DynamicDecisionTree.cc
 *
 *  Created on: 17.10.2016
 *      Author: Max
 */

//#include "DynamicDecisionTree.h"
//#include "../Data/ClassKnowledge.h"

#include "../Data/ClassKnowledge.h"
#include "../Utility/GlobalStopWatch.h"

#ifndef __INCLUDE_DYNAMICDECISIONTREE
#error "Don't include DynamicDecisionTree_i.h directly. Include DynamicDecisionTree.h instead."
#endif

template<typename dimType>
DynamicDecisionTree<dimType>::DynamicDecisionTree(OnlineStorage<LabeledVectorX *> &storage, const unsigned int maxDepth,
										 const unsigned int amountOfClasses, const unsigned int amountOfPointsPerSplit):
		m_storage(storage),
		m_maxDepth((dimType) maxDepth),
		m_maxNodeNr((dimType) pow2(maxDepth + 1) - 1),
		m_maxInternalNodeNr((dimType) pow2(maxDepth) - 1),
		m_amountOfClasses((dimType) amountOfClasses),
		m_amountOfPointsCheckedPerSplit((decltype(m_amountOfPointsCheckedPerSplit)) amountOfPointsPerSplit),
		m_splitValues(m_maxInternalNodeNr + 1), // + 1 -> no use of the first element
		m_splitDim(m_maxInternalNodeNr + 1, (dimType) NodeType::NODE_IS_NOT_USED),
		m_labelsOfWinningClassesInLeaves(m_maxInternalNodeNr + 1, (dimType) UNDEF_CLASS_LABEL),
		m_dataPositions(nullptr),
		m_useOnlyThisDataPositions(nullptr){
	static_assert(m_maxAmountOfElements > UNDEF_CLASS_LABEL, "The undef class label is higher than the highest value allowed in the DDT!");
	if(maxDepth >= sizeof(dimType) * 8 || amountOfClasses >= m_maxAmountOfElements){
		printErrorAndQuit("For this training set the amount of classes or dimension is higher than "
						   << m_maxAmountOfElements << ", which is not supported here!");
	}
	if(m_amountOfPointsCheckedPerSplit == 0){
		printError("This tree can not be trained!");
	}
	if(m_maxDepth <= 0 || m_maxDepth >= 28){
		printError("This height is not supported here: " << m_maxDepth);
	}
//	printOnScreen("Size is: " << (m_splitDim.size() * sizeof(int) + m_splitValues.size() * sizeof(Real) + m_labelsOfWinningClassesInLeaves.size() * sizeof(int)));
}


// construct empty tree
template<typename dimType>
DynamicDecisionTree<dimType>::DynamicDecisionTree(OnlineStorage<LabeledVectorX*>& storage):
		DynamicDecisionTree(storage, 1, ClassKnowledge::instance().amountOfClasses(), 100){
}

// fill empty tree
template<typename dimType>
void DynamicDecisionTree<dimType>::prepareForSetting(const unsigned int maxDepth, const unsigned int amountOfClasses){
	if(maxDepth >= sizeof(dimType) * 8 || amountOfClasses >= m_maxAmountOfElements){
		printErrorAndQuit("For this training set the amount of classes or dimension is higher than "
						   << m_maxAmountOfElements << ", which is not supported here!");
	}
	if(m_maxDepth == 1 && maxDepth > 0 && maxDepth < 28){
		overwriteConst(m_maxDepth, (dimType) maxDepth);
		overwriteConst(m_amountOfClasses, (dimType) amountOfClasses);
		overwriteConst(m_maxNodeNr, (dimType) (pow2(maxDepth + 1) - 1));
		overwriteConst(m_maxInternalNodeNr, (dimType) (pow2(maxDepth) - 1));
	}else{
		printError("The empty tree constructor was not called before!");
	}
}

// copy construct
template<typename dimType>
DynamicDecisionTree<dimType>::DynamicDecisionTree(const DynamicDecisionTree& tree):
		DynamicDecisionTreeInterface(tree),
		m_storage(tree.m_storage),
		m_maxDepth(tree.m_maxDepth),
		m_maxNodeNr(tree.m_maxNodeNr),
		m_maxInternalNodeNr(tree.m_maxInternalNodeNr),
		m_amountOfClasses(tree.m_amountOfClasses),
		m_amountOfPointsCheckedPerSplit(tree.m_amountOfPointsCheckedPerSplit),
		m_splitValues(tree.m_splitValues),
		m_splitDim(tree.m_splitDim),
		m_labelsOfWinningClassesInLeaves(tree.m_labelsOfWinningClassesInLeaves),
		m_dataPositions(nullptr),
		m_useOnlyThisDataPositions(nullptr){
}

template<typename dimType>
DynamicDecisionTree<dimType>::~DynamicDecisionTree(){
	m_useOnlyThisDataPositions = nullptr; // never delete this pointer! (does not belong to this object)
	deleteDataPositions();
}

template<typename dimType>
bool DynamicDecisionTree<dimType>::train(dimType amountOfUsedDims, RandomNumberGeneratorForDT &generator,
										 const dimType tryCounter, const bool saveDataPosition){
	StopWatch dyTrain;
	if(m_splitDim[1] != NodeType::NODE_IS_NOT_USED || m_splitDim[1] != NodeType::NODE_CAN_BE_USED){
		// reset training
		std::fill(m_splitDim.begin(), m_splitDim.end(), NodeType::NODE_IS_NOT_USED);
		std::fill(m_labelsOfWinningClassesInLeaves.begin(), m_labelsOfWinningClassesInLeaves.end(), UNDEF_CLASS_LABEL);
	}
	std::vector<dimType> usedDims(amountOfUsedDims,-1);
	if(amountOfUsedDims == ClassKnowledge::instance().amountOfDims()){
		for(dimType i = 0; i < amountOfUsedDims; ++i){
			usedDims[i] = i;
		}
	}else{
		generator.setRandForDim(ClassKnowledge::instance().amountOfDims() - 1);
		for(dimType i = 0; i < amountOfUsedDims; ++i){
			bool doAgain = false;
			int counter = 0;
			do{
				doAgain = false;
				const int randNr = generator.getRandDim(); // generates number in the range 0...data.rows() - 1;
				if(!generator.useDim(randNr)){
					++counter;
					doAgain = true;
					continue;
				}else if(counter > 200){
					// no other dimension found which can be used
					amountOfUsedDims = i;
					break;
				}
				for(unsigned int j = 0; j < i; ++j){
					if(randNr == usedDims[j]){
						doAgain = true;
					}
					break;
				}
				if(!doAgain){
					usedDims[i] = randNr;
				}
				++counter;
			}while(doAgain);
		}
		generator.setRandForDim(amountOfUsedDims - 1);
	}
	m_splitDim[1] = NodeType::NODE_CAN_BE_USED; // init the root value
	std::vector<unsigned int> leftHisto(m_amountOfClasses), rightHisto(m_amountOfClasses);
	m_dataPositions = new std::vector<DataPositions>(m_maxNodeNr + 1, std::vector<unsigned int>());
	std::vector<DataPositions>& dataPosition(*m_dataPositions);
//	int breakPoint = 3; // 1 + 2 -> should have at least 2 layers
//	int actLayer = 2;
//	bool atLeastPerformedOneSplit = false;
	//  1
	// 2 3
	if(m_useOnlyThisDataPositions == nullptr){
		auto& dataPos = dataPosition[1];
		const bool useShortenSpanOfStorage = generator.useRealOnlineUpdate() && !m_storage.isInPoolMode();
		const auto startPos = useShortenSpanOfStorage ? m_storage.getLastUpdateIndex() : 0;
		const auto size = useShortenSpanOfStorage ? m_storage.getAmountOfNew() : m_storage.size();
		bool useWholeDataSet = false;
		if(generator.useWholeDataSet()){
			useWholeDataSet = true;
		}else{
			const auto& baggingInfo = generator.getBaggingInfo();
			if(baggingInfo.useStepSize()){
				const auto amountOfPoints = (unsigned int) (size / (baggingInfo.m_stepSizeOverData * 0.375));
				dataPos.reserve(amountOfPoints);
			}else if(baggingInfo.useTotalAmountOfPoints()){
				if(size > baggingInfo.m_totalUseOfData){
					if(generator.isRandStepOverStorageUsed()){ // the step value is bigger than 1
						const auto steps = (size / baggingInfo.m_totalUseOfData) * 2;
						dataPos.reserve(baggingInfo.m_totalUseOfData * 2);
					}else{ // is just 1
						useWholeDataSet = true; // is faster that way
					}
				}else{
					useWholeDataSet = true;
				}
			}else{
				printError("This type is unknown here!");
			}
			if(!useWholeDataSet){
				// -1 that the first value in the storage is used too
				for(unsigned int i = startPos + generator.getRandStepOverStorage() - 1;
					i < m_storage.size(); i += generator.getRandStepOverStorage()){
					dataPos.push_back(i);
				}
			}
		}
		if(useWholeDataSet){
			dataPos.resize(size);
			std::iota(dataPos.begin(), dataPos.end(), startPos);
		}
	}
//	else{ // no need take ref to
//		dataPosition[1].insert(dataPosition[1].end(), m_useOnlyThisDataPositions->begin(), m_useOnlyThisDataPositions->end());
//	}
	const auto amountOfTriedDims = (unsigned int) std::min(100, std::max((int) (usedDims.size() * 0.1), 2));
	for(int iActNode = 1; iActNode < (int) m_maxInternalNodeNr + 1; ++iActNode){ // first element is not used!
		if(m_splitDim[iActNode] == NodeType::NODE_IS_NOT_USED){ // checks if node contains data or not
			continue; // if node is not used, go to next node, if node can be used process it
		}
		DataPositions& actDataPos = (m_useOnlyThisDataPositions != nullptr && iActNode == 1)
									? (*m_useOnlyThisDataPositions) : (dataPosition[iActNode]);
//		if(iActNode == breakPoint){ // check if early breaking is possible, check is performed always at the start of a layer
//			if(!atLeastPerformedOneSplit){
//				break;
//			}
//			atLeastPerformedOneSplit = false;
//			breakPoint += pow2(actLayer); // first iteration from 3 -> 7, 7 -> 15, 15 -> 31
//			++actLayer;
//		}
		// calc actual nodesararg
		// calc split value for each node
		// choose dimension for split
		const auto actDataSize = actDataPos.size();
		dimType randDim;
		Real minDimValue = REAL_MAX, maxDimValue = NEG_REAL_MAX;
		// try different dimension and find one where the points have a difference
		for(unsigned int i = 0; i < amountOfTriedDims && actDataSize > 0; ++i){
			randDim = usedDims[generator.getRandDim()]; // generates number in the range 0...amountOfUsedDims - 1
			minDimValue = REAL_MAX;
			maxDimValue = NEG_REAL_MAX;
			for(auto& pos : actDataPos){
				if(m_storage[pos]->coeff(randDim) > maxDimValue){
					maxDimValue = m_storage[pos]->coeff(randDim);
				}
				if(m_storage[pos]->coeff(randDim) < minDimValue){
					minDimValue = m_storage[pos]->coeff(randDim);
				}
			}
			if(minDimValue < maxDimValue){ // there is a difference in this dimension
				break;
			}
		}
		if(minDimValue >= maxDimValue){ // splitting impossible
			//				printError("No dimension was found, where a split could be performed!");
			m_splitDim[iActNode] = NodeType::NODE_IS_NOT_USED; // do not split here
			if(iActNode == 1){
				m_splitDim[iActNode] = NodeType::NODE_CAN_BE_USED; // there should be a split
			}
			continue;
		}
		generator.setMinAndMaxForSplitInDim((unsigned int) randDim, minDimValue, maxDimValue);

		Real maxScoreElementValue = 0;
		Real actScore = NEG_REAL_MAX; // TODO check magic number
		std::sort(actDataPos.begin(), actDataPos.end(),
				  [this, &randDim](const auto& a, const auto& b) -> bool
				  { return m_storage[a]->coeff(randDim) < m_storage[b]->coeff(randDim); });
		const auto amountOfUsedData = generator.getRandAmountOfUsedData();
		for(int j = 0; j < amountOfUsedData; ++j){ // amount of checks for a specified split
//			const int randElementId = generator.getRandNextDataEle();
//			const Real usedValue = (*m_storage[usedNode])[usedDim];
			const Real usedValue = generator.getRandSplitValueInDim(randDim);
			const Real score = trySplitFor(usedValue, randDim,
					actDataPos, leftHisto, rightHisto, generator);
//			if(iActNode == 1 && randDim == 0){
//				printOnScreen(usedValue <<"," << score);
//			}
			if(score > actScore){
				actScore = score;
				maxScoreElementValue = usedValue;
			}
		}
		// save actual split
		m_splitValues[iActNode] = maxScoreElementValue;//(Real) (*m_storage[maxScoreElement])[randDim];
		m_splitDim[iActNode] = randDim;
		// apply split to data
		const int leftPos = iActNode * 2, rightPos = iActNode * 2 + 1;
		auto& leftDataPos = dataPosition[leftPos];
		leftDataPos.reserve(actDataPos.size());
		auto& rightDataPos = dataPosition[rightPos];
		rightDataPos.reserve(actDataPos.size());
		for(const auto& pos : actDataPos){
			if(m_storage[pos]->coeff(randDim) >= m_splitValues[iActNode]){ // TODO check >= like below  or only >
				rightDataPos.push_back(pos);
			}else{
				leftDataPos.push_back(pos);
			}
		}

		/*std::cout << "i: " << iActNode << std::endl;
			 std::cout << "length: " << actDataPos.size() << std::endl;
			 std::cout << "Found data left  " << foundDataLeft << std::endl;
			 std::cout << "Found data right " << foundDataRight << std::endl;*/
		if(rightDataPos.empty() || leftDataPos.empty()){
			// split is not needed
//			dataPosition[leftPos].clear();
//			dataPosition[rightPos].clear();
			m_splitDim[iActNode] = NodeType::NODE_IS_NOT_USED; // do not split here
			if(iActNode == 1){
				m_splitDim[iActNode] = NodeType::NODE_CAN_BE_USED; // there should be a split
			}
		}else{
//			atLeastPerformedOneSplit = true;
//			actDataPos.clear();
			// set the use flag for children:
			if(rightPos < (int) m_maxInternalNodeNr + 1){ // if right is not a leave, than left is too -> just control one
				m_splitDim[leftPos] = leftDataPos.empty()   ? NodeType::NODE_IS_NOT_USED : NodeType::NODE_CAN_BE_USED;
				m_splitDim[rightPos] = rightDataPos.empty() ? NodeType::NODE_IS_NOT_USED : NodeType::NODE_CAN_BE_USED;
			}
		}
	}
	const auto leafAmount = getNrOfLeaves();
	const auto offset = leafAmount; // pow2(maxDepth - 1)
	std::vector<unsigned int> histo(m_amountOfClasses, 0u);
	for(unsigned int i = 0; i < leafAmount; ++i){
		auto lastValue = i + offset;
		unsigned int actNode = lastValue / 2;
		while(m_splitDim[actNode] == NodeType::NODE_IS_NOT_USED && actNode > 1){
			lastValue = actNode; // save correct child
			actNode /= 2; // if node is not take parent and try again
		}

		for(auto& pos : dataPosition[lastValue]){
			++histo[m_storage[pos]->getLabel()];
		}
		m_labelsOfWinningClassesInLeaves[i] = argMax<decltype(histo), unsigned int>(histo);
		if(i + 1 != leafAmount){
			std::fill(histo.begin(), histo.end(), 0u);
		}
	}
	if(m_splitDim[1] == NodeType::NODE_CAN_BE_USED && tryCounter < 5){ // five splits are enough to try
		// try again!
		return train(amountOfUsedDims,generator, tryCounter + 1, saveDataPosition);
	}else if(tryCounter >= 5){
		return false;
	}
	if(!saveDataPosition){ // if it is not saved this pointer is deleted
		deleteDataPositions();
	}
//	printStream(std::cout);
	// just in the positive case record Time
	GlobalStopWatch<DynamicDecisionTreeTrain>::instance().recordActTime(dyTrain.elapsedSeconds());
	return true;
}

template<typename dimType>
Real DynamicDecisionTree<dimType>::trySplitFor(const Real usedSplitValue, const unsigned int usedDim,
											   const DataPositions& dataInNode, std::vector<unsigned int>& leftHisto,
											   std::vector<unsigned int>& rightHisto, RandomNumberGeneratorForDT& generator){
	int leftAmount = 0, rightAmount = 0;
	// * 2 ensures that stepSize in else is bigger than 1
	if(dataInNode.size() < m_amountOfPointsCheckedPerSplit * 2){ // under 100 take each value
		for(const auto& pos : dataInNode){
			if(usedSplitValue < m_storage[pos]->coeff(usedDim)){ // TODO check < or <=
				++leftAmount;
				++leftHisto[m_storage[pos]->getLabel()];
			}else{
				++rightAmount;
				++rightHisto[m_storage[pos]->getLabel()];
			}
		}
	}else{
		const auto stepSize = dataInNode.size() / m_amountOfPointsCheckedPerSplit;
		generator.setRandFromRange(1, (int) stepSize);
		const auto dataSize = (int) dataInNode.size();
		for(int i = generator.getRandFromRange(); i < dataSize; i += generator.getRandFromRange()){
			if(i < dataSize){
				const int val = dataInNode[i];
				if(usedSplitValue < m_storage[val]->coeff(usedDim)){ // TODO check < or <=
					++leftAmount;
					++leftHisto[m_storage[val]->getLabel()];
				}else{
					++rightAmount;
					++rightHisto[m_storage[val]->getLabel()];
				}
			}
		}
	}
	if(leftAmount == 0 || rightAmount == 0){
		return 0;
	}
	// Entropy -> TODO maybe Gini
	Real leftCost = 0;
#ifndef USE_GINI
	Real rightCost = 0;
#endif
	for(dimType i = 0; i < m_amountOfClasses; ++i){
		const unsigned int normalizer = leftHisto[i] + rightHisto[i];
		if(normalizer != 0){
			const Real leftClassProb = leftHisto[i] / (Real) normalizer;
#ifndef USE_GINI // first entropy -> if not defined
//			if(leftClassProb > 0._r){
//				leftCost -= leftClassProb * logReal(leftClassProb);
//			}
//			if(leftClassProb < 1.0_r){
//				rightCost -= (1.0_r - leftClassProb) * logReal(1._r - leftClassProb);
//			}
#else
			const Real val = leftClassProb * (1.0_r - leftClassProb);
			leftCost += val;
#endif
		}
		leftHisto[i] = 0;
		rightHisto[i] = 0;
	}
#ifndef USE_GINI
	return rightAmount * rightCost + leftAmount * leftCost;
#else
	return (rightAmount + leftAmount) * leftCost;
#endif
}

template<typename dimType>
unsigned int DynamicDecisionTree<dimType>::predict(const VectorX& point) const{
	int iActNode = 1; // start in root
	return predict(point, iActNode);
}

// is named iActNode here, is easier, but represents in the end the winningLeafNode
template<typename dimType>
unsigned int DynamicDecisionTree<dimType>::predict(const VectorX& point, int& iActNode) const {
	iActNode = 1;
	if(m_splitDim[1] < NodeType::NODE_CAN_BE_USED){
		const auto maxIntern = (int) m_maxInternalNodeNr + 1;
		while(iActNode < maxIntern){
			if(m_splitDim[iActNode] >= NodeType::NODE_CAN_BE_USED){
				// if there is a node which isn't used on the way down to the leave
				while(iActNode < maxIntern){
					// go down always on the left side (it doesn't really matter)
					iActNode *= 2;
				}
				break;
			}
			const bool right = (m_splitValues[iActNode] < point.coeff(m_splitDim[iActNode]));
			iActNode *= 2; // get to next level
			iActNode += (int) right;
		}
		iActNode -= pow2(m_maxDepth);
		return m_labelsOfWinningClassesInLeaves[iActNode];
	}else{
		printError("A tree must be trained before it can predict anything!");
		return UNDEF_CLASS_LABEL;
	}
}

template<typename dimType>
bool DynamicDecisionTree<dimType>::predictIfPointsShareSameLeaveWithHeight(const VectorX& point1,
																  const VectorX& point2,
																  const int usedHeight) const {
	int iActNode = 1; // start in root
	int actLevel = 1;
	if(m_splitDim[1] != NodeType::NODE_IS_NOT_USED && m_splitDim[1] != NodeType::NODE_CAN_BE_USED){
		while(iActNode <= (int) m_maxInternalNodeNr){
			if(m_splitDim[iActNode] >= NodeType::NODE_CAN_BE_USED){
				// if there is a node which isn't used on the way down to the leave
				while(iActNode <= (int) m_maxInternalNodeNr){ // go down always on the left side (it doesn't really matter)
					iActNode *= 2;
				}
				break;
			}
			const bool firstPointRight = m_splitValues[iActNode] < point1.coeff(m_splitDim[iActNode]);
			const bool secondPointRight = m_splitValues[iActNode] < point2.coeff(m_splitDim[iActNode]);
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

template<typename dimType>
void DynamicDecisionTree<dimType>::adjustToNewData(){
	std::fill(m_labelsOfWinningClassesInLeaves.begin(), m_labelsOfWinningClassesInLeaves.end(), 0);
	const unsigned int leafAmount = pow2(m_maxDepth);
	std::vector< std::vector<int> > histo(leafAmount, std::vector<int>(m_amountOfClasses, 0));
	auto startPos = 0; // TODO
	for(unsigned int i = startPos; i < m_storage.size(); ++i){
		const auto point = m_storage[i];
		int iActNode = 1; // start in root
		while(iActNode <= (int) m_maxInternalNodeNr){
			if(m_splitDim[iActNode] >= NodeType::NODE_CAN_BE_USED){
				// if there is a node which isn't used on the way down to the leave
				while(iActNode <= (int) m_maxInternalNodeNr){ // go down always on the left side (it doesn't really matter)
					iActNode *= 2;
				}
				break;
			}
			const bool right = m_splitValues[iActNode] < point->coeff(m_splitDim[iActNode]);
			iActNode *= 2; // get to next level
			if(right){ // point is on right side of split
				++iActNode; // go to right node
			}

		}
		++histo[iActNode - leafAmount][point->getLabel()];
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

template<typename dimType>
unsigned int DynamicDecisionTree<dimType>::getNrOfLeaves(){
	return pow2(m_maxDepth);
}

template<typename dimType>
unsigned int DynamicDecisionTree<dimType>::amountOfClasses() const{
	return m_amountOfClasses;
}

template<typename dimType>
void DynamicDecisionTree<dimType>::deleteDataPositions(){
	saveDelete(m_dataPositions);
}

template<typename dimType>
MemoryType DynamicDecisionTree<dimType>::getMemSize() const {
	const auto splits = ((MemoryType) m_maxInternalNodeNr + 1) * (sizeof(Real) + sizeof(dimType) * 2); // size of data
	const auto refs = (MemoryType) sizeof(int*) * 6; // size of pointers and refs
	return refs + splits;
}

//void DynamicDecisionTree<dimType>::printStream(std::ostream &output, const Real precision){
//	if(&output == &std::cout){
//#ifdef USE_SCREEN_OUPUT
//		printError("This print message is not supported if output is std::cout and the panels are used!");
//		return;
//#endif
//		output << "---------------DynamicDecisonTree---------------" << "\n";
//		const bool isEven = m_maxDepth % 2 == 0;
//		for(unsigned int i = 0; i < m_maxDepth; ++i){
//			const auto firstElementInLayer = (i > 0 ? pow2(i) : 1);
//			const auto maxNumberInLayer = pow2(i+1);
//			const auto forStart = m_maxDepth - i;
//			for(unsigned int k = 0; k < forStart * 2; ++k){
//				output << "\t\t";
//			}
////			output << firstElementInLayer << "; " << maxNumberInLayer << std::endl;
//			auto tabs = std::string("\t");
//			for(unsigned int k = 0; k < forStart; ++k){
//				tabs += "\t";
//			}
//			for(auto iActNode = firstElementInLayer; iActNode < maxNumberInLayer; ++iActNode){
//				output << "(" << m_splitDim[iActNode] << ", " << StringHelper::number2String(m_splitValues[iActNode], precision) << ")" << tabs;
//			}
//			output << "\n";
//		}
//		output << "------------------------------------------------" << std::endl;
//	}
//
//}
