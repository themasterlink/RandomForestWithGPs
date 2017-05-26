/*
 * DynamicDecisionTree.cc
 *
 *  Created on: 17.10.2016
 *      Author: Max
 */

#include "DynamicDecisionTree.h"
#include "../Data/ClassKnowledge.h"

DynamicDecisionTree::DynamicDecisionTree(OnlineStorage<LabeledVectorX *> &storage, const unsigned int maxDepth,
										 const unsigned int amountOfClasses, const unsigned int amountOfPointsPerSplit):
		m_storage(storage),
		m_maxDepth(maxDepth),
		m_maxNodeNr(pow2(maxDepth + 1) - 1),
		m_maxInternalNodeNr(pow2(maxDepth) - 1),
		m_amountOfClasses(amountOfClasses),
		m_amountOfPointsCheckedPerSplit((decltype(m_amountOfPointsCheckedPerSplit)) amountOfPointsPerSplit),
		m_splitValues(m_maxInternalNodeNr + 1), // + 1 -> no use of the first element
		m_splitDim(m_maxInternalNodeNr + 1, NodeType::NODE_IS_NOT_USED),
		m_labelsOfWinningClassesInLeaves(pow2(maxDepth), UNDEF_CLASS_LABEL),
		m_dataPositions(nullptr),
		m_useOnlyThisDataPositions(nullptr){
	if(m_amountOfPointsCheckedPerSplit == 0){
		printError("This tree can not be trained!");
	}
	if(m_maxDepth <= 0 || m_maxDepth >= 28){
		printError("This height is not supported here: " << m_maxDepth);
	}
//	printOnScreen("Size is: " << (m_splitDim.size() * sizeof(int) + m_splitValues.size() * sizeof(Real) + m_labelsOfWinningClassesInLeaves.size() * sizeof(int)));
}


// construct empty tree
DynamicDecisionTree::DynamicDecisionTree(OnlineStorage<LabeledVectorX*>& storage):
		DynamicDecisionTree(storage, 1, ClassKnowledge::amountOfClasses(), 100){
}

// fill empty tree
void DynamicDecisionTree::prepareForSetting(const unsigned int maxDepth, const unsigned int amountOfClasses){
	if(m_maxDepth == 1 && maxDepth > 0 && maxDepth < 28){
		overwriteConst(m_maxDepth, maxDepth);
		overwriteConst(m_amountOfClasses, amountOfClasses);
		overwriteConst(m_maxNodeNr, pow2(maxDepth + 1) - 1);
		overwriteConst(m_maxInternalNodeNr, pow2(maxDepth) - 1);
	}else{
		printError("The empty tree constructor was not called before!");
	}
}

// copy construct
DynamicDecisionTree::DynamicDecisionTree(const DynamicDecisionTree& tree):
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

DynamicDecisionTree::~DynamicDecisionTree(){
	m_useOnlyThisDataPositions = nullptr; // never delete this pointer! (does not belong to this object)
	deleteDataPositions();
}

bool DynamicDecisionTree::train(unsigned int amountOfUsedDims, RandomNumberGeneratorForDT& generator,
								const unsigned int tryCounter, const bool saveDataPosition){
	if(m_splitDim[1] != NodeType::NODE_IS_NOT_USED || m_splitDim[1] != NodeType::NODE_CAN_BE_USED){
		// reset training
		std::fill(m_splitDim.begin(), m_splitDim.end(), NodeType::NODE_IS_NOT_USED);
		std::fill(m_labelsOfWinningClassesInLeaves.begin(), m_labelsOfWinningClassesInLeaves.end(), UNDEF_CLASS_LABEL);
	}
	std::vector<int> usedDims(amountOfUsedDims,-1);
	if(amountOfUsedDims == ClassKnowledge::amountOfDims()){
		for(unsigned int i = 0; i < amountOfUsedDims; ++i){
			usedDims[i] = i;
		}
	}else{
		generator.setRandForDim(0, ClassKnowledge::amountOfDims() - 1);
		for(unsigned int i = 0; i < amountOfUsedDims; ++i){
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
						break;
					}
				}
				if(!doAgain){
					usedDims[i] = randNr;
				}
				++counter;
			}while(doAgain);
		}
		generator.setRandForDim(0, amountOfUsedDims - 1);
	}
	m_splitDim[1] = NodeType::NODE_CAN_BE_USED; // init the root value
	std::vector<unsigned int> leftHisto(m_amountOfClasses), rightHisto(m_amountOfClasses);
	m_dataPositions = new std::vector<std::vector<unsigned int> >(m_maxNodeNr + 1, std::vector<unsigned int>());
	std::vector<std::vector<unsigned int> >& dataPosition(*m_dataPositions);
//	int breakPoint = 3; // 1 + 2 -> should have at least 2 layers
//	int actLayer = 2;
//	bool atLeastPerformedOneSplit = false;
	//  1
	// 2 3
	if(m_useOnlyThisDataPositions == nullptr){
		if(generator.useWholeDataSet()){
			dataPosition[1].resize(m_storage.size());
			std::iota(dataPosition[1].begin(), dataPosition[1].end(), 0);
		}else{
			const auto amountOfPoints = (unsigned int) (m_storage.size() / (generator.getStepSize() * 0.375));
			dataPosition[1].reserve(amountOfPoints);
			// -1 that the first value in the storage is used too
			for(unsigned int i = generator.getRandStepOverStorage() - 1;
				i < m_storage.size(); i += generator.getRandStepOverStorage()){
				dataPosition[1].push_back(i);
			}
		}
	}
//	else{ // no need take ref to
//		dataPosition[1].insert(dataPosition[1].end(), m_useOnlyThisDataPositions->begin(), m_useOnlyThisDataPositions->end());
//	}
	const auto amountOfTriedDims = (unsigned int) std::min(100, std::max((int) (usedDims.size() * 0.1), 2));
	for(int iActNode = 1; iActNode < (int) m_maxInternalNodeNr + 1; ++iActNode){ // first element is not used!
		std::vector<unsigned int>& actDataPos = m_useOnlyThisDataPositions != nullptr && iActNode == 1
				? *m_useOnlyThisDataPositions : dataPosition[iActNode];
//		if(iActNode == breakPoint){ // check if early breaking is possible, check is performed always at the start of a layer
//			if(!atLeastPerformedOneSplit){
//				break;
//			}
//			atLeastPerformedOneSplit = false;
//			breakPoint += pow2(actLayer); // first iteration from 3 -> 7, 7 -> 15, 15 -> 31
//			++actLayer;
//		}
		if(m_splitDim[iActNode] == NodeType::NODE_IS_NOT_USED){ // checks if node contains data or not
			continue; // if node is not used, go to next node, if node can be used process it
		}
		// calc actual nodesararg
		// calc split value for each node
		// choose dimension for split
		int randDim, amountOfUsedData;
		Real minDimValue, maxDimValue;
		// try different dimension and find one where the points have a difference
		amountOfUsedData = generator.getRandAmountOfUsedData();
		for(unsigned int i = 0; i < amountOfTriedDims; ++i){
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
		for(auto it = actDataPos.cbegin(); it != actDataPos.cend(); ++it){
			if(m_storage[*it]->coeff(randDim) >= m_splitValues[iActNode]){ // TODO check >= like below  or only >
				rightDataPos.push_back(*it);
			}else{
				leftDataPos.push_back(*it);
			}
		}

		/*std::cout << "i: " << iActNode << std::endl;
			 std::cout << "length: " << actDataPos.size() << std::endl;
			 std::cout << "Found data left  " << foundDataLeft << std::endl;
			 std::cout << "Found data right " << foundDataRight << std::endl;*/
		if(rightDataPos.empty() || leftDataPos.empty()){
			// split is not needed
			dataPosition[leftPos].clear();
			dataPosition[rightPos].clear();
			m_splitDim[iActNode] = NodeType::NODE_IS_NOT_USED; // do not split here
			if(iActNode == 1){
				m_splitDim[iActNode] = NodeType::NODE_CAN_BE_USED; // there should be a split
			}
		}else{
//			atLeastPerformedOneSplit = true;
			actDataPos.clear();
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
		m_labelsOfWinningClassesInLeaves[i] = (unsigned int) argMax(histo.cbegin(), histo.cend());
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
	return true;
}

Real DynamicDecisionTree::trySplitFor(const Real usedSplitValue, const unsigned int usedDim,
		const std::vector<unsigned int>& dataInNode, std::vector<unsigned int>& leftHisto,
		std::vector<unsigned int>& rightHisto, RandomNumberGeneratorForDT& generator){
	int leftAmount = 0, rightAmount = 0;
	if(dataInNode.size() < m_amountOfPointsCheckedPerSplit){ // under 100 take each value
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
	// Entropy -> TODO maybe Gini
	Real leftCost = 0, rightCost = 0;
	for(unsigned int i = 0; i < m_amountOfClasses; ++i){
		const Real normalizer = leftHisto[i] + rightHisto[i];
		if(normalizer > 0){
			const Real leftClassProb = leftHisto[i] / normalizer;
			if(leftClassProb > 0){
				leftCost -= leftClassProb * log(leftClassProb);
			}
			if(leftClassProb < 1.0){
				rightCost -= (1. - leftClassProb) * log(1. - leftClassProb);
			}
//			if(leftClassProb > 0){
//				leftCost += leftClassProb * (1- leftClassProb);
//			}
//			if(leftClassProb < 1.0){
//				rightCost += (leftClassProb) * ((1. - leftClassProb));
//			}
		}
		leftHisto[i] = 0;
		rightHisto[i] = 0;
	}
	return rightAmount * rightCost + leftAmount * leftCost;
}

unsigned int DynamicDecisionTree::predict(const VectorX& point) const{
	int iActNode = 1; // start in root
	return predict(point, iActNode);
}

// is named iActNode here, is easier, but represents in the end the winningLeafNode
unsigned int DynamicDecisionTree::predict(const VectorX& point, int& iActNode) const {
	iActNode = 1;
	if(m_splitDim[1] != NodeType::NODE_IS_NOT_USED && m_splitDim[1] != NodeType::NODE_CAN_BE_USED){
		while(iActNode <= (int) m_maxInternalNodeNr){
			if(m_splitDim[iActNode] == NodeType::NODE_IS_NOT_USED || m_splitDim[iActNode] == NodeType::NODE_CAN_BE_USED){
				// if there is a node which isn't used on the way down to the leave
				while(iActNode <= (int) m_maxInternalNodeNr){
					// go down always on the left side (it doesn't really matter)
					iActNode *= 2;
				}
				break;
			}
			const bool right = m_splitValues[iActNode] < point.coeff(m_splitDim[iActNode]);
			iActNode *= 2; // get to next level
			if(right){ // point is on right side of split
				++iActNode; // go to right node
			}
		}
		iActNode -= pow2(m_maxDepth);
		return m_labelsOfWinningClassesInLeaves[iActNode];
	}else{
		printError("A tree must be trained before it can predict anything!");
		return UNDEF_CLASS_LABEL;
	}
}

bool DynamicDecisionTree::predictIfPointsShareSameLeaveWithHeight(const VectorX& point1,
																  const VectorX& point2,
																  const int usedHeight) const {
	int iActNode = 1; // start in root
	int actLevel = 1;
	if(m_splitDim[1] != NodeType::NODE_IS_NOT_USED && m_splitDim[1] != NodeType::NODE_CAN_BE_USED){
		while(iActNode <= (int) m_maxInternalNodeNr){
			if(m_splitDim[iActNode] == NodeType::NODE_IS_NOT_USED){
				// if there is a node which isn't used on the way down to the leave
				while(iActNode <= (int) m_maxInternalNodeNr){ // go down always on the left side (it doesn't really matter)
					iActNode *= 2;
				}
				break;
			}else if(m_splitDim[iActNode] == NodeType::NODE_CAN_BE_USED){
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

void DynamicDecisionTree::adjustToNewData(){
	std::fill(m_labelsOfWinningClassesInLeaves.begin(), m_labelsOfWinningClassesInLeaves.end(), 0);
	const unsigned int leafAmount = pow2(m_maxDepth);
	std::vector< std::vector<int> > histo(leafAmount, std::vector<int>(m_amountOfClasses, 0));
	for(OnlineStorage<LabeledVectorX*>::ConstIterator it = m_storage.begin(); it != m_storage.end(); ++it){
		int iActNode = 1; // start in root
		while(iActNode <= (int) m_maxInternalNodeNr){
			if(m_splitDim[iActNode] == NodeType::NODE_IS_NOT_USED){
				// if there is a node which isn't used on the way down to the leave
				while(iActNode <= (int) m_maxInternalNodeNr){ // go down always on the left side (it doesn't really matter)
					iActNode *= 2;
				}
				break;
			}else if(m_splitDim[iActNode] == NodeType::NODE_CAN_BE_USED){
				// if there is a node which isn't used on the way down to the leave
				while(iActNode <= (int) m_maxInternalNodeNr){ // go down always on the left side (it doesn't really matter)
					iActNode *= 2;
				}
				break;
			}
			const bool right = m_splitValues[iActNode] < (**it).coeff(m_splitDim[iActNode]);
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

unsigned int DynamicDecisionTree::getNrOfLeaves(){
	return pow2(m_maxDepth);
}

unsigned int DynamicDecisionTree::amountOfClasses() const{
	return m_amountOfClasses;
}

void DynamicDecisionTree::deleteDataPositions(){
	SAVE_DELETE(m_dataPositions);
}

MemoryType DynamicDecisionTree::getMemSize() const {
	const auto splits = ((MemoryType) m_maxInternalNodeNr + 1) * (sizeof(Real) + sizeof(unsigned int) + sizeof(int));
	const auto refs = (MemoryType) sizeof(int*) * 6; // size of pointers and refs
	return refs + splits; // 16 + 24 = 40, 16 general info, 40 pointers and ref
}

//void DynamicDecisionTree::printStream(std::ostream &output, const Real precision){
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
//				output << "(" << m_splitDim[iActNode] << ", " << number2String(m_splitValues[iActNode], precision) << ")" << tabs;
//			}
//			output << "\n";
//		}
//		output << "------------------------------------------------" << std::endl;
//	}
//
//}
