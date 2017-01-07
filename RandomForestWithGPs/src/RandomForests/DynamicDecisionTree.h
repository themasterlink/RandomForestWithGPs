/*
 * DynamicDecisionTree.h
 *
 *  Created on: 17.10.2016
 *      Author: Max
 */

#ifndef RANDOMFORESTS_DYNAMICDECISIONTREE_H_
#define RANDOMFORESTS_DYNAMICDECISIONTREE_H_

#include "../Data/OnlineStorage.h"
#include "../Data/ClassPoint.h"
#include "../RandomNumberGenerator/RandomNumberGeneratorForDT.h"
#include "../RandomForests/DecisionTreeData.h"
#include "../Base/Predictor.h"
#include "DynamicDecisionTreeInterface.h"

class DynamicDecisionTree : public DynamicDecisionTreeInterface {
public:
	enum NodeType{ // saved in m_splitDim
		NODE_IS_NOT_USED = -1,
		NODE_CAN_BE_USED = -2,
	};

	DynamicDecisionTree(OnlineStorage<ClassPoint*>& storage, const int maxDepth, const int amountOfClasses);

	// copy construct
	DynamicDecisionTree(const DynamicDecisionTree& tree);

	virtual ~DynamicDecisionTree();


	void train(int amountOfUsedDims, RandomNumberGeneratorForDT& generator){
		train(amountOfUsedDims, generator, 0, false);
	}

	bool train(int amountOfUsedDims, RandomNumberGeneratorForDT& generator, const int tryCounter, const bool saveDataPosition);

	double trySplitFor(const int actNode, const double usedSplitValue, const int usedDim,
			const std::vector<int>& dataInNode, std::vector<int>& leftHisto,
			std::vector<int>& rightHisto, RandomNumberGeneratorForDT& generator);

	void adjustToNewData();

	int predict(const DataPoint& point) const;

	int predict(const DataPoint& point, int& winningLeafNode) const;

	bool predictIfPointsShareSameLeaveWithHeight(const DataPoint& point1, const DataPoint& point2, const int usedHeight) const;

	void predictData(const Data& data, Labels& labels) const{
		printError("This function is not implemented!");
	}

	void predictData(const Data& points, Labels& labels, std::vector< std::vector<double> >& probabilities) const{
		printError("Not implemented yet!");
	}

	int getNrOfLeaves();

	unsigned int amountOfClasses() const;

	std::vector<std::vector<int> >* getDataPositions(){ return m_dataPositions; };

	void setUsedDataPositions(std::vector<int>* usedDataPositions){ m_useOnlyThisDataPositions = usedDataPositions; };

private:
	OnlineStorage<ClassPoint*>& m_storage;
	// max depth allowed in this tree
	const int m_maxDepth;
	// max number of nodes possible in this tree
	const int m_maxNodeNr; // = pow(2, m_maxDepth +1) - 1
	// max number of nodes, which have children
	const int m_maxInternalNodeNr; // = pow(2, m_maxDepth) - 1

	const int m_amountOfClasses;
	// contains the split values for the nodes:
	// the order of the nodes is like that:
	// !!!! first element is not used !!!!
	// 			1
	// 		2		3
	//  4	   5  6 	7
	// 8 9 	10 11 12 13  14 15
	std::vector<double> m_splitValues;
	// order is like with split values
	std::vector<int> m_splitDim;

	std::vector<int> m_labelsOfWinningClassesInLeaves;

	// is used in the BigDynamicDecisionTree
	std::vector<std::vector<int> >* m_dataPositions;

	std::vector<int>* m_useOnlyThisDataPositions;
};

#endif /* RANDOMFORESTS_DYNAMICDECISIONTREE_H_ */
