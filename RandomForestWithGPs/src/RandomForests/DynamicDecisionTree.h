/*
 * DynamicDecisionTree.h
 *
 *  Created on: 17.10.2016
 *      Author: Max
 */

#ifndef RANDOMFORESTS_DYNAMICDECISIONTREE_H_
#define RANDOMFORESTS_DYNAMICDECISIONTREE_H_

#include "../Data/OnlineStorage.h"
#include "../Data/LabeledVectorX.h"
#include "../RandomNumberGenerator/RandomNumberGeneratorForDT.h"
#include "../RandomForests/DecisionTreeData.h"
#include "../Base/Predictor.h"
#include "DynamicDecisionTreeInterface.h"

#define USE_GINI // if not used -> Entropy is used

class ReadWriterHelper;

template<typename dimType = unsigned short>
class DynamicDecisionTree : public DynamicDecisionTreeInterface{

	/** \brief This constant defines how many dimensions and classes can be used. It further defines the amount of maximum
	 * amount of nodes in this tree, the amount of levels is dimType.
	 * sizeof only return bytes - 1, because it starts at zero
	 */
	static const dimType m_maxAmountOfElements = (dimType) (pow2((unsigned long) sizeof(dimType) * 8) - 1);

	friend ReadWriterHelper;

public:
	/** \brief This type saves the ids of each point, the id is the index in the Storage used in the ORF
	 */
	using DataPositions = std::vector<unsigned int>;

	/**
	 * \brief Saved in the dimension array (m_splitDim), if a node is not used the constant NODE_IS_NOT_USED.
	 *
	 * In the beginning all nodes are filled with NODE_CAN_BE_USED, for these nodes during the training a check is
	 * performed
	 */
	enum NodeType : dimType{
		NODE_IS_NOT_USED = m_maxAmountOfElements,
		NODE_CAN_BE_USED = m_maxAmountOfElements - 1 // must be the smaller element (code depends on it)
	};

	DynamicDecisionTree(OnlineStorage<LabeledVectorX *> &storage, const unsigned int maxDepth,
						const unsigned int amountOfClasses, const unsigned int amountOfPointsPerSplit);

	/**
	 * \brief Construct empty tree, the max depth is set to 1, has to be reset before the training
	 * \param storage which is used to update the tree
	 */
	DynamicDecisionTree(OnlineStorage<LabeledVectorX *> &storage);

	// copy construct
	DynamicDecisionTree(const DynamicDecisionTree &tree);

	virtual ~DynamicDecisionTree();

	void train(unsigned int amountOfUsedDims, RandomNumberGeneratorForDT &generator) override {
		train((dimType) amountOfUsedDims, generator, 0, false);
	}

	/**
	 * \brief Train the decision tree, goes over all nodes in the tree and checks if a split is possible there.
	 * 		In each node several random splits are drawn and checked against several points of the storage, the best
	 * 		split is selected
	 * \param amountOfUsedDims amount of used dims in the training
	 * \param generator generates the random numbers needed for the training
	 * \param tryCounter amount of tries of the training, is initalized with 0, if it reaches 5 no new try is done
	 * \param saveDataPosition save the data positions beyond over the training function
	 * \return if the training was successful true is returned
	 */
	bool train(dimType amountOfUsedDims, RandomNumberGeneratorForDT &generator, const dimType tryCounter,
			   const bool saveDataPosition);

	/**
	 * \brief Calculate the winning classes new depending on the current storage
	 */
	void adjustToNewData();

	/**
	 * \brief Try a split for the usedSplitValue in the usedDim for the data points saved in the dataInNode,
	 * 		the results are saved in the leftHisto and the rightHisto, for each class.
	 *
	 * If the amount of points in dataInNode is below two times of m_amountOfPointsCheckedPerSplit. All points of this
	 * 	node are checked, against the split value and the corresponding histogram is updated at the label of the
	 * 	current point. If it is above the generator is used to sample a subset of the data in note to test it on them.
	 * \param usedSplitValue which is used to generate the performance
	 * \param usedDim the current dimension in which the split should be performed
	 * \param dataInNode data in the current node
	 * \param leftHisto left histogram, has the length of the amount of classes used in this tree
	 * \param rightHisto right histogram, has the length of the amount of classes used in this tree
	 * \param generator used to calc the steps over the storage
	 * \return returns the quality value of the split
	 */
	Real trySplitFor(const Real usedSplitValue, const unsigned int usedDim,
					 const std::vector<unsigned int> &dataInNode, std::vector<unsigned int> &leftHisto,
					 std::vector<unsigned int> &rightHisto, RandomNumberGeneratorForDT &generator);

	/** @defgroup Predict All prediction functions
 	 *  @{
 	 */

	/**
	 * \brief Predict the class for a given point, returns the majority class of the reached leaf in the tree
	 * \param point for which the class is estimated
	 * \return the predicted class of the given point
	 */
	unsigned int predict(const VectorX &point) const override;

	/**
	 * \brief Predict the class for a given point, returns the majority class of the reached leaf in the tree
	 * 		writes in the variables winningLeafNode the id of the node which won at the end.
	 * \param point for which the class is estimated
	 * \param winningLeafNode id of the winning leaf node
	 * \return the predicted class of the given point
	 */
	unsigned int predict(const VectorX &point, int &winningLeafNode) const;

	/**
	 * \brief Predict if two points share the same leaf at a certain height
	 * \param point1 first point to compare
	 * \param point2 second point to compare
	 * \param usedHeight at which both points have to be in the same split
	 * \return if these two points share the same node at the given height
	 */
	bool predictIfPointsShareSameLeafWithHeight(const VectorX& point1, const VectorX& point2,
												const int usedHeight) const override;

	/**
	 * \brief Is not implemented (only override derived function)
	 */
	void predictData(const Data &data, Labels &labels) const override{
		UNUSED(data);
		UNUSED(labels);
		printErrorAndQuit("This function is not implemented!");
	}

	/**
	 * \brief Is not implemented (only override derived function)
	 */
	void predictData(const Data &points, Labels &labels, std::vector<std::vector<Real> > &probabilities) const override{
		UNUSED(points);
		UNUSED(labels);
		UNUSED(probabilities);
		printErrorAndQuit("Not implemented yet!");
	}

	/** @} */ // end of predict group

	/**
	 * \brief Get the number of leaves in the tree
	 * \return the number of leaves in the tree
	 */
	unsigned int getNrOfLeaves();

	/**
	 * \brief Return the number of used classes in this tree
	 * \return the amount of classes in the tree
	 */
	unsigned int amountOfClasses() const override;

	/**
	 * \brief Return a pointer to the used data positions, the vector has the length of the maximum node amount and for
	 * 	each node a vector is saved, which contains all the data points, which have been in it. At the end of the training
	 * 	only the leaves are still filled
	 * \return a pointer to the used data positions
	 */
	std::vector<std::vector<unsigned int> > *getDataPositions(){ return m_dataPositions; };

	/**
	 * \brief Delete the array for the data positions
	 */
	void deleteDataPositions();

	/**
	 * \brief If not the whole storage should be used a data position array can be used, which specifies the points,
	 * which should be used.
	 * \param usedDataPositions
	 */
	void setUsedDataPositions(std::vector<unsigned int> *usedDataPositions){ m_useOnlyThisDataPositions = usedDataPositions; };

	/**
	 * \brief Returns the memory size for the tree
	 * \return returns the amount of memory used
	 */
	MemoryType getMemSize() const override;

private:
	// this function is only called if the empty tree constructor was used!
	void prepareForSetting(const unsigned int maxDepth, const unsigned int amountOfClasses);

	OnlineStorage<LabeledVectorX*>& m_storage;
	// max depth allowed in this tree
	const dimType m_maxDepth;
	// max number of nodes possible in this tree
	const dimType m_maxNodeNr; // = pow2(m_maxDepth +1) - 1
	// max number of nodes, which have children
	const dimType m_maxInternalNodeNr; // = pow2(m_maxDepth) - 1

	const dimType m_amountOfClasses;

	const DataPositions::size_type m_amountOfPointsCheckedPerSplit;

	// contains the split values for the nodes:
	// the order of the nodes is like that:
	// !!!! first element is not used !!!!
	// 			1
	// 		2		3
	//  4	   5  6 	7
	// 8 9 	10 11 12 13  14 15
	std::vector<Real> m_splitValues;
	// order is like with split values
	std::vector<dimType> m_splitDim;

	std::vector<dimType> m_labelsOfWinningClassesInLeaves;

	// is used in the BigDynamicDecisionTree
	// fill contain the data for each node ->
	// 	in the end only the last pow2(m_maxDepth) will have values, the rest will be empty
	std::vector<DataPositions>* m_dataPositions;

	DataPositions* m_useOnlyThisDataPositions;

};


#define __INCLUDE_DYNAMICDECISIONTREE

#include "DynamicDecisionTree_i.h"

#undef __INCLUDE_DYNAMICDECISIONTREE

#endif /* RANDOMFORESTS_DYNAMICDECISIONTREE_H_ */
