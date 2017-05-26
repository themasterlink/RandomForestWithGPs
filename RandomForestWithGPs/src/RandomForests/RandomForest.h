/*
 * OtherRandomForest.h
 *
 *  Created on: 01.06.2016
 *      Author: Max
 */

#ifndef OTHERRANDOMFORESTS_OTHERRANDOMFOREST_H_
#define OTHERRANDOMFORESTS_OTHERRANDOMFOREST_H_

#include <boost/thread.hpp> // Boost threads
#include "../RandomNumberGenerator/RandomNumberGeneratorForDT.h"
#include "DecisionTree.h"
#include "DecisionTreeData.h"
#include "../Base/Predictor.h"
#include "TreeCounter.h"

class RandomForest : public PredictorMultiClass {
public:
	using DecisionTreesContainer = std::vector<DecisionTree>;

	RandomForest(const int maxDepth, const int amountOfTrees, const int amountOfUsedClasses);
	virtual ~RandomForest();

	void addForest(const RandomForest& forest);

	void train(const LabeledData& data, const int amountOfUsedDims,
			const Vector2i& minMaxUsedData);

	void init(const int amountOfTrees);

	void generateTreeBasedOnData(const DecisionTreeData& data, const int element);

	unsigned int predict(const VectorX& point) const override;

	unsigned int predict(const LabeledVectorX& point) const;

	void predictData(const Data& points, Labels& labels) const override;

	void predictData(const LabeledData& points, Labels& labels) const;

	void predictData(const Data& points, Labels& labels, std::vector< std::vector<real> >& probabilities) const override{
		UNUSED(points); UNUSED(labels); UNUSED(probabilities);
		printError("Not implemented yet!");
	}

	int getNrOfTrees() const { return m_trees.size(); };

	const DecisionTreesContainer& getTrees() const{ return m_trees; };

	DecisionTreesContainer& getTrees(){ return m_trees; };

	void getLeafNrFor(const LabeledData& data, std::vector<int>& leafNrs);

	unsigned int amountOfClasses() const override;

private:
	const int m_amountOfTrees;

	const int m_amountOfClasses;

	// amount of values needed to update the tree counter variable
	int m_counterIncreaseValue;

	DecisionTreesContainer m_trees;

	void trainInParallel(const LabeledData& data, const int amountOfUsedDims,
			RandomNumberGeneratorForDT* generator, const int start, const int end,
			TreeCounter* counter);

	void predictDataInParallel(const Data& points, Labels* labels, const int start,
			const int end) const;

	void predictDataInParallelClass(const LabeledData& points, Labels* labels, const int start,
			const int end) const;

};

#endif /* OTHERRANDOMFORESTS_OTHERRANDOMFOREST_H_ */
