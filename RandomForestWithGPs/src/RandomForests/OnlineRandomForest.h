/*
 * OnlineRandomForest.h
 *
 *  Created on: 13.10.2016
 *      Author: Max
 */

#ifndef RANDOMFORESTS_ONLINERANDOMFOREST_H_
#define RANDOMFORESTS_ONLINERANDOMFOREST_H_

#include "DynamicDecisionTree.h"
#include "../Data/Data.h"
#include "../Data/OnlineStorage.h"
#include "../Data/ClassData.h"
#include <list>
#include "TreeCounter.h"

class OnlineRandomForest : public Observer, public PredictorMultiClass, public Subject {
public:

	typedef typename std::list<DynamicDecisionTree> DecisionTreesContainer;
	typedef typename DecisionTreesContainer::iterator DecisionTreeIterator;
	typedef typename DecisionTreesContainer::const_iterator DecisionTreeConstIterator;

	OnlineRandomForest(OnlineStorage<ClassPoint*>& storage, const int maxDepth, const int amountOfUsedClasses);

	virtual ~OnlineRandomForest();

	void train();

	int predict(const DataPoint& point) const;

	void predictData(const Data& points, Labels& labels) const;

	void predictData(const ClassData& points, Labels& labels) const;

	void predictData(const Data& points, Labels& labels, std::vector< std::vector<double> >& probabilities) const;

	void getLeafNrFor(std::vector<int>& leafNrs);

	int getNrOfTrees() const { return m_trees.size(); };

	void update(Subject* caller, unsigned int event);

	bool update();

	int amountOfClasses() const;

	OnlineStorage<ClassPoint*>& getStorageRef();

	ClassTypeSubject classType() const;

	const std::vector<Eigen::Vector2d >& getMinMaxValues(){ return m_minMaxValues;};

private:

	typedef typename std::pair<DecisionTreeIterator, double> SortedDecisionTreePair;
	typedef typename std::list<SortedDecisionTreePair > SortedDecisionTreeList;

	void predictDataInParallel(const Data& points, Labels* labels, const int start, const int end) const;

	void predictClassDataInParallel(const ClassData& points, Labels* labels, const int start, const int end) const;

	void predictDataProbInParallel(const Data& points, Labels* labels, std::vector< std::vector<double> >* probabilities, const int start, const int end) const;

	void trainInParallel(RandomNumberGeneratorForDT* generator, InformationPackage* package);

	void sortTreesAfterPerformance(SortedDecisionTreeList& list);

	void internalAppendToSortedList(SortedDecisionTreeList* list, DecisionTreeIterator& itTree, double correctVal);

	void updateInParallel(SortedDecisionTreeList* list, const int amountOfSteps,
			boost::mutex* mutex, unsigned int threadNr, InformationPackage* package, int* counter);

	void updateMinMaxValues(unsigned int event);

	const int m_maxDepth;

	const int m_amountOfClasses;

	int m_amountOfPointsUntilRetrain;

	int m_counterForRetrain;

	int m_amountOfUsedDims;

	double m_factorForUsedDims;

	Eigen::Vector2d m_minMaxUsedDataFactor;

	// used in all decision trees -> no copies needed!
	std::vector<Eigen::Vector2d > m_minMaxValues;

	OnlineStorage<ClassPoint*>& m_storage;

	DecisionTreesContainer m_trees;

	DecisionTreeIterator findWorstPerformingTree(double& correctAmount);

	Eigen::Vector2i getMinMaxData();

	std::vector<RandomNumberGeneratorForDT*> m_generators;

	boost::mutex m_treesMutex;

	bool m_firstTrainingDone;
};


#endif /* RANDOMFORESTS_ONLINERANDOMFOREST_H_ */
