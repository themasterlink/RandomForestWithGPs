/*
 * OnlineRandomForest.h
 *
 *  Created on: 13.10.2016
 *      Author: Max
 */

#ifndef RANDOMFORESTS_ONLINERANDOMFOREST_H_
#define RANDOMFORESTS_ONLINERANDOMFOREST_H_

#include "BigDynamicDecisionTree.h"
#include "../Data/Data.h"
#include "../Data/OnlineStorage.h"
#include "../Data/ClassData.h"
#include "../RandomNumberGenerator/RandomUniformNr.h"
#include <list>
#include "TreeCounter.h"

class OnlineRandomForest : public Observer, public PredictorMultiClass, public Subject {
public:

	using DecisionTreesContainer = std::list<DynamicDecisionTreeInterface*>;
	using DecisionTreeIterator = DecisionTreesContainer::iterator;
	using DecisionTreeConstIterator = DecisionTreesContainer::const_iterator;

	OnlineRandomForest(OnlineStorage<ClassPoint *> &storage, 
					   const unsigned int maxDepth, const int amountOfUsedClasses);

	virtual ~OnlineRandomForest();

	void train();

	// if amountOfTrees == 0 -> the samplingTime is used
	void setDesiredAmountOfTrees(const unsigned int desiredAmountOfTrees){
		m_desiredAmountOfTrees = desiredAmountOfTrees;
	}

	unsigned int predict(const DataPoint& point) const override;

	double predict(const DataPoint& point1, const DataPoint& point2, const unsigned int sampleAmount) const;

	double predictPartitionEquality(const DataPoint& point1, const DataPoint& point2,
									RandomUniformNr& uniformNr, unsigned int amountOfSamples) const;

	void predictData(const Data& points, Labels& labels) const override;

	void predictData(const Data& points, Labels& labels,
					 std::vector< std::vector<double> >& probabilities) const override;

	void predictData(const ClassData& points, Labels& labels) const;

	void predictData(const ClassData& points, Labels& labels, std::vector< std::vector<double> >& probabilities) const;

	void getLeafNrFor(std::vector<int>& leafNrs);

	int getNrOfTrees() const { return m_amountOfTrainedTrees; };

	void update(Subject* caller, unsigned int event) override;

	bool update();

	unsigned int amountOfClasses() const override;

	OnlineStorage<ClassPoint*>& getStorageRef();

	const OnlineStorage<ClassPoint*>& getStorageRef() const;

	ClassTypeSubject classType() const override;

	const std::vector<Eigen::Vector2d >& getMinMaxValues(){ return m_minMaxValues;};

private:

	using SortedDecisionTreePair = std::pair<DynamicDecisionTreeInterface*, double>;
	using SortedDecisionTreeList = std::list<SortedDecisionTreePair>;

	void predictDataInParallel(const Data& points, Labels* labels,
							   const unsigned int start, const unsigned int end) const;

	void predictClassDataInParallel(const ClassData& points, Labels* labels,
									const unsigned int start, const unsigned int end) const;

	void predictDataProbInParallelStartEnd(const Data& points, Labels* labels,
										   std::vector< std::vector<double> >* probabilities, const unsigned int start,
										   const unsigned int end) const;

	void predictClassDataProbInParallelStartEnd(const ClassData& points, Labels* labels,
												std::vector< std::vector<double> >* probabilities,
												const unsigned int start, const unsigned int end) const;

	void predictDataProbInParallel(const Data& points, std::vector< std::vector<double> >* probabilities,
			unsigned int* iBatchNr, boost::mutex* mutex, DecisionTreeIterator* itOfActElement) const;

	void predictClassDataProbInParallel(const ClassData& points, std::vector< std::vector<double> >* probabilities,
			unsigned int* iBatchNr, boost::mutex* mutex, DecisionTreeIterator* itOfActElement) const;

	void trainInParallel(RandomNumberGeneratorForDT* generator, InformationPackage* package,
						 const unsigned int amountOfTrees, std::vector<std::vector<unsigned int> >* counterForClasses,
						 boost::mutex* mutexForCounter);

	void sortTreesAfterPerformance(SortedDecisionTreeList& list);

	void internalAppendToSortedList(SortedDecisionTreeList* list,
									DynamicDecisionTreeInterface* pTree, double correctVal);

	void mergeSortedLists(SortedDecisionTreeList* aimList, SortedDecisionTreeList* other);

	void sortTreesAfterPerformanceInParallel(SortedDecisionTreeList* list, DecisionTreesContainer* trees,
											 boost::mutex* readMutex, boost::mutex* appendMutex,
											 InformationPackage* package);

	void updateInParallel(SortedDecisionTreeList* list, const unsigned int amountOfSteps,
			boost::mutex* mutex, unsigned int threadNr, InformationPackage* package, unsigned int* counter);

	void updateMinMaxValues(unsigned int event);

	void tryAmountForLayers(RandomNumberGeneratorForDT* generator, const double secondsPerSplit,
							std::list<std::pair<unsigned int, unsigned int> >* layerValues,
							boost::mutex* mutex, std::pair<int, int>* bestLayerSplit, double* bestCorrectness);

	void writeTreesToDisk(const unsigned int amountOfTrees) const;

	void loadBatchOfTreesFromDisk(const unsigned int batchNr) const;

	const unsigned int m_maxDepth;

	const unsigned int m_amountOfClasses;

	int m_amountOfPointsUntilRetrain;

	int m_counterForRetrain;

	int m_amountOfUsedDims;

	double m_factorForUsedDims;

	Eigen::Vector2d m_minMaxUsedDataFactor;

	// used in all decision trees -> no copies needed!
	std::vector<Eigen::Vector2d > m_minMaxValues;

	OnlineStorage<ClassPoint*>& m_storage;

	mutable DecisionTreesContainer m_trees;

	mutable std::vector<std::pair<std::string, std::string> > m_savedToDiskTreesFilePaths;

	DecisionTreeIterator findWorstPerformingTree(double& correctAmount);

	std::vector<RandomNumberGeneratorForDT*> m_generators;

	mutable boost::mutex m_treesMutex;

	bool m_firstTrainingDone;

	double m_ownSamplingTime;

	unsigned int m_desiredAmountOfTrees;

	bool m_useBigDynamicDecisionTrees;

	std::pair<unsigned int, unsigned int> m_amountOfUsedLayer;

	std::string m_folderForSavedTrees;

	bool m_savedAnyTreesToDisk;

	unsigned int m_amountOfTrainedTrees;

	mutable MemoryType m_usedMemory;
};


#endif /* RANDOMFORESTS_ONLINERANDOMFOREST_H_ */
