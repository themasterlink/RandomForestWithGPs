/*
 * OnlineRandomForest.h
 *
 *  Created on: 13.10.2016
 *      Author: Max
 */

#ifndef RANDOMFORESTS_ONLINERANDOMFOREST_H_
#define RANDOMFORESTS_ONLINERANDOMFOREST_H_

#include "BigDynamicDecisionTree.h"
#include "../Data/OnlineStorage.h"
#include "../Data/LabeledVectorX.h"
#include "../RandomNumberGenerator/RandomUniformNr.h"
#include "TreeCounter.h"
#include "AcceptanceCalculator.h"

class OnlineRandomForest : public Observer, public PredictorMultiClass, public Subject {
public:

	using DecisionTreePointer = SharedPtr<DynamicDecisionTreeInterface>;
	using DecisionTreesContainer = std::list<DecisionTreePointer>;
	using DecisionTreeIterator = DecisionTreesContainer::iterator;
	using DecisionTreeConstIterator = DecisionTreesContainer::const_iterator;

	OnlineRandomForest(OnlineStorage<LabeledVectorX *> &storage,
					   const unsigned int maxDepth, const int amountOfUsedClasses);

	virtual ~OnlineRandomForest() = default;

	void train();

	unsigned int predict(const VectorX& point) const override;

	Real predict(const VectorX& point1, const VectorX& point2, const unsigned int sampleAmount) const;

	Real predictPartitionEquality(const VectorX& point1, const VectorX& point2,
									RandomUniformNr& uniformNr, unsigned int amountOfSamples) const;

	void predictData(const Data& points, Labels& labels) const override;

	void predictData(const Data& points, Labels& labels,
					 std::vector< std::vector<Real> >& probabilities) const override;

	void predictData(const LabeledData& points, Labels& labels) const;

	void predictData(const LabeledData& points, Labels& labels, std::vector< std::vector<Real> >& probabilities) const;

	void getLeafNrFor(std::vector<int>& leafNrs);

	int getNrOfTrees() const { return m_amountOfTrainedTrees; };

	void update(Subject* caller, unsigned int event) override;

	bool update();

	unsigned int amountOfClasses() const override;

	OnlineStorage<LabeledVectorX*>& getStorageRef();

	const OnlineStorage<LabeledVectorX*>& getStorageRef() const;

	ClassTypeSubject classType() const override;

	const std::vector<Vector2>& getMinMaxValues(){ return m_minMaxValues;};

	struct TrainingsConfig {

		enum class TrainingsMode{
			TIME,
			TIME_WITH_MEMORY,
			MEMORY,
			TREEAMOUNT,
			TREEAMOUNT_WITH_MEMORY,
			UNDEFINED
		};

		TrainingsConfig(): m_mode(TrainingsMode::UNDEFINED),
						   m_seconds(0.0), m_memory(0),
						   m_amountOfTrees(0){};

		TrainingsMode m_mode;
		Real m_seconds;
		MemoryType m_memory;
		unsigned int m_amountOfTrees;

		bool isTimeMode(){
			return m_mode == TrainingsMode::TIME || m_mode == TrainingsMode::TIME_WITH_MEMORY;
		}

		bool isTreeAmountMode(){
			return m_mode == TrainingsMode::TREEAMOUNT || m_mode == TrainingsMode::TREEAMOUNT_WITH_MEMORY;
		}

		bool hasMemoryConstraint(){
			return m_mode == TrainingsMode::TIME_WITH_MEMORY ||
				   m_mode == TrainingsMode::TREEAMOUNT_WITH_MEMORY ||
				   m_mode == TrainingsMode::MEMORY;
		}
	};

	void setTrainingsMode(const TrainingsConfig& config);

	void readTrainingsModeFromSetting();

	void setValidationSet(LabeledData* pValidation);

	bool isTrained(){ return m_firstTrainingDone; }

private:

	using SortedDecisionTreePair = std::pair<DecisionTreePointer, Real>;
	using SortedDecisionTreeList = std::list<SortedDecisionTreePair>;

	void predictDataInParallel(const Data& points, Labels* labels, SharedPtr<InformationPackage> package,
							   const unsigned int start, const unsigned int end) const;

	void predictClassDataInParallel(const LabeledData& points, Labels* labels, SharedPtr<InformationPackage> package,
									const unsigned int start, const unsigned int end) const;

	void predictDataProbInParallelStartEnd(const Data& points, Labels* labels,
										   std::vector< std::vector<Real> >* probabilities, SharedPtr<InformationPackage> package,
										   const unsigned int start, const unsigned int end) const;

	void predictClassDataProbInParallelStartEnd(const LabeledData& points, Labels* labels, SharedPtr<InformationPackage> package,
												std::vector< std::vector<Real> >* probabilities,
												const unsigned int start, const unsigned int end) const;

	void predictDataProbInParallel(const Data& points, std::vector< std::vector<Real> >* probabilities,
			unsigned int* iBatchNr, Mutex* mutex, DecisionTreeIterator* itOfActElement) const;

	void predictClassDataProbInParallel(const LabeledData& points, std::vector< std::vector<Real> >* probabilities,
			unsigned int* iBatchNr, Mutex* mutex, DecisionTreeIterator* itOfActElement) const;

	void trainInParallel(SharedPtr<RandomNumberGeneratorForDT> generator, SharedPtr<InformationPackage> package,
						 const unsigned int amountOfTrees,
						 SharedPtr<std::vector<std::vector<unsigned int> > > counterForClasses,
						 SharedPtr<Mutex> mutexForCounter);

	void sortTreesAfterPerformance(SortedDecisionTreeList& list);

	void internalAppendToSortedList(SortedDecisionTreeList* list,
									DecisionTreePointer&& pTree, Real acceptance);

	void mergeSortedLists(SortedDecisionTreeList* aimList, SortedDecisionTreeList* other);

	void sortTreesAfterPerformanceInParallel(SortedDecisionTreeList* list, DecisionTreesContainer* trees,
											 SharedPtr<Mutex> readMutex, SharedPtr<Mutex> appendMutex,
											 SharedPtr<InformationPackage> package);

	void updateInParallel(SharedPtr<SortedDecisionTreeList> list, const unsigned int amountOfSteps,
						  SharedPtr<Mutex> mutex, unsigned int threadNr, SharedPtr<InformationPackage> package,
						  SharedPtr<std::pair<unsigned int, unsigned int> > counter,
						  SharedPtr<AcceptanceCalculator> acceptanceCalculator,
						  const unsigned int amountOfForcedRetrain);

	void updateMinMaxValues(unsigned int event);

	void tryAmountForLayers(SharedPtr<RandomNumberGeneratorForDT> generator, const Real secondsPerSplit,
							SharedPtr<std::list<std::pair<unsigned int, unsigned int> > > layerValues,
							SharedPtr<Mutex> mutex, SharedPtr<std::pair<int, int> > bestLayerSplit, SharedPtr<Real> bestAmountOfTrainedTrees,
							SharedPtr<InformationPackage> package);

	void writeTreesToDisk(const unsigned int amountOfTrees) const;

	void loadBatchOfTreesFromDisk(const unsigned int batchNr) const;

	void packageUpdateForPrediction(SharedPtr<InformationPackage>& package, const unsigned int i, const unsigned int start,
									const unsigned int end) const;

	Real calcAccuracyForOneTree(const DynamicDecisionTreeInterface& tree);

	const unsigned int m_maxDepth;

	const unsigned int m_amountOfClasses;

	int m_amountOfPointsUntilRetrain;

	int m_counterForRetrain;

	int m_amountOfUsedDims;

	Real m_factorForUsedDims;

	Vector2 m_minMaxUsedDataFactor;

	// used in all decision trees -> no copies needed!
	std::vector<Vector2 > m_minMaxValues;

	OnlineStorage<LabeledVectorX*>& m_storage;

	LabeledData* m_validationSet;

	mutable DecisionTreesContainer m_trees;

	mutable std::vector<std::pair<std::string, std::string> > m_savedToDiskTreesFilePaths;

	DecisionTreeIterator findWorstPerformingTree(Real& correctAmount);

	std::vector<SharedPtr<RandomNumberGeneratorForDT> > m_generators;

	UniquePtr<RandomNumberGeneratorForDT::BaggingInformation> m_baggingInformation;

	mutable Mutex m_treesMutex;

	bool m_firstTrainingDone;

	bool m_useBigDynamicDecisionTrees;

	std::pair<unsigned int, unsigned int> m_amountOfUsedLayer;

	std::string m_folderForSavedTrees;

	bool m_savedAnyTreesToDisk;

	unsigned int m_amountOfTrainedTrees;

	mutable MemoryType m_usedMemory;

	unsigned int m_amountOfPointsCheckedPerSplit;

	TrainingsConfig m_trainingsConfig;

	const bool m_useRealOnlineUpdate;

	bool m_useOnlinePool;

	std::vector<Real> m_classCounterForValidationSet;

	// are copied to the threads (automatic reference counting for them)
	mutable SharedPtr<Mutex> m_read;
	mutable SharedPtr<Mutex> m_append;
	mutable SharedPtr<Mutex> m_mutexForCounter;
	mutable SharedPtr<Mutex> m_mutexForTrees;

};


#endif /* RANDOMFORESTS_ONLINERANDOMFOREST_H_ */
