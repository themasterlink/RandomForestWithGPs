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

class OnlineRandomForest : public Observer, public PredictorMultiClass, public Subject {
public:

	using DecisionTreesContainer = std::list<DynamicDecisionTreeInterface*>;
	using DecisionTreeIterator = DecisionTreesContainer::iterator;
	using DecisionTreeConstIterator = DecisionTreesContainer::const_iterator;

	OnlineRandomForest(OnlineStorage<LabeledVectorX *> &storage,
					   const unsigned int maxDepth, const int amountOfUsedClasses);

	virtual ~OnlineRandomForest();

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

	void setValidationSet(LabeledData* pValidation){ m_validationSet = pValidation; }

private:

	using SortedDecisionTreePair = std::pair<DynamicDecisionTreeInterface*, Real>;
	using SortedDecisionTreeList = std::list<SortedDecisionTreePair>;

	void predictDataInParallel(const Data& points, Labels* labels, InformationPackage* package,
							   const unsigned int start, const unsigned int end) const;

	void predictClassDataInParallel(const LabeledData& points, Labels* labels, InformationPackage* package,
									const unsigned int start, const unsigned int end) const;

	void predictDataProbInParallelStartEnd(const Data& points, Labels* labels,
										   std::vector< std::vector<Real> >* probabilities, InformationPackage* package,
										   const unsigned int start, const unsigned int end) const;

	void predictClassDataProbInParallelStartEnd(const LabeledData& points, Labels* labels, InformationPackage* package,
												std::vector< std::vector<Real> >* probabilities,
												const unsigned int start, const unsigned int end) const;

	void predictDataProbInParallel(const Data& points, std::vector< std::vector<Real> >* probabilities,
			unsigned int* iBatchNr, boost::mutex* mutex, DecisionTreeIterator* itOfActElement) const;

	void predictClassDataProbInParallel(const LabeledData& points, std::vector< std::vector<Real> >* probabilities,
			unsigned int* iBatchNr, boost::mutex* mutex, DecisionTreeIterator* itOfActElement) const;

	void trainInParallel(RandomNumberGeneratorForDT* generator, InformationPackage* package, const unsigned int amountOfTrees,
						 std::vector<std::vector<unsigned int> >* counterForClasses,
						 boost::mutex* mutexForCounter);

	void sortTreesAfterPerformance(SortedDecisionTreeList& list);

	void internalAppendToSortedList(SortedDecisionTreeList* list,
									DynamicDecisionTreeInterface* pTree, Real correctVal);

	void mergeSortedLists(SortedDecisionTreeList* aimList, SortedDecisionTreeList* other);

	void sortTreesAfterPerformanceInParallel(SortedDecisionTreeList* list, DecisionTreesContainer* trees,
											 boost::mutex* readMutex, boost::mutex* appendMutex,
											 InformationPackage* package);

	void updateInParallel(SortedDecisionTreeList* list, const unsigned int amountOfSteps,
						  boost::mutex* mutex, unsigned int threadNr, InformationPackage* package,
						  unsigned int* counter, const Real standartDeviation);

	void updateMinMaxValues(unsigned int event);

	void tryAmountForLayers(RandomNumberGeneratorForDT* generator, const Real secondsPerSplit,
							std::list<std::pair<unsigned int, unsigned int> >* layerValues,
							boost::mutex* mutex, std::pair<int, int>* bestLayerSplit, Real* bestCorrectness);

	void writeTreesToDisk(const unsigned int amountOfTrees) const;

	void loadBatchOfTreesFromDisk(const unsigned int batchNr) const;

	void packageUpdateForPrediction(InformationPackage* package, const unsigned int i, const unsigned int start,
									const unsigned int end) const;

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

	std::vector<RandomNumberGeneratorForDT*> m_generators;

	std::unique_ptr<RandomNumberGeneratorForDT::BaggingInformation> m_baggingInformation;

	mutable boost::mutex m_treesMutex;

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

	mutable std::unique_ptr<boost::mutex> m_read;
	mutable std::unique_ptr<boost::mutex> m_append;
	mutable std::unique_ptr<boost::mutex> m_mutexForCounter;
	mutable std::unique_ptr<boost::mutex> m_mutexForTrees;

};


#endif /* RANDOMFORESTS_ONLINERANDOMFOREST_H_ */
