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

	/* The given storage is used for the training updates to it, will start the training process
	 * 		The maxDepth parameter specifies how deep each tree can be
	 */
	OnlineRandomForest(OnlineStorage<LabeledVectorX *> &storage,
					   const unsigned int maxDepth, const int amountOfUsedClasses);

	virtual ~OnlineRandomForest() = default;

	/* Train the trees, the criterion to stop is defined in the control file, which is executed by the test manager
	 * 		To maximize the training performance the training is performed multi-threaded.
	 * 		In each thread:
	 * 			A new Tree is trained and added to the forest until the stop criterion is meet
	 * 				See DynamicDecisionTree and BigDynamicDecisionTree for more information about the training
	 * 				of single trees
	 */
	void train();

	/* Predict the label for a given point
	 * 		If you have more than point to predict, try to use predictData is much faster
	 */
	unsigned int predict(const VectorX& point) const override;

	Real predict(const VectorX& point1, const VectorX& point2, const unsigned int sampleAmount) const;

	/* Predict the the partition equality between to points, this is relevant if the Random Forest is used as a kernel
	 * 		Checks for amountOfSamples times, if these two points lay in the same leaf in the tree
	 * 		If the amountOfSamples is bigger than the amount of trees (getNrOfTrees()), than the amountOfSamples is
	 * 		reduced to that number.
	 * 		The uniformNr is used to get the depth until the trees should be parced, the resulting numbers should be
	 * 		below the max depth and above ideally 3.
	 *		The resulting value will be between 0 and 1 and will show how similiar these two points are.
	 */
	Real predictPartitionEquality(const VectorX& point1, const VectorX& point2,
									RandomUniformNr& uniformNr, unsigned int amountOfSamples) const;

	/* Predict the labels for the given points (multi-threaded)
	 */
	void predictData(const Data& points, Labels& labels) const override;

	/* Predict the labels and the probabilities for the given points (multi-threaded)
	 * 		The probabilities are stored like this, each point gets its own vector with real values,
	 * 		where each values corresponds to the likelihood that this class is correct
	 */
	void predictData(const Data& points, Labels& labels,
					 std::vector< std::vector<Real> >& probabilities) const override;

	/* Predict the labels for the given points (multi-threaded)
	 * 		The start value can be used if not all points of the points storage are used
	 * 		The labels storage will have in the end the size = points.size() - start
	 */
	void predictData(const LabeledData& points, Labels& labels, const unsigned int start = 0) const;

	/* Predict the labels and the probabilities for the given points (multi-threaded)
	 * 		The probabilities are stored like this, each point gets its own vector with real values,
	 * 		where each values corresponds to the likelihood that this class is correct
	 */
	void predictData(const LabeledData& points, Labels& labels, std::vector< std::vector<Real> >& probabilities) const;

	/* Returns the number of trained trees in the forest, this value is updated during the training
	 *
	 */
	int getNrOfTrees() const { return m_amountOfTrainedTrees; };

	/* This function is automatically called by the connected storage, if a change to storage happens
	 * 		Depending of the kind of event a certain action is then performed
	 */
	void update(Subject* caller, unsigned int event) override;

	/* Update procedure (this function performs a initial training on the first call and
	 * only an update on every call after that)
	 * The initial training is described above the function train()
	 * The update step is structured as:
	 *		* if the pool is used:
	 * 			* calculate the performance on the validation set (V),
	 * 			  if there is none available the training set is used
	 * 			* calculate the performance of each class (how many points of a class are correctly predicted)
	 * 			* change the pool for each of these classes accordingly
	 * 		* sort all trees in parallel after their acceptance on the validation set
	 * 			* the acceptance depends on the acceptance calculator and the settings
	 * 		* start the update procedure in parallel in each thread:
	 * 			* at first the fixed amount of trees are retrained specified in the settings file
	 * 				* it is wise to use a value, so that after a certain amount of iterations
	 * 				  the half of the trees are replace
	 * 			* after this new trees are trained and compared to the tree with the worst acceptance rate
	 * 			  the better performing tree is kept
	 * 			* this is repeated until the break criterion is meet (specified in the settings)
	 * 	At the end true is returned (not used at the moment)
	 *
	 * 	This function is mainly called by void update(Subject* caller, unsigned int event) override;
	 *
	 * 	Calling it on its own is supported but unwise, adding points to the used storage will already call
	 * 	this function on its own
	 */
	bool update();


	/* Returns the amount of classes used here
	 */
	unsigned int amountOfClasses() const override;

	/* Returns a reference to the used storage
	 */
	OnlineStorage<LabeledVectorX*>& getStorageRef();

	const OnlineStorage<LabeledVectorX*>& getStorageRef() const;

	/* Returns the class type this is only relevant for the subject and observer pattern
	 */
	ClassTypeSubject classType() const override;

	/* Get a const reference to min and max value for each dimension of the used storage
	 */
	const std::vector<Vector2>& getMinMaxValues(){ return m_minMaxValues;};

	/* The TrainingsConfig is used to specify the breaking condition for the training
	 * 		Several options are available:
	 * 			1. Break after a certain period of time (TIME, TIME_WITH_MEMORY)
	 * 			2. Break after a certain amount of trained trees (TREEAMOUNT, TREEAMOUNT_WITH_MEMORY)
	 * 			3. Break after a certain amount of used memory (this one can be combined with the first and second one)
	 * 				(MEMORY, TIME_WITH_MEMORY, TREEAMOUNT_WITH_MEMORY)
	 * 		Can be specified in the settings file as well, use function of ORF -> readTrainingsModeFromSetting()
	 */
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

	/* Sets the breaking criterion, which is saved in the TrainingsConfig, can be switched after each update step
	 */
	void setTrainingsMode(const TrainingsConfig& config);

	/* Read the trainings mode, which specify the breaking criterion, from the settings
	 */
	void readTrainingsModeFromSetting();

	/* Set the validation set, if none is used the training set is used a validation set
	 * 		The settings contain a parameter, which adds a possible validation set to the trainings storage
	 */
	void setValidationSet(LabeledData* pValidation);

	/* Is the first training already done, to check if an update to the used storage can be performed
	 */
	bool isTrained(){ return m_firstTrainingDone; }

private:

	using SortedDecisionTreePair = std::pair<DecisionTreePointer, Real>;
	using SortedDecisionTreeList = std::list<SortedDecisionTreePair>;

	void predictDataInParallel(const Data& points, Labels* labels, SharedPtr<InformationPackage> package,
							   const unsigned int start, const unsigned int end) const;

	void predictClassDataInParallel(const LabeledData& points, Labels* labels, SharedPtr<InformationPackage> package,
									const unsigned int start, const unsigned int end, const unsigned int offset) const;

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

	// TODO get rid of this parameter -> replace it with look up in ClassKnowledge or make it adaptable to change
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

	// are copied to the threads (automatic reference counting for them)
	mutable SharedPtr<Mutex> m_read;
	mutable SharedPtr<Mutex> m_append;
	mutable SharedPtr<Mutex> m_mutexForCounter;
	mutable SharedPtr<Mutex> m_mutexForTrees;

};


#endif /* RANDOMFORESTS_ONLINERANDOMFOREST_H_ */
