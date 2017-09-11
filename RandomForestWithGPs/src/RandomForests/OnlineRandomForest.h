/**
 *  OnlineRandomForest.h
 *
 *   Created on: 13.10.2016
 *       Author: Max
 */

#ifndef RANDOMFORESTS_ONLINERANDOMFOREST_H_
#define RANDOMFORESTS_ONLINERANDOMFOREST_H_

#include "BigDynamicDecisionTree.h"
#include "../Data/OnlineStorage.h"
#include "../Data/LabeledVectorX.h"
#include "../RandomNumberGenerator/RandomUniformNr.h"
#include "TreeCounter.h"
#include "AcceptanceCalculator.h"

/**
 * \brief The online random forest, the initial training and the updates are activated over the updates of the connected
 * 		Online Storage. In the settings file a lot of things about the forest can be specified.
 */
class OnlineRandomForest : public Observer, public PredictorMultiClass, public Subject {
public:

	/**
	 * \brief Pointer to one of the Trees
	 */
	using DecisionTreePointer = SharedPtr<DynamicDecisionTreeInterface>;
	/**
	 * \brief List of decision trees
	 */
	using DecisionTreesContainer = std::list<DecisionTreePointer>;
	/**
	 * \brief Iterator for the DecisionTreesContainer
	 */
	using DecisionTreeIterator = DecisionTreesContainer::iterator;
	/**
	 * \brief Const iterator for the DecisionTreesContainer
	 */
	using DecisionTreeConstIterator = DecisionTreesContainer::const_iterator;

	/**
	 * \brief The given storage is used for the training updates to it, will start the training process.
	 *
	 *  		The maxDepth parameter specifies how deep each tree can be.
	 * \param storage which is used for updating the training procedure
	 * \param maxDepth which defines the max depth of the trees
	 * \param amountOfUsedClasses the total amount of used classes in the RF (can be bigger than the actual amount)
	 */
	OnlineRandomForest(OnlineStorage<LabeledVectorX *> &storage,
					   const unsigned int maxDepth, const int amountOfUsedClasses);

	virtual ~OnlineRandomForest() = default;

	/**
	 * \brief Train the trees for the first time.
	 *
	 * 		The criterion to stop is defined in the control file, which is executed by the test manager.
	 *  		To maximize the training performance the training is performed multi-threaded.
	 *  		In each thread:
	 *  			A new Tree is trained and added to the forest until the stop criterion is meet
	 *  				See DynamicDecisionTree and BigDynamicDecisionTree for more information about the training
	 *  				of single trees
	 */
	void train();

	/**
	 * \brief Predict the label for a given point.
	 *
	 * 		If you have more than point to predict, try to use predictData is much faster
	 * \param point predict the class of this point
	 * \return the class of the given point
	 */
	unsigned int predict(const VectorX& point) const override;
	/**
	 * \brief Predict the similarity between two points
	 *
	 * 		By comparing how often these two points have the same class in a tree
	 * \param point1 first point to use
	 * \param point2 second point to use
	 * \param sampleAmount how often the sampling should be done, this amount can not be bigger than the total amount of trees
	 * \return the similarity between the two points
	 */
	Real predict(const VectorX& point1, const VectorX& point2, const unsigned int sampleAmount) const;

	/**
	 * \brief Predict the the partition equality between to points. This is relevant if the Random Forest is used as a kernel
	 *  		Checks for amountOfSamples times, if these two points lay in the same leaf in the tree.
	 *
	 *  		If the amountOfSamples is bigger than the amount of trees (getNrOfTrees()), than the amountOfSamples is
	 *  		reduced to that number.
	 *  		The uniformNr is used to get the depth until the trees should be parced, the resulting numbers should be
	 *  		below the max depth and above ideally 3.
	 *		The resulting value will be between 0 and 1 and will show how similiar these two points are.
	 * \return the similarity between the input points
	 */
	Real predictPartitionEquality(const VectorX& point1, const VectorX& point2,
									RandomUniformNr& uniformNr, unsigned int amountOfSamples) const;

	/**
	 * \brief Predict the labels for the given points (multi-threaded).
	 */
	void predictData(const Data& points, Labels& labels) const override;

	/**
	 * \brief Predict the labels and the probabilities for the given points (multi-threaded).
	 *
	 *  		The probabilities are stored like this, each point gets its own vector with real values,
	 *  		where each values corresponds to the likelihood that this class is correct
	 */
	void predictData(const Data& points, Labels& labels,
					 std::vector< std::vector<Real> >& probabilities) const override;

	/**
	 * \brief Predict the labels for the given points (multi-threaded).
	 *  		The start value can be used if not all points of the points storage are used
	 *  		The labels storage will have in the end the size = points.size() - start
	 */
	void predictData(const LabeledData& points, Labels& labels, const unsigned int start = 0) const;

	/**
	 * \brief Predict the labels and the probabilities for the given points (multi-threaded).
	 *  		The probabilities are stored like this, each point gets its own vector with real values,
	 *  		where each values corresponds to the likelihood that this class is correct
	 */
	void predictData(const LabeledData& points, Labels& labels, std::vector< std::vector<Real> >& probabilities) const;

	/**
	 * \brief Returns the number of trained trees in the forest, this value is updated during the training.
	 *
	 * \return the amount of trained trees
	 */
	int getNrOfTrees() const { return m_amountOfTrainedTrees; };

	/**
	 * \brief This function is automatically called by the connected storage, if a change to storage happens.
	 *  		Depending of the kind of event a certain action is then performed
	 */
	void update(Subject* caller, unsigned int event) override;

	/**
	 * \brief Update procedure (this function performs a initial training on the first call and
	 *  only an update on every call after that).
	 *
	 *  The initial training is described above the function train()
	 *  The update step is structured as:
	 *		* if the pool is used:
	 *  			* calculate the performance on the validation set (V),
	 *  			  if there is none available the training set is used
	 *  			* calculate the performance of each class (how many points of a class are correctly predicted)
	 *  			* change the pool for each of these classes accordingly
	 *  		* sort all trees in parallel after their acceptance on the validation set
	 *  			* the acceptance depends on the acceptance calculator and the settings
	 *  		* start the update procedure in parallel in each thread:
	 *  			* at first the fixed amount of trees are retrained specified in the settings file
	 *  				* it is wise to use a value, so that after a certain amount of iterations
	 *  				  the half of the trees are replace
	 *  			* after this new trees are trained and compared to the tree with the worst acceptance rate
	 *  			  the better performing tree is kept
	 *  			* this is repeated until the break criterion is meet (specified in the settings)
	 *
	 *  	This function is mainly called by void update(Subject* caller, unsigned int event) override;
	 *
	 *  	Calling it on its own is supported but unwise, adding points to the used storage will already call
	 *  	this function on its own
	 * \return always true (not used at the moment)
	 */
	bool update();


	/**
	 * \brief Returns the amount of classes used here.
	 *
	 * \return the amount of classes
	 */
	unsigned int amountOfClasses() const override;

	/** Returns a reference to the used storage.
	 */
	OnlineStorage<LabeledVectorX*>& getStorageRef();

	const OnlineStorage<LabeledVectorX*>& getStorageRef() const;

	/** Returns the class type this is only relevant for the subject and observer pattern.
	 */
	ClassTypeSubject classType() const override;

	/** Get a const reference to min and max value for each dimension of the used storage.
	 */
	const std::vector<Vector2>& getMinMaxValues(){ return m_minMaxValues;};

	/** The TrainingsConfig is used to specify the breaking condition for the training.
	 *  		Several options are available:
	 *  			1. Break after a certain period of time (TIME, TIME_WITH_MEMORY)
	 *  			2. Break after a certain amount of trained trees (TREEAMOUNT, TREEAMOUNT_WITH_MEMORY)
	 *  			3. Break after a certain amount of used memory (this one can be combined with the first and second one)
	 *  				(MEMORY, TIME_WITH_MEMORY, TREEAMOUNT_WITH_MEMORY)
	 *  		Can be specified in the settings file as well, use function of ORF -> readTrainingsModeFromSetting()
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

	/** Sets the breaking criterion, which is saved in the TrainingsConfig, can be switched after each update step.
	 */
	void setTrainingsMode(const TrainingsConfig& config);

	/** Read the trainings mode, which specify the breaking criterion, from the settings.
	 */
	void readTrainingsModeFromSetting();

	/** Set the validation set, if none is used the training set is used a validation set.
	 *  		The settings contain a parameter, which adds a possible validation set to the trainings storage
	 */
	void setValidationSet(LabeledData* pValidation);

	/** Is the first training already done, to check if an update to the used storage can be performed.
	 */
	bool isTrained(){ return m_firstTrainingDone; }

private:

	/** For sorting all trees after their performance/acceptance this is calculated in the Acceptance Calculator.
	 */
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

	/** This functions implements an insertion sort where a new with an acceptance is added to the given list.
	 */
	void internalAppendToSortedList(SortedDecisionTreeList* list,
									DecisionTreePointer&& pTree, Real acceptance);

	/** Sort the other list per merge sort into the aimList, this means we walk over both sorted listes and always add
	 *  a value if it is bigger than the last existing value, until we added all values to the aimList.
	 */
	void mergeSortedLists(SortedDecisionTreeList* aimList, SortedDecisionTreeList* other);

	void sortTreesAfterPerformanceInParallel(SortedDecisionTreeList* list, DecisionTreesContainer* trees,
											 SharedPtr<Mutex> readMutex, SharedPtr<Mutex> appendMutex,
											 SharedPtr<InformationPackage> package);

	void updateInParallel(SharedPtr<SortedDecisionTreeList> list, const unsigned int amountOfSteps,
						  SharedPtr<Mutex> mutex, unsigned int threadNr, SharedPtr<InformationPackage> package,
						  SharedPtr<std::pair<unsigned int, unsigned int> > counter,
						  SharedPtr<AcceptanceCalculator> acceptanceCalculator,
						  const unsigned int amountOfForcedRetrain);

	/** When new points are added to the storage, this method calculates for each dimension the min and max value on the
	 *  current storage.
	 */
	void updateMinMaxValues(unsigned int event);

	void tryAmountForLayers(SharedPtr<RandomNumberGeneratorForDT> generator, const Real secondsPerSplit,
							SharedPtr<std::list<std::pair<unsigned int, unsigned int> > > layerValues,
							SharedPtr<Mutex> mutex, SharedPtr<std::pair<int, int> > bestLayerSplit, SharedPtr<Real> bestAmountOfTrainedTrees,
							SharedPtr<InformationPackage> package);

	void writeTreesToDisk(const unsigned int amountOfTrees) const;

	void loadBatchOfTreesFromDisk(const unsigned int batchNr) const;

	void packageUpdateForPrediction(SharedPtr<InformationPackage>& package, const unsigned int i, const unsigned int start,
									const unsigned int end) const;

	/** Calculate the accuracy for one tree, the accuarcy is measured between 0.0 and 1.0.
	 *  		0.0 = None of the points could be classified correctly
	 *  		1.0 = All points are correctly classified
	 *  Be aware that the storage on which this tree is tested depends on the used settings:
	 *  		If a validation set is available, it is used to calculate the accuracy
	 *  		Else the training set is used, how the training set is defined depends on the current training mode
	 *  			(Pure Online (only the last points), Incremental Adaptive (all points since the start), Pool
	 *  			(online the points in the pool)
	 */
	Real calcAccuracyForOneTree(const DynamicDecisionTreeInterface& tree);

	const unsigned int m_maxDepth;

	// TODO get rid of this parameter -> replace it with look up in ClassKnowledge or make it adaptable to change
	const unsigned int m_amountOfClasses;

	/** This counter defines at which point a new update iteration should be started. This means after how many added
	 *  points a new update step is performed.
	 */
	int m_amountOfPointsUntilRetrain;

	/** The counter for the amount of points until retrain, for each new point which is added this counter is increased.
	 */
	int m_counterForRetrain;

	/** Amount of available dimension, must fit the length of the vector of the input points.
	 */
	int m_amountOfUsedDims;

	/** How many of the available dims are used in each tree:
	 *  		1.0 = all dimensions are used
	 *  		0.0 = none are used (does not work)
	 *  		0.0 < x < 1.0 = the amount of dimension used relative to the total amount of dimensions
	 *  		anything else can not be used
	 */
	Real m_factorForUsedDims;

	Vector2 m_minMaxUsedDataFactor;

	// used in all decision trees -> no copies needed.
	std::vector<Vector2 > m_minMaxValues;

	/** Reference to the used storage, any change to it will call the update function of the this class.
	 */
	OnlineStorage<LabeledVectorX*>& m_storage;

	/** The validation set can be set and if it is set it is used during the training for the validation of the trees.
	 */
	LabeledData* m_validationSet;

	/** Container which contains all used trees, is not valid during an update step, because of the multi-thread
	 *  architecture, trees are removed and updated during it.
	 */
	mutable DecisionTreesContainer m_trees;

	mutable std::vector<std::pair<std::string, std::string> > m_savedToDiskTreesFilePaths;

	DecisionTreeIterator findWorstPerformingTree(Real& correctAmount);

	/** Random Numbers generators for the training and each update iteration, there are only generated during the first
	 *  training and after that only reused, there is more than one because of the multi-thread capabilities of this
	 *  approach.
	 *  A Shared Pointer is used, because a thread gets a copy of the shared pointer, which increases the counter during
	 *  the execution of the function and decreases it at the end. This guarentees that the object is not destroyed
	 *  before every execution path is done with it.
	 */
	std::vector<SharedPtr<RandomNumberGeneratorForDT> > m_generators;

	UniquePtr<RandomNumberGeneratorForDT::BaggingInformation> m_baggingInformation;

	mutable Mutex m_treesMutex;

	/** Is the first training done (true = done)
	 */
	bool m_firstTrainingDone;

	/** Defines if the Big Dynamic Decision Trees are used or not (true = use them)
	 */
	bool m_useBigDynamicDecisionTrees;

	/** Specifies the configuration for Big Dynamic Decision Trees, the first param specifies the amount of fast layers
	 *  and the second the amount of small layers
	 */
	std::pair<unsigned int, unsigned int> m_amountOfUsedLayer;

	/** Correspond to the variable m_savedAnyTreesToDisk, determines the location where the trees are stored
	 */
	std::string m_folderForSavedTrees;

	/** If this mode is activated a portion of the trees is written to the disk, however this increases the training
	 *  and predicition time in a big way, not all functions support this mode yet
	 */
	bool m_savedAnyTreesToDisk;

	/** Contains the amount of trained trees, this value is updated during the initial training.
	 */
	unsigned int m_amountOfTrainedTrees;

	/** Amount of used Memory of this forest, contains the memory demand for each tree.
	 */
	mutable MemoryType m_usedMemory;

	/** Amount of points checked in each split of a generated Decision tree
	 */
	unsigned int m_amountOfPointsCheckedPerSplit;

	/** Contains the breaking mode for the next training step, can be changed during the training,
	 *  after each full update step, ideally for the initial training a different goal is set then for
	 *  the following update steps.
	 */
	TrainingsConfig m_trainingsConfig;


	/** If a real update step is used, that means not all old points are available, see bool m_useOnlinePool
	 *  for more information.
	 */
	const bool m_useRealOnlineUpdate;

	/** If the pool is used:
	 *  	 Be aware there are only three different modes here:
	 *  	 		1. Pure online, (m_useRealOnlineUpdate = true, m_useOnlinePool = false)
	 *  	 		2. Adaptive incremental, (m_useRealOnlineUpdate = false, m_useOnlinePool = false)
	 *  			3. Pool learning, (m_useRealOnlineUpdate = true, m_useOnlinePool = true)
	 *			4. Not valid!, (m_useRealOnlineUpdate = false, m_useOnlinePool = true), the pool can not be used if
	 *				the online update method is false
	 */
	bool m_useOnlinePool;

	// are copied to the threads (automatic reference counting for them)
	mutable SharedPtr<Mutex> m_read;
	mutable SharedPtr<Mutex> m_append;
	mutable SharedPtr<Mutex> m_mutexForCounter;
	mutable SharedPtr<Mutex> m_mutexForTrees;

};


#endif /** RANDOMFORESTS_ONLINERANDOMFOREST_H_ */
