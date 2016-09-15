/*
 * GaussianProcessMultiBinary.cc
 *
 *  Created on: 21.07.2016
 *      Author: Max
 */

#include "GaussianProcessMultiBinary.h"
#include "../Data/DataConverter.h"
#include "../Utility/Settings.h"
#include "../GaussianProcess/BayesOptimizer.h"

GaussianProcessMultiBinary::GaussianProcessMultiBinary(int amountOfUsedClasses):
	m_amountOfUsedClasses(amountOfUsedClasses),
	m_amountOfDataPoints(0),
	m_amountOfDataPointsForUseAllTestsPoints(300),
	m_maxPointsUsedInGpSingleTraining(1000),
	m_pClassNames(NULL),
	m_gps(amountOfUsedClasses, NULL){
}

void GaussianProcessMultiBinary::train(const DataContainer& container, const Labels* guessedLabels){
	m_amountOfDataPoints = container.amountOfPoints;
	// count the occurence of each pre class of the random forest
	const Labels& refGuessedLabels = guessedLabels != NULL ? *guessedLabels : container.labels;
	std::vector<int> countClasses(m_amountOfUsedClasses, 0);
	for(int i = 0; i < m_amountOfDataPoints; ++i){
		countClasses[refGuessedLabels[i]] += 1;
	}
	m_amountOfDataPointsForUseAllTestsPoints = 300;
	m_pClassNames = const_cast<std::vector<std::string>* >(&(container.namesOfClasses));
	int thresholdForNoise = 5;
	Settings::getValue("RFGP.thresholdForNoise", thresholdForNoise);
	int pointsPerClassForBayOpt = 16;
	Settings::getValue("RFGP.pointsPerClassForBayOpt", pointsPerClassForBayOpt);
	m_maxPointsUsedInGpSingleTraining = 1500;
	Settings::getValue("RFGP.maxPointsUsedInGpSingleTraining", m_maxPointsUsedInGpSingleTraining);
	int maxNrOfPointsForBayesOpt = 250;
	Settings::getValue("RFGP.maxNrOfPointsForBayesOpt", maxNrOfPointsForBayesOpt);
	boost::thread_group group;
	for(int iActClass = 0; iActClass < m_amountOfUsedClasses; ++iActClass){
		if(countClasses[iActClass] < thresholdForNoise){
			m_output.printSwitchingColor(container.namesOfClasses[iActClass] + " is not used, because count class is: " + number2String(countClasses[iActClass]) + "!");
			continue; // do not use this class
		}
		m_output.printSwitchingColor("In Class: " + container.namesOfClasses[iActClass] + " has so many points: " + number2String(countClasses[iActClass]));
		//m_isGpInUse[iActRfRes][iActClass] = true; // there is actually a gp for this config

		m_gps[iActClass] = new GaussianProcess();

		while(!m_threadCounter.addNewThread()){ // if it is not possible wait
			usleep(0.35 * 1e6);
		}
		group.add_thread(new boost::thread(boost::bind(&GaussianProcessMultiBinary::trainInParallel, this, iActClass,
				min(maxNrOfPointsForBayesOpt, pointsPerClassForBayOpt * m_amountOfUsedClasses),
				container, countClasses, m_gps[iActClass])));
		/*trainInParallel(iActClass, amountOfDataInRfRes,
							pointsPerClassForBayOpt, amountOfClassesOverThreshold,
							maxPointsUsedInGpSingleTraining, dataOfActRf,
							labelsOfActRf, classCounts, actGp);*/
	}
	group.join_all();
	int c = 0;
	for(int i = 0; i < m_amountOfUsedClasses; ++i){
		c += m_gps[i] != NULL? 1 : 0;
	}
	std::cout << "Amount of gps: " << c << std::endl;
}

void GaussianProcessMultiBinary::trainInParallel(const int iActClass,
		const int amountOfHyperPoints, const DataContainer& container,
		const std::vector<int>& classCounts, GaussianProcess* actGp) {
	int nrOfNoChanges;
	Settings::getValue("RFGP.nrOfNoChanges", nrOfNoChanges);
	Eigen::MatrixXd dataMat;
	Eigen::VectorXd yGpInit;
	// calc for final training
	std::vector<bool> usedElements;
	std::vector<bool> blockElements(container.data.size(), false); // block none
	DataConverter::toRandClassAndHalfUniformDataMatrix(container.data, container.labels, classCounts, dataMat, yGpInit, m_maxPointsUsedInGpSingleTraining, iActClass, usedElements, blockElements); // get a uniform portion of at most 1000 points

	Eigen::MatrixXd testDataMat;
	Eigen::VectorXd testYGpInit;
	const int amountOfPointsUsedForTrainingTheTest = 300; // min(max(200,container.amountOfPoints / 3),m_maxPointsUsedInGpSingleTraining / 3); TODO
	// calc for final training
	std::vector<bool> usedElementsForTheValidationSet; // save the blocked values, for the testing of the hyper params
	std::vector<bool> blockElementsForValidationSet(container.data.size(), false); // block none
	DataConverter::toRandClassAndHalfUniformDataMatrix(container.data, container.labels, classCounts, testDataMat, testYGpInit, amountOfPointsUsedForTrainingTheTest, iActClass, usedElementsForTheValidationSet, blockElementsForValidationSet); // get a uniform portion of at most 1000 points

	// compare to all other classes! // one vs. all
	const int numberOfPointsForClass = classCounts[iActClass];
	const std::string betweenNames = ", for " + (*m_pClassNames)[iActClass] + " has " + number2String(numberOfPointsForClass);
	m_output.printSwitchingColor("Start parallel with " + number2String(amountOfPointsUsedForTrainingTheTest) + " amount of points, which are used in the training for the testing" + betweenNames);
	//Eigen::VectorXd y(numberOfPointsForClass);
	StopWatch sw;
	BestHyperParams bestHyperParams(20);
	boost::thread_group group;
	bool isFinish = false;
	int iCounter = 0;
	while(!isFinish){
		bool addNewThread = false;
		if(iCounter == 0){
			addNewThread = true;
		}else{ // only add if iCounter != 0
			if(m_threadCounter.addNewThread())
				addNewThread = true;
		}
		if(addNewThread){ // if adding is possible
			// create a new one for this problem
			group.add_thread(new boost::thread(boost::bind(&GaussianProcessMultiBinary::optimizeHyperParams, this,
					iActClass, amountOfHyperPoints, container, classCounts, usedElementsForTheValidationSet, testDataMat, testYGpInit, &bestHyperParams)));
			m_output.printInColor("At the moment are " + number2String(m_threadCounter.currentThreadCount()) + " threads running" + betweenNames, RESET);
		}
		bestHyperParams.getFinishLast(isFinish);
		if(!isFinish){
			usleep(0.35 * 1e6);
		}
		++iCounter;
	}
	group.join_all();
	m_output.printInColor("Finish optimizing with in: " + sw.elapsedAsPrettyTime()
					+ ", with: " + bestHyperParams.prettyStringOfBest()
					+ betweenNames, MAGENTA);
	sw.startTime();
	// set hyper params
	double len, sigmaF;
	bestHyperParams.getBestHypParams(len,sigmaF);
	actGp->getKernel().setHyperParams(len, sigmaF, actGp->getKernel().sigmaN());
	// train on whole data set
	actGp->init(dataMat,yGpInit);
	m_output.printSwitchingColor("Finish init for " + number2String(yGpInit.rows()) + " elements in: " + sw.elapsedAsPrettyTime() + betweenNames);
	sw.startTime();
	actGp->trainWithoutKernelOptimize();
	m_output.printSwitchingColor("Finish training for " + number2String(yGpInit.rows()) + " elements in: " + sw.elapsedAsPrettyTime() + betweenNames);
	m_threadCounter.removeThread();
}

void GaussianProcessMultiBinary::optimizeHyperParams(const int iActClass,
		const int amountOfHyperPoints, const DataContainer& container,
		const std::vector<int>& classCounts, const std::vector<bool>& elementsUsedForValidation,
		const Eigen::MatrixXd& testDataMat, const Eigen::VectorXd& testYGpInit, BestHyperParams* bestHyperParams){
	GaussianProcess usedGp;
	const int numberOfPointsForClass = classCounts[iActClass];
	const std::string betweenNames = ", for " + (*m_pClassNames)[iActClass] + " has " + number2String(numberOfPointsForClass);
	int noChange = 1;
	const bool useAllTestValues = true; //m_amountOfDataPoints <= m_amountOfDataPointsForUseAllTestsPoints; TODO
	int size = min(m_amountOfDataPointsForUseAllTestsPoints, m_amountOfDataPoints);
	bool isFinished = false;
	bestHyperParams->getNoChangeCounter(noChange);
	vectord lowerBound(2); // for hyper params in bayesian optimization
	lowerBound[0] = 1.0;
	lowerBound[1] = 0.5;
	vectord upperBound(2);
	upperBound[0] = 17; //;actGp->getKernel().getLenVar() / 3;
	upperBound[1] = 2.5;
	while(!isFinished){
		Eigen::MatrixXd dataHyper;
		Eigen::VectorXd yHyper;
		std::vector<bool> usedElements;
		DataConverter::toRandClassAndHalfUniformDataMatrix(container.data, container.labels, classCounts, dataHyper, yHyper, min(100, noChange * amountOfHyperPoints), iActClass, usedElements, elementsUsedForValidation); // get a uniform portion of all points

		/* copy the #hyperPoints points of both TODO find way of randomly taking the values
int oneCounter = 0, minusCounter = 0, counterHyper = 0;
if(amountOfDataInRfRes > hyperPoints){
	for(int j = 0; j < amountOfDataInRfRes; ++j){
		if(sortedLabels[iActRfRes][j] == iActClass){
			y[j] = 1;
			isThere = true;
			if(oneCounter < hyperPoints / 2 && counterHyper < hyperPoints){
				dataHyper.col(counterHyper) = dataOfActRf[j];
				yHyper[counterHyper] = 1;
				++counterHyper;
				++oneCounter;
			}
		}else{
			y[j] = -1;
			if(minusCounter < hyperPoints / 2 && counterHyper < hyperPoints){
				dataHyper.col(counterHyper) = dataOfActRf[j];
				yHyper[counterHyper] = -1;
				++minusCounter;
				++counterHyper;
			}
		}
	}
}
if(minusCounter < hyperPoints / 2 || oneCounter < hyperPoints / 2 ){
	// reduce the number of hyperpoints, not enough counter parts there!
	dataHyper.resize(dim, minusCounter + oneCounter);
	yHyper.resize(minusCounter + oneCounter);
}
if(yHyper.rows() < hyperPoints * 0.75){
	m_pureClassLabelForRfClass[iActRfRes] = idOfMaxClass;
	std::cout << "to less points -> make it pure! amount of points: " << yHyper.rows() << std::endl;
	continue;
}
for(int i = 0; i < yHyper.rows(); ++i){
	std::cout << (double) yHyper[i] << std::endl;
}
std::cout << "One: " << oneCounter << std::endl;
		 */
		// find good hyperparameters with bayesian optimization:
		bestHyperParams->getFinishDuring(isFinished); // only if one result is above the testing mark
		if(isFinished){ break; }
		vectord result(2);
		usedGp.init(dataHyper, yHyper);
		bayesopt::Parameters par = initialize_parameters_to_default();
		par.noise = 1e-12;
		par.epsilon = 0.2;
		par.verbose_level = 6;
		par.n_iterations = 200;
		par.surr_name = "sGaussianProcessML";
		BayesOptimizer bayOpt(usedGp, par);
		bayOpt.setBoundingBox(lowerBound, upperBound);
		try{
			bestHyperParams->getFinishDuring(isFinished); // only if one result is above the testing mark
			if(isFinished){ break; }
			bayOpt.optimize(result);
		}catch(std::runtime_error& e){
			upperBound[1] = 1.5; // reduce noice!
			m_output.printSwitchingColor(e.what() + betweenNames);
			continue;
		}
		usedGp.getKernel().setHyperParams(result[0], result[1], usedGp.getKernel().sigmaN());
		bestHyperParams->getFinishDuring(isFinished); // only if one result is above the testing mark
		if(isFinished){ break; }
		usedGp.init(testDataMat, testYGpInit);
		bestHyperParams->getFinishDuring(isFinished); // only if one result is above the testing mark
		if(isFinished){ break; }
		usedGp.trainWithoutKernelOptimize();
		bestHyperParams->getFinishDuring(isFinished); // only if one result is above the testing mark
		if(isFinished){ break; }
		int wright = 0;
		int rr = 0;
		int amountOfCorrectLabels = 0;
		int amountOfUsedValues = 0;
		if(!useAllTestValues){
			for(int i = 0; i < size; ++i){
				const int nextEle = rand() / (double) RAND_MAX * m_amountOfDataPoints;
				if(!elementsUsedForValidation[nextEle]){ // is not part of the validation trainings part
					double prob = usedGp.predict(container.data[nextEle], 500);
					if(prob > 0.5 && container.labels[nextEle] == iActClass){
						++wright;
					}else if(prob < 0.5 && container.labels[nextEle] != iActClass){
						++wright;
					}
				}
			}
		}else{
			for(int i = 0; i < container.amountOfPoints; ++i){
				if(!elementsUsedForValidation[i]){ // is not part of the validation trainings part
					++amountOfUsedValues;
					double prob = usedGp.predict(container.data[i], 500);
					if(container.labels[i] == iActClass)
						++amountOfCorrectLabels;
					if(prob > 0.75 && container.labels[i] == iActClass){
						++rr;
						++wright; // stronger weight on correct values, than on wrong values, the goal is to identify good values
					}else if(prob < 0.2 && container.labels[i] != iActClass){
						++wright;
					}
				}
			}
		}
		bestHyperParams->trySet(wright, rr, amountOfUsedValues, amountOfCorrectLabels, result[0], result[1]);
		bestHyperParams->getNoChangeCounter(noChange);
		m_output.printSwitchingColor("Act is: " + number2String(result[0]) + ", " + number2String(result[1])
						+ " with: " + number2String(wright / (double) container.amountOfPoints * 100.0)
						+ " %, just right: " + number2String(rr / (double) amountOfCorrectLabels * 100.0)+ " %, best is at the moment: "
						+ bestHyperParams->prettyStringOfBest() + ", use "
						+ number2String(min(100, noChange * amountOfHyperPoints))
						+ " HPs" + betweenNames + " time in trainF was: " + usedGp.getTrainFWatch().elapsedAvgAsPrettyTime());

		if(bestHyperParams->checkGoal()){
			break;
		}
		bestHyperParams->getFinishLast(isFinished); // last -> stops even if the best result is not reached (after max nr of no changes)
	}
	m_threadCounter.removeThread();
}

int GaussianProcessMultiBinary::predict(const DataElement& point, std::vector<double>& prob) const {
	prob.resize(m_amountOfUsedClasses);
	double p = 0;
	for(int i = 0; i < m_amountOfUsedClasses; ++i){
		if(m_gps[i] != NULL){
			prob[i] = m_gps[i]->predict(point, 500);
		}else{
			prob[i] = 0.0; // there were not enough elements to identify a prob for this class!
		}
		if(fabs(prob[i] - 0.5) < 0.4999){ // 0.3 - 0.7 // do precise measure!
			prob[i] = m_gps[i]->predict(point, 5000);
		}
		p += prob[i];
	}
	if(fabs(p) <= 1e-14){
		p = 0;
		for(int i = 0; i < m_amountOfUsedClasses; ++i){
			if(m_gps[i] != NULL){
				prob[i] = m_gps[i]->predict(point, 3000);
			}else{
				prob[i] = 0.0; // there were not enough elements to identify a prob for this class!
			}
			if(fabs(prob[i] - 0.5) < 0.4999){ // 0.3 - 0.7 // do precise measure!
				prob[i] = m_gps[i]->predict(point, 6000);
			}
			p += prob[i];
		}
		if(fabs(p) <= 1e-14){
			return -1; // no result!
		}
	}
	return std::distance(prob.cbegin(), std::max_element(prob.cbegin(), prob.cend()));
}

GaussianProcessMultiBinary::~GaussianProcessMultiBinary() {
	// TODO Auto-generated destructor stub
}

