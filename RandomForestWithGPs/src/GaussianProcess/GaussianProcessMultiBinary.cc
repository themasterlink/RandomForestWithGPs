/*
 * GaussianProcessMultiBinary.cc
 *
 *  Created on: 21.07.2016
 *      Author: Max
 */

#ifdef BUILD_OLD_CODE

#include "GaussianProcessMultiBinary.h"

#include "../Data/DataConverter.h"

GaussianProcessMultiBinary::GaussianProcessMultiBinary(int amountOfUsedClasses):
	m_amountOfUsedClasses(amountOfUsedClasses),
	m_amountOfDataPoints(0),
	m_amountOfDataPointsForUseAllTestsPoints(300),
	m_maxPointsUsedInGpSingleTraining(1000),
//	m_lowerBound(2),
//	m_upperBound(2),
	m_gps(amountOfUsedClasses, NULL){
//	Settings::instance().getValue("MultiBinaryGP.lowerBoundLength", m_lowerBound[0]);
//	Settings::instance().getValue("MultiBinaryGP.lowerBoundNoise",  m_lowerBound[1]);
//	Settings::instance().getValue("MultiBinaryGP.upperBoundLength", m_upperBound[0]);
//	Settings::instance().getValue("MultiBinaryGP.upperBoundNoise",  m_upperBound[1]);
}

void GaussianProcessMultiBinary::train(const LabeledData& data, const Labels* guessedLabels){
	UNUSED(guessedLabels);
	m_amountOfDataPoints = data.size();
	// count the occurence of each pre class of the random forest
	std::vector<int> countClasses(m_amountOfUsedClasses, 0);
	for(LabeledDataConstIterator it = data.begin(); it != data.end(); ++it){
		countClasses[(*it)->getLabel()] += 1;
	}
	m_amountOfDataPointsForUseAllTestsPoints = 300;
	int thresholdForNoise = 5;
	Settings::instance().getValue("RFGP.thresholdForNoise", thresholdForNoise);
	int pointsPerClassForBayOpt = 16;
	Settings::instance().getValue("RFGP.pointsPerClassForBayOpt", pointsPerClassForBayOpt);
	m_maxPointsUsedInGpSingleTraining = 1500;
	Settings::instance().getValue("RFGP.maxPointsUsedInGpSingleTraining", m_maxPointsUsedInGpSingleTraining);
	int maxNrOfPointsForBayesOpt = 250;
	Settings::instance().getValue("RFGP.maxNrOfPointsForBayesOpt", maxNrOfPointsForBayesOpt);
	ThreadGroup group;
	for(int iActClass = 0; iActClass < m_amountOfUsedClasses; ++iActClass){
		if(countClasses[iActClass] < thresholdForNoise){
			m_output.printSwitchingColor(
					ClassKnowledge::instance().getNameFor(iActClass) + " is not used, because count class is: " +
					StringHelper::number2String(countClasses[iActClass]) + "!");
			continue; // do not use this class
		}
		m_output.printSwitchingColor(
				"In Class: " + ClassKnowledge::instance().getNameFor(iActClass) + " has so many points: " +
				StringHelper::number2String(countClasses[iActClass]));
		//m_isGpInUse[iActRfRes][iActClass] = true; // there is actually a gp for this config

		m_gps[iActClass] = new GaussianProcess();

		while(!m_threadCounter.addNewThread()){ // if it is not possible wait
			sleepFor(0.35);
		}
		group.addThread(makeThread(&GaussianProcessMultiBinary::trainInParallel, this, iActClass,
				std::min(maxNrOfPointsForBayesOpt, pointsPerClassForBayOpt * (int) m_amountOfUsedClasses),
				data, countClasses, m_gps[iActClass]));
		/*trainInParallel(iActClass, amountOfDataInRfRes,
							pointsPerClassForBayOpt, amountOfClassesOverThreshold,
							maxPointsUsedInGpSingleTraining, dataOfActRf,
							labelsOfActRf, classCounts, actGp);*/
	}
	group.joinAll();
	int c = 0;
	for(int i = 0; i < m_amountOfUsedClasses; ++i){
		c += m_gps[i] != NULL? 1 : 0;
	}
	std::cout << "Amount of gps: " << c << std::endl;
}

void GaussianProcessMultiBinary::trainInParallel(const int iActClass,
		const int amountOfHyperPoints, const LabeledData& data,
		const std::vector<int>& classCounts, GaussianProcess* actGp) {
	int nrOfNoChanges;
	Settings::instance().getValue("RFGP.nrOfNoChanges", nrOfNoChanges);
	Matrix dataMat;
	VectorX yGpInit;
	// calc for final training
	std::vector<bool> usedElements;
	std::vector<bool> blockElements(data.size(), false); // block none
	DataConverter::toRandClassAndHalfUniformDataMatrix(data, classCounts,
			dataMat, yGpInit, m_maxPointsUsedInGpSingleTraining, iActClass, usedElements, blockElements); // get a uniform portion of at most 1000 points

	Matrix testDataMat;
	VectorX testYGpInit;
	const int amountOfPointsUsedForTrainingTheTest = 300; // min(max(200,container.amountOfPoints / 3),m_maxPointsUsedInGpSingleTraining / 3); TODO
	// calc for final training
	std::vector<bool> usedElementsForTheValidationSet; // save the blocked values, for the testing of the hyper params
	std::vector<bool> blockElementsForValidationSet(data.size(), false); // block none
	DataConverter::toRandClassAndHalfUniformDataMatrix(data, classCounts, testDataMat, testYGpInit,
			amountOfPointsUsedForTrainingTheTest, iActClass, usedElementsForTheValidationSet, blockElementsForValidationSet); // get a uniform portion of at most 1000 points

	// compare to all other classes! // one vs. all
	const int numberOfPointsForClass = classCounts[iActClass];
	const std::string betweenNames = ", for " + ClassKnowledge::instance().getNameFor(iActClass) + " has " +
									 StringHelper::number2String(numberOfPointsForClass);
	m_output.printSwitchingColor("Start parallel with " + StringHelper::number2String(amountOfPointsUsedForTrainingTheTest) + " amount of points, which are used in the training for the testing" + betweenNames);
	//VectorX y(numberOfPointsForClass);
	StopWatch sw;
	BestHyperParams bestHyperParams(20);
	ThreadGroup group;
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
			group.addThread(makeThread(&GaussianProcessMultiBinary::optimizeHyperParams, this,
					iActClass, amountOfHyperPoints, data, classCounts, usedElementsForTheValidationSet, testDataMat, testYGpInit, &bestHyperParams));
			m_output.printInColor("At the moment are " + StringHelper::number2String(m_threadCounter.currentThreadCount()) + " threads running" + betweenNames, RESET);
		}
		bestHyperParams.getFinishLast(isFinish);
		if(!isFinish){
			sleepFor(0.35);
		}
		++iCounter;
	}
	group.joinAll();
	m_output.printInColor("Finish optimizing with in: " + sw.elapsedAsPrettyTime()
					+ ", with: " + bestHyperParams.prettyStringOfBest()
					+ betweenNames, MAGENTA);
	sw.startTime();
	// set hyper params
	Real len, sigmaF;
	bestHyperParams.getBestHypParams(len,sigmaF);
	actGp->getKernel().setHyperParams(len, sigmaF);
	// train on whole data set
	actGp->init(dataMat,yGpInit);
	m_output.printInColor("Finish init for " + StringHelper::number2String(yGpInit.rows()) + " elements in: " + sw.elapsedAsPrettyTime() + betweenNames, MAGENTA);
	sw.startTime();
	actGp->trainWithoutKernelOptimize();
	m_output.printInColor("Finish training for " + StringHelper::number2String(yGpInit.rows()) + " elements in: " + sw.elapsedAsPrettyTime() + betweenNames, MAGENTA);
	m_threadCounter.removeThread();
}

void GaussianProcessMultiBinary::optimizeHyperParams(const unsigned int iActClass,
		const int amountOfHyperPoints, const LabeledData& data,
		const std::vector<int>& classCounts, const std::vector<bool>& elementsUsedForValidation,
		const Matrix& testDataMat, const VectorX& testYGpInit, BestHyperParams* bestHyperParams){
	GaussianProcess usedGp;
	const int numberOfPointsForClass = classCounts[iActClass];
	const std::string betweenNames = ", for " + ClassKnowledge::instance().getNameFor(iActClass) + " has " +
									 StringHelper::number2String(numberOfPointsForClass);
	int noChange = 1;
	const bool useAllTestValues = true; //m_amountOfDataPoints <= m_amountOfDataPointsForUseAllTestsPoints; TODO
	int size = std::min(m_amountOfDataPointsForUseAllTestsPoints, m_amountOfDataPoints);
	bool isFinished = false;
	bestHyperParams->getNoChangeCounter(noChange);
	while(!isFinished){
		Matrix dataHyper;
		VectorX yHyper;
		std::vector<bool> usedElements;
		DataConverter::toRandClassAndHalfUniformDataMatrix(data, classCounts, dataHyper,
				yHyper, std::min(100, noChange * amountOfHyperPoints), iActClass, usedElements, elementsUsedForValidation); // get a uniform portion of all points
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
	std::cout << (Real) yHyper[i] << std::endl;
}
std::cout << "One: " << oneCounter << std::endl;
		 */
		// find good hyperparameters with bayesian optimization:
		bestHyperParams->getFinishDuring(isFinished); // only if one result is above the testing mark
		if(isFinished){ break; }
		std::cout << "Bayesopt was removed here, if this code is used again, add lib bayesopt again!" << std::endl;
//		vectord result(2);
//		usedGp.init(dataHyper, yHyper); // TODO reinit of kernel matrix -> is not necessary
//		bayesopt::Parameters par = initialize_parameters_to_default();
//		par.noise = 1e-12;
//		par.epsilon = 0.2;
//		par.verbose_level = 6;
//		par.n_iterations = 200;
//		par.surr_name = "sGaussianProcessML";
//		BayesOptimizer bayOpt(usedGp, par);
//		bayOpt.setBoundingBox(m_lowerBound, m_upperBound);
//		try{
//			bestHyperParams->getFinishDuring(isFinished); // only if one result is above the testing mark
//			if(isFinished){ break; }
//			bayOpt.optimize(result);
//		}catch(std::runtime_error& e){
//			//upperBound[1] = 1.5; // reduce noice!
//			m_output.printSwitchingColor(e.what() + betweenNames);
//			continue;
//		}
//		usedGp.getKernel().setHyperParams(result[0], result[1]);
		bestHyperParams->getFinishDuring(isFinished); // only if one result is above the testing mark
		if(isFinished){ break; }
		usedGp.init(testDataMat, testYGpInit);
		bestHyperParams->getFinishDuring(isFinished); // only if one result is above the testing mark
		if(isFinished){ break; }
		usedGp.trainWithoutKernelOptimize();
		bestHyperParams->getFinishDuring(isFinished); // only if one result is above the testing mark
		if(isFinished){ break; }
		int right = 0;
		int rr = 0;
		int amountOfCorrectLabels = 0;
		int amountOfUsedValues = 0;
		if(!useAllTestValues){
			for(int i = 0; i < size; ++i){
				const int nextEle = rand() / (Real) RAND_MAX * m_amountOfDataPoints;
				if(!elementsUsedForValidation[nextEle]){ // is not part of the validation trainings part
					LabeledVectorX& ele = *data[nextEle];
					Real prob = usedGp.predict(ele, 500);
					if(prob > 0.5 && ele.getLabel() == iActClass){
						++right;
					}else if(prob < 0.5 && ele.getLabel() != iActClass){
						++right;
					}
				}
			}
		}else{
			for(int i = 0; i < (int) data.size(); ++i){
				if(!elementsUsedForValidation[i]){ // is not part of the validation trainings part
					++amountOfUsedValues;
					LabeledVectorX& ele = *data[i];
					Real prob = usedGp.predict(ele, 500);
					if(ele.getLabel() == iActClass)
						++amountOfCorrectLabels;
					if(prob > 0.75 && ele.getLabel() == iActClass){
						++rr;
						++right; // stronger weight on correct values, than on wrong values, the goal is to identify good values
					}else if(prob < 0.2 && ele.getLabel() != iActClass){
						++right;
					}
				}
			}
		}
		std::cout << "Change was made here too!" << std::endl;
//		bestHyperParams->trySet(right, rr, amountOfUsedValues, amountOfCorrectLabels, result[0], result[1]);
		bestHyperParams->getNoChangeCounter(noChange);
		const int percentagePrecision = 2;
//		m_output.printSwitchingColor("Act is: " + StringHelper::number2String(result[0], percentagePrecision) + ", " + StringHelper::number2String(result[1], percentagePrecision)
//						+ " with: " + StringHelper::number2String(right / (Real) amountOfUsedValues * 100.0, percentagePrecision) + " %, for " + StringHelper::number2String(amountOfUsedValues)
//						+ " test elements, just right: " + StringHelper::number2String(rr / (Real) amountOfCorrectLabels * 100.0, percentagePrecision)
//						+ " %, best now: "
//						+ bestHyperParams->prettyStringOfBest(percentagePrecision) + ", use "
//						+ StringHelper::number2String(std::min(100, noChange * amountOfHyperPoints))
//						+ " HPs" + betweenNames + " time in trainF was: " + usedGp.getTrainFWatch().elapsedAvgAsPrettyTime());
		if(bestHyperParams->checkGoal()){
			break;
		}
		bestHyperParams->getFinishLast(isFinished); // last -> stops even if the best result is not reached (after max nr of no changes)
	}
	m_threadCounter.removeThread();
}

unsigned int GaussianProcessMultiBinary::predict(const VectorX& point, std::vector<Real>& prob) const {
	prob.resize(m_amountOfUsedClasses);
	Real p = 0;
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
	if(fabs(p) <= EPSILON){
		p = 0;
		for(int i = 0; i < m_amountOfUsedClasses; ++i){
			if(m_gps[i] != NULL){
				m_gps[i]->resetFastPredict(); // to avoid that the fast prediction is wrong!
				prob[i] = m_gps[i]->predict(point, 3000);
			}else{
				prob[i] = 0.0; // there were not enough elements to identify a prob for this class!
			}
			if(fabs(prob[i] - 0.5) < 0.4999){ // 0.3 - 0.7 // do precise measure!
				prob[i] = m_gps[i]->predict(point, 6000);
			}
			p += prob[i];
		}
		if(fabs(p) <= EPSILON){
			return -1; // no result!
		}
	}
	return argMax(prob);
}

unsigned int GaussianProcessMultiBinary::predict(const VectorX& point) const{
	std::vector<Real> prob;
	return predict(point, prob);
}

unsigned int GaussianProcessMultiBinary::amountOfClasses() const{
	return m_amountOfUsedClasses;
}


GaussianProcessMultiBinary::~GaussianProcessMultiBinary() {
	for(int iActClass = 0; iActClass < m_amountOfUsedClasses; ++iActClass){
		saveDelete(m_gps[iActClass]);
	}
}


#endif // BUILD_OLD_CODE