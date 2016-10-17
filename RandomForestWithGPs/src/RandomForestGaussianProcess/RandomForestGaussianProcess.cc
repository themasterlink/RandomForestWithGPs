/*
 * RandomForestGaussianProcess.cc
 *
 *  Created on: 29.06.2016
 *      Author: Max
 */

#include "../RandomForestGaussianProcess/RandomForestGaussianProcess.h"
#include "../GaussianProcess/BayesOptimizer.h"
#include "../Data/DataWriterForVisu.h"
#include "../RandomForests/RandomForestWriter.h"
#include "../Data/DataConverter.h"
#include "../Utility/Settings.h"
#include "boost/filesystem.hpp"

RandomForestGaussianProcess::RandomForestGaussianProcess(const DataSets& data, const int heightOfTrees,
		const int amountOfTrees, const std::string& folderPath) :
	m_data(data), m_amountOfUsedClasses(data.size()),
	m_amountOfDataPoints(0),
	m_forest(heightOfTrees, amountOfTrees, m_amountOfUsedClasses),
	m_pureClassLabelForRfClass(m_amountOfUsedClasses, GP_USED),
	m_gps(m_amountOfUsedClasses, std::vector<GaussianProcess*>(m_amountOfUsedClasses, NULL)),
	m_classNames(m_amountOfUsedClasses, ""),
	m_maxPointsUsedInGpSingleTraining(1000),
	m_folderPath(folderPath),
	m_didLoadTree(false),
	m_nrOfRunningThreads(0){
	if(m_data.size() == 0){
		printError("No data given!");
		return;
	}
	std::cout << "Amount of classes: " << data.size() << std::endl;
	if(m_folderPath.length() > 0){
		// check if something was already saved in this folder!
		boost::filesystem::path targetDir(folderPath);
		boost::filesystem::directory_iterator end_itr;
		// cycle through the directory
		for(boost::filesystem::directory_iterator itr(targetDir); itr != end_itr; ++itr){
			if(boost::filesystem::is_regular_file(itr->path())){
				const std::string file(itr->path().c_str());
				if(file.length() > 3 && file.substr(file.length() - 3, 3) == "brf"){
					// remove the empty trees otherwise, he will add the loaded trees to the empty ones
					m_forest.init(0);
					RandomForestWriter::readFromFile(file, m_forest);
					m_didLoadTree = true;
					std::cout << "Read Random Forest from file: " << m_folderPath + "randomForests.brf" << std::endl;
					break;
				}
			}
		}
	}
}

void RandomForestGaussianProcess::train(){
	const int dim = m_data.begin()->second[0]->rows();
	ClassData data;
	DataConverter::setToData(m_data, data);
	/*// count total data points in dataset
	for(DataSets::const_iterator it = m_data.begin(); it != m_data.end(); ++it){
		m_amountOfDataPoints += it->second.size();
	}
		// copy all points in one Data field for training of the RF
	Data rfData(m_amountOfDataPoints);
	Labels labels(m_amountOfDataPoints);
	int labelsCounter = 0, offset = 0;
	for(DataSets::const_iterator it = m_data.begin(); it != m_data.end(); ++it){
		m_classNames[labelsCounter] = it->first;
		for(int i = 0; i < it->second.size(); ++i){
			labels[offset + i] = labelsCounter;
			rfData[offset + i] = it->second[i];
		}
		offset += it->second.size();
		++labelsCounter;
	}*/
	if(!m_didLoadTree){
		// calc min used data for training of random forest bool useFixedValuesForMinMaxUsedData;
		bool useFixedValuesForMinMaxUsedData;
		Settings::getValue("MinMaxUsedData.useFixedValuesForMinMaxUsedData", useFixedValuesForMinMaxUsedData);
		Eigen::Vector2i minMaxUsedData;
		if(useFixedValuesForMinMaxUsedData){
			int minVal = 0, maxVal = 0;
			Settings::getValue("MinMaxUsedData.minValue", minVal);
			Settings::getValue("MinMaxUsedData.maxValue", maxVal);
			minMaxUsedData << minVal, maxVal;
		}else{
			double minVal = 0, maxVal = 0;
			Settings::getValue("MinMaxUsedData.minValueFraction", minVal);
			Settings::getValue("MinMaxUsedData.maxValueFraction", maxVal);
			minMaxUsedData << (int) (minVal * data.size()),  (int) (maxVal * data.size());
		}
		std::cout << "Min max used data, min: " << minMaxUsedData[0] << " max: " << minMaxUsedData[1] << "\n";
		// train the random forest
		m_forest.train(data, dim, minMaxUsedData);
		// save it!
		RandomForestWriter::writeToFile(m_folderPath + "randomForests.brf", m_forest);
		std::cout << "Write Random Forest to file: " << m_folderPath + "randomForests.brf" << std::endl;
	}
	std::cout << "Total amount of data points for training: " << m_amountOfDataPoints << std::endl;
	// get the pre classes for each data point
	Labels guessedLabels; // contains for each data point the rf result classes
	m_forest.predictData(data, guessedLabels);
	// count the occurence of each pre class of the random forest
	std::vector<int> countClasses(m_amountOfUsedClasses, 0);
	for(int i = 0; i < guessedLabels.size(); ++i){
		countClasses[guessedLabels[i]] += 1;
	}
	// sort the data based on the pre classes of the rf
	std::vector<ClassData> sortedData;
	sortedData.resize(m_amountOfUsedClasses);
	for(int i = 0; i < m_amountOfUsedClasses; ++i){
		sortedData[i].resize(countClasses[i]);
	}
	// copy the data in the right pre classes
	std::vector<int> counter(m_amountOfUsedClasses,0);
	for(int i = 0; i < m_amountOfDataPoints;  ++i){
		const int label = guessedLabels[i];
		sortedData[label][counter[label]] = data[i];
		counter[label] += 1;
	}
	/*	for(int i = 0; i < m_amountOfUsedClasses; ++i){
		std::cout << "Data for " << i << ":" << std::endl;
		for(int j = 0; j < sortedData[i].size(); ++j){
			std::cout << sortedData[i][j].transpose() << ", ";
		}
		std::cout << "\nLabels for " << i << ":" << std::endl;
		for(int j = 0; j < sortedLabels[i].size(); ++j){
			std::cout << sortedLabels[i][j] << ", ";
		}
		std::cout << std::endl;
	}*/
	int thresholdForNoise = 5;
	Settings::getValue("RFGP.thresholdForNoise", thresholdForNoise);
	int pointsPerClassForBayOpt = 16;
	Settings::getValue("RFGP.pointsPerClassForBayOpt", pointsPerClassForBayOpt);
	m_maxPointsUsedInGpSingleTraining = 1500;
	Settings::getValue("RFGP.maxPointsUsedInGpSingleTraining", m_maxPointsUsedInGpSingleTraining);
	int maxNrOfPointsForBayesOpt = 250;
	Settings::getValue("RFGP.maxNrOfPointsForBayesOpt", maxNrOfPointsForBayesOpt);
	boost::thread_group group;
	for(int iActRfRes = 0; iActRfRes < m_amountOfUsedClasses; ++iActRfRes){ // go over all classes
		//m_output.printSwitchingColor("Act Class: " + m_classNames[iActRfRes]);
		const ClassData& dataOfActRf = sortedData[iActRfRes];
		const int amountOfDataInRfRes = dataOfActRf.size();

		//std::cout << "Amount of data: " << amountOfDataInRfRes << std::endl;
		// count the amount of class labels per pre class
		std::vector<int> classCounts(m_amountOfUsedClasses, 0);
		for(int counterLabels = 0; counterLabels < amountOfDataInRfRes; ++counterLabels){
			classCounts[dataOfActRf[counterLabels]->getLabel()] += 1;
		}
		// if there is enough data for gp
		int amountOfClassesOverThreshold = 0;
		int idOfMaxClass = -1;
		int amountOfMaxClass = -1;
		// count the classes over threshold and find the class with the most members
		for(int i = 0; i < m_amountOfUsedClasses; ++i){
			if(classCounts[i] > thresholdForNoise){
				++amountOfClassesOverThreshold;
			}
			if(classCounts[i] > amountOfMaxClass){
				amountOfMaxClass = classCounts[i];
				idOfMaxClass = i;
			}
		}
		m_output.printSwitchingColor("Class: " + m_classNames[iActRfRes] + ", has " + number2String(amountOfDataInRfRes) + " points, pure level is: " + number2String(m_pureClassLabelForRfClass[iActRfRes]));

		if(amountOfDataInRfRes > thresholdForNoise * 2){
			if(amountOfClassesOverThreshold <= 1){ // only one class or no class
				if(idOfMaxClass == -1){
					// use the result of the rf -> but bad sign that there is no trainings element in this rf class
					idOfMaxClass = iActRfRes;
				}
				m_pureClassLabelForRfClass[iActRfRes] = idOfMaxClass;
				continue; // no gps needed! for this class
			}
			m_output.printSwitchingColor("Class: " + m_classNames[iActRfRes] + ", best for it is: " + m_classNames[idOfMaxClass]);
			/*
			Eigen::MatrixXd dataMat; // contains all the data for this specified pre class result of the RF
			dataMat.conservativeResize(sortedData[iActRfRes][0].rows(), amountOfDataInRfRes);
			int i = 0;
			for(ClassDataIterator it = sortedData[iActRfRes].begin(); it != sortedData[iActRfRes].end(); ++it){
				dataMat.col(i++) = *it;
			}
			*/
			// resize gps for all other classes

			for(int iActClass = 0; iActClass < m_amountOfUsedClasses; ++iActClass){
				if(classCounts[iActClass] <= thresholdForNoise && iActClass != iActRfRes){ // check if class is there, otherwise go to next!
					continue;
				}
				m_output.printSwitchingColor("In Class: " + m_classNames[iActRfRes] + ", has act class: " + m_classNames[iActClass] + " so many points: " + number2String(classCounts[iActClass]));
				//m_isGpInUse[iActRfRes][iActClass] = true; // there is actually a gp for this config
				m_gps[iActRfRes][iActClass] = new GaussianProcess();
				const int nrOfParallel = boost::thread::hardware_concurrency();
				while(m_nrOfRunningThreads >= nrOfParallel){
					usleep(0.35 * 1e6);
				}
				group.add_thread(new boost::thread(boost::bind(&RandomForestGaussianProcess::trainInParallel, this, iActClass, amountOfDataInRfRes,
						min(maxNrOfPointsForBayesOpt, pointsPerClassForBayOpt * amountOfClassesOverThreshold),
						iActRfRes, dataOfActRf, classCounts, m_gps[iActRfRes][iActClass])));
				/*trainInParallel(iActClass, amountOfDataInRfRes,
						pointsPerClassForBayOpt, amountOfClassesOverThreshold,
						maxPointsUsedInGpSingleTraining, dataOfActRf,
						labelsOfActRf, classCounts, actGp);*/
			}
		}else{
			// not enough data for gp
			m_pureClassLabelForRfClass[iActRfRes] = idOfMaxClass; // pure class -> save id
		}
	}
	group.join_all();
	int c = 0;
	for(int i = 0; i < m_amountOfUsedClasses; ++i){
		for(int j = 0; j < m_amountOfUsedClasses; ++j){
			c += m_gps[i][j] != NULL? 1 : 0;
		}
	}
	std::cout << "Amount of gps: " << c << std::endl;
}

void RandomForestGaussianProcess::trainInParallel(const int iActClass,
		const int amountOfDataInRfRes, const int amountOfHyperPoints,
		const int iActRfClass, const ClassData& dataOfActRf, const std::vector<int>& classCounts,
		GaussianProcess* actGp) {
	++m_nrOfRunningThreads;
	int nrOfNoChanges;
	Settings::getValue("RFGP.nrOfNoChanges", nrOfNoChanges);
	Eigen::MatrixXd dataMat;
	Eigen::VectorXd yGpInit;
	// calc for final training
	DataConverter::toRandUniformDataMatrix(dataOfActRf, classCounts, dataMat, yGpInit, m_maxPointsUsedInGpSingleTraining, iActClass); // get a uniform portion of at most 1000 points

	Eigen::MatrixXd testDataMat;
	Eigen::VectorXd testYGpInit;
	// calc for final training
	DataConverter::toRandUniformDataMatrix(dataOfActRf, classCounts, testDataMat, testYGpInit, m_maxPointsUsedInGpSingleTraining / 5, iActClass); // get a uniform portion of at most 1000 points

	// compare to all other classes! // one vs. all
	std::string betweenNames = ", for " + m_classNames[iActClass] + " in " + m_classNames[iActRfClass] + ", which has " + number2String(amountOfDataInRfRes);
	m_output.printSwitchingColor("Start parallel" + betweenNames);
	Eigen::VectorXd y(amountOfDataInRfRes);
	const int size = min(300, (int)(dataOfActRf.size()));
	int bestRight = -1;
	double len = 10, sigmaF = 0.4;
	StopWatch sw;
	int noChange = 1;
	for(int i = 0; i < 20; ++i){
		m_output.printSwitchingColor("Is in: " + number2String(i) + ", best is at the moment: " + number2String(len) + ", " + number2String(sigmaF) + ", with: " + number2String(bestRight / (double)size * 100.0) + betweenNames);
		Eigen::MatrixXd dataHyper;
		Eigen::VectorXd yHyper;
		DataConverter::toRandUniformDataMatrix(dataOfActRf, classCounts, dataHyper, yHyper, noChange * amountOfHyperPoints, iActClass); // get a uniform portion of all points
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
		vectord result(2);
		actGp->init(dataHyper, yHyper);
		bayesopt::Parameters par = initialize_parameters_to_default();
		par.noise = 1e-12;
		par.epsilon = 0.2;
		par.verbose_level = 6;
		par.n_iterations = 200;
		par.surr_name = "sGaussianProcessML";
		vectord lowerBound(2); // for hyper params in bayesian optimization
		lowerBound[0] = 0.1;
		lowerBound[1] = 0.05;
		vectord upperBound(2);
		upperBound[0] = 100; //;actGp->getKernel().getLenVar() / 3;
		upperBound[1] = 1.5;
		BayesOptimizer bayOpt(*actGp, par);
		bayOpt.setBoundingBox(lowerBound, upperBound);
		bool hasError = false;
		do{
			try{
				bayOpt.optimize(result);
			}catch(std::runtime_error& e){
				m_output.printSwitchingColor(e.what());
				hasError = true;
				getchar();
			}
		}while(hasError);
		actGp->getKernel().setHyperParams(result[0], result[1], actGp->getKernel().sigmaN());

		actGp->init(testDataMat,testYGpInit);
		actGp->trainWithoutKernelOptimize();
		int right = 0;
		for(int i = 0; i < size; ++i){
			const int nextEle = rand() / (double) RAND_MAX * dataOfActRf.size();
			ClassPoint& ele = *dataOfActRf[nextEle];
			double prob = actGp->predict(ele, 500);
			if(prob > 0.5 && ele.getLabel() == iActClass){
				++right;
			}else if(prob < 0.5 && ele.getLabel() != iActClass){
				++right;
			}
		}
		if(right > bestRight){
			bestRight = right;
			len = result[0];
			noChange = 1;
			sigmaF = result[1];
		}else{
			++noChange;
		}
		if(noChange > nrOfNoChanges){
			break;
		}
		if(bestRight / (double)size * 100.0 > 95.0){
			break;
		}
	}
	// set hyper params
	m_output.printSwitchingColor("Finish optimizing with " + number2String(len) + ", " + number2String(sigmaF) + " in: " + sw.elapsedAsPrettyTime() + ", with: " + number2String(bestRight / (double)size * 100.0) + betweenNames);
	sw.startTime();
	// train on whole data set
	actGp->getKernel().setHyperParams(len, sigmaF, actGp->getKernel().sigmaN());
	actGp->init(dataMat,yGpInit);
	m_output.printSwitchingColor("Finish init in: " + sw.elapsedAsPrettyTime() + betweenNames );
	sw.startTime();
	actGp->trainWithoutKernelOptimize();
	m_output.printSwitchingColor("Finish training in: " + sw.elapsedAsPrettyTime() + betweenNames);
	--m_nrOfRunningThreads;
}

int RandomForestGaussianProcess::predict(const DataPoint& point, std::vector<double>& prob) const {
	const int rfLabel = m_forest.predict(point);
	return rfLabel;
	if(m_pureClassLabelForRfClass[rfLabel] != GP_USED){ // is pure
		prob = std::vector<double>(m_amountOfUsedClasses, 0.0); // set all probs to zero
		prob[m_pureClassLabelForRfClass[rfLabel]] = 1.0; // set the
		return m_pureClassLabelForRfClass[rfLabel];
	}
//	std::cout << "Use gp: " << std::endl;
	prob.resize(m_amountOfUsedClasses);
	for(int i = 0; i < m_amountOfUsedClasses; ++i){
		if(m_gps[rfLabel][i] != NULL){
			prob[i] = m_gps[rfLabel][i]->predict(point);
		}else{
			prob[i] = 0.0; // there were not enough elements to identify a prob for this class!
		}
	}
	double p = 0;
	for(int i = 0; i < m_amountOfUsedClasses; ++i){
		p += prob[i];
	}
	if(fabs(p) <= 1e-7){
		//std::cout << "p is zero, p: " << p << std::endl;
		return m_pureClassLabelForRfClass[rfLabel];
	}
	int sum = 0;
	for(int i = 0; i < m_amountOfUsedClasses; ++i){
		if(fabs(prob[i] - 1.0) <= 1e-7){
			++sum;
		}
	}
	if(sum > 1){
		//std::cout << "fuck yu bitch!" << std::endl;
		return m_pureClassLabelForRfClass[rfLabel];
	}
	return std::distance(prob.cbegin(), std::max_element(prob.cbegin(), prob.cend()));
}

RandomForestGaussianProcess::~RandomForestGaussianProcess(){
}

