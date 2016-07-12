/*
 * RandomForestGaussianProcess.cc
 *
 *  Created on: 29.06.2016
 *      Author: Max
 */

#include "../RandomForestGaussianProcess/RandomForestGaussianProcess.h"
#include "../GaussianProcess/BayesOptimizer.h"
#include "../Data/DataWriterForVisu.h"
#include "../Data/DataConverter.h"

RandomForestGaussianProcess::RandomForestGaussianProcess(const DataSets& data, const int heightOfTrees,
		const int amountOfTrees) :
	m_data(data), m_heightOfTrees(heightOfTrees),
	m_amountOfTrees(amountOfTrees), m_amountOfUsedClasses(data.size()),
	m_amountOfDataPoints(0),
	m_forest(m_heightOfTrees, m_amountOfTrees, m_amountOfUsedClasses),
	m_pureClassLabelForRfClass(m_amountOfUsedClasses, -1){
	if(m_data.size() == 0){
		printError("No data given!");
		return;
	}
	const int dim = m_data.begin()->second[0].rows();
	// count total data points in dataset
	for(DataSets::const_iterator it = data.begin(); it != data.end(); ++it){
		m_amountOfDataPoints += it->second.size();
	}
	// calc min used data for training of random forest TODO values should be from settings
	Eigen::Vector2i minMaxUsedData;
	minMaxUsedData << m_amountOfDataPoints * 0.2 , m_amountOfDataPoints * 0.6;
	// copy all points in one Data field for training of the RF
	Labels labels(m_amountOfDataPoints);
	Data rfData(m_amountOfDataPoints);
	std::vector<std::string> namesToNumbers(data.size()); // save name for a class id
	int labelsCounter = 0, offset = 0;
	for(DataSets::const_iterator it = data.begin(); it != data.end(); ++it){
		namesToNumbers[labelsCounter] = it->first;
		for(int i = 0; i < it->second.size(); ++i){
			labels[offset + i] = labelsCounter;
			rfData[offset + i] = it->second[i];
		}
		offset += it->second.size();
		++labelsCounter;
	}
	// train the random forest
	m_forest.train(rfData, labels, dim, minMaxUsedData);

	// get the pre classes for each data point
	std::vector<int> guessedLabels; // contains for each data point the rf result classes
	m_forest.predictData(rfData, guessedLabels);
	// count the occurence of each pre class of the random forest
	std::vector<int> countClasses(m_amountOfUsedClasses, 0);
	for(int i = 0; i < guessedLabels.size(); ++i){
		countClasses[guessedLabels[i]] += 1;
	}
	// sort the data based on the pre classes of the rf
	std::vector<Data > sortedData;
	std::vector<Labels > sortedLabels;
	sortedData.resize(m_amountOfUsedClasses);
	sortedLabels.resize(m_amountOfUsedClasses);
	for(int i = 0; i < m_amountOfUsedClasses; ++i){
		sortedData[i].resize(countClasses[i]);
		sortedLabels[i].resize(countClasses[i]);
	}
	// copy the data in the right pre classes
	std::vector<int> counter(m_amountOfUsedClasses,0);
	for(int i = 0; i < m_amountOfDataPoints;  ++i){
		const int label = guessedLabels[i];
		sortedData[label][counter[label]] = rfData[i];
		sortedLabels[label][counter[label]] = labels[i];
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
	const int thresholdForNoise = 15; // TODO get out of settings
	m_gps.resize(m_amountOfUsedClasses);
	for(int iActRfRes = 0; iActRfRes < m_amountOfUsedClasses; ++iActRfRes){ // go over all classes
		const Data& dataOfActRf = sortedData[iActRfRes];
		const Labels& labelsOfActRf = sortedLabels[iActRfRes];
		const int amountOfDataInRfRes = dataOfActRf.size();
		std::cout << "Amount of data: " << amountOfDataInRfRes << std::endl;
		// count the amount of class labels per pre class
		std::vector<int> classCounts(m_amountOfUsedClasses, 0);
		for(int counterLabels = 0; counterLabels < amountOfDataInRfRes; ++counterLabels){
			classCounts[labelsOfActRf[counterLabels]] += 1;
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
		if(amountOfDataInRfRes > thresholdForNoise * 2){
			if(amountOfClassesOverThreshold <= 1){ // only one class or no class
				m_pureClassLabelForRfClass[iActRfRes] = idOfMaxClass;
				continue; // no gps needed! for this class
			}
			/*
			Eigen::MatrixXd dataMat; // contains all the data for this specified pre class result of the RF
			dataMat.conservativeResize(sortedData[iActRfRes][0].rows(), amountOfDataInRfRes);
			int i = 0;
			for(Data::iterator it = sortedData[iActRfRes].begin(); it != sortedData[iActRfRes].end(); ++it){
				dataMat.col(i++) = *it;
			}
			*/
			// resize gps for all other classes
			m_gps[iActRfRes].resize(m_amountOfUsedClasses);
			for(int iActClass = 0; iActClass < m_amountOfUsedClasses; ++iActClass){
				// walk over all classes! // one vs. all
				std::cout << "Class: " << iActClass << std::endl;
				Eigen::VectorXd y(amountOfDataInRfRes);
				bool isThere = false;
				const int hyperPoints = 36;
				Eigen::MatrixXd dataHyper;
				dataHyper.conservativeResize(dim, hyperPoints);
				Eigen::VectorXd yHyper(hyperPoints);
				int oneCounter = 0, minusCounter = 0, counterHyper = 0;
				// copy the first 35 points of both TODO find way of randomly taking the values
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
				if(isThere && minusCounter > 2){ // an element of this class is here, so a gp must be performed!
					// find good hyperparameters with bayesian optimization:

					GaussianProcessBinary& actGp = m_gps[iActRfRes][iActClass];

					actGp.init(dataHyper, yHyper);
					bayesopt::Parameters par = initialize_parameters_to_default();
					par.noise = 1e-12;
					par.epsilon = 0.2;
					par.surr_name = "sGaussianProcessML";
					BayesOptimizer bayOpt(actGp, par);
					vectord result(2);
					vectord lowerBound(2);
					lowerBound[0] = 0.1;
					lowerBound[1] = 0.1;
					vectord upperBound(2);
					upperBound[0] = actGp.getKernel().getLenVar() / 3;
					upperBound[1] = 1.3;
					bayOpt.setBoundingBox(lowerBound, upperBound);
					bayOpt.optimize(result);

					// set hyper params
					actGp.getKernel().setHyperParams(result[0], result[1], actGp.getKernel().sigmaN());

					// train on whole data set
					Eigen::MatrixXd dataMat;
					DataConverter::toDataMatrix(dataOfActRf, dataMat, amountOfDataInRfRes);
					actGp.init(dataMat,y);
					actGp.trainWithoutKernelOptimize();
				}
			}
		}else{
			// not enough data for gp
			m_pureClassLabelForRfClass[iActRfRes] = idOfMaxClass; // pure class -> save id
		}
	}
}

int RandomForestGaussianProcess::predict(const DataElement& point, std::vector<double>& prob) const {
	const int rfLabel = m_forest.predict(point);
	if(m_pureClassLabelForRfClass[rfLabel] != -1){ // is pure
		prob = std::vector<double>(m_amountOfUsedClasses, 0.0); // set all probs to zero
		prob[m_pureClassLabelForRfClass[rfLabel]] = 1.0; // set the
		return rfLabel;
	}
	prob.resize(m_amountOfUsedClasses);
	for(int i = 0; i < m_amountOfUsedClasses; ++i){
		prob[i] = m_gps[rfLabel][i].predict(point);
	}
	return std::distance(prob.cbegin(), std::max_element(prob.cbegin(), prob.cend()));
}

RandomForestGaussianProcess::~RandomForestGaussianProcess()
{
	// TODO Auto-generated destructor stub
}

