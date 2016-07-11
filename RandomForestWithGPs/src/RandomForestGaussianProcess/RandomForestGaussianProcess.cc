/*
 * RandomForestGaussianProcess.cc
 *
 *  Created on: 29.06.2016
 *      Author: Max
 */

#include "../RandomForestGaussianProcess/RandomForestGaussianProcess.h"
#include "../GaussianProcess/BayesOptimizer.h"
#include "../Data/DataWriterForVisu.h"

RandomForestGaussianProcess::RandomForestGaussianProcess(const DataSets& data, const int heightOfTrees,
		const int amountOfTrees) :
	m_data(data), m_heightOfTrees(heightOfTrees),
	m_amountOfTrees(amountOfTrees), m_amountOfUsedClasses(data.size()),
	m_amountOfDataPoints(0),
	m_forest(m_heightOfTrees, m_amountOfTrees, m_amountOfUsedClasses),
	m_isPure(m_amountOfUsedClasses, false){

	if(m_data.size() == 0){
		printError("No data given!");
		return;
	}
	for(DataSets::const_iterator it = data.begin(); it != data.end(); ++it){
		m_amountOfDataPoints += it->second.size();
	}
	Eigen::Vector2i minMaxUsedData;
	minMaxUsedData << m_amountOfDataPoints * 0.2 , m_amountOfDataPoints * 0.6;
	Labels labels(m_amountOfDataPoints);
	Data rfData(m_amountOfDataPoints);
	std::vector<std::string> namesToNumbers(data.size());
	int l = 0, offset = 0;
	for(DataSets::const_iterator it = data.begin(); it != data.end(); ++it){
		namesToNumbers[l] = it->first;
		for(int i = 0; i < it->second.size(); ++i){
			labels[offset + i] = l;
			rfData[offset + i] = it->second[i];
		}
		offset += it->second.size();
		++l;
	}
	m_forest.train(rfData, labels, rfData[0].rows(), minMaxUsedData);
	std::vector<int> guessedLabels;
	m_forest.predictData(rfData, guessedLabels);
	std::vector<int> countClasses(m_amountOfUsedClasses, 0);
	for(int i = 0; i < guessedLabels.size(); ++i){
		countClasses[guessedLabels[i]] += 1;
	}

	std::vector<Data > sortedData;
	std::vector<Labels > sortedLabels;
	sortedData.resize(m_amountOfUsedClasses);
	sortedLabels.resize(m_amountOfUsedClasses);
	for(int i = 0; i < m_amountOfUsedClasses; ++i){
		sortedData[i].resize(countClasses[i]);
		sortedLabels[i].resize(countClasses[i]);
	}
	std::vector<int> counter(m_amountOfUsedClasses,0);
	for(int i = 0; i < m_amountOfDataPoints;  ++i){
		const int label = guessedLabels[i];
		sortedData[label][counter[label]] = rfData[i];
		sortedLabels[label][counter[label]] = labels[i];
		counter[label] += 1;
	}
	/*
	for(int i = 0; i < m_amountOfUsedClasses; ++i){
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
	m_gps.resize(m_amountOfUsedClasses);
	for(int iActRfRes = 0; iActRfRes < m_amountOfUsedClasses; ++iActRfRes){
		const int amountOfDataInRfRes = sortedData[iActRfRes].size();
		std::cout << "Amount of data: " << amountOfDataInRfRes << std::endl;
		if(amountOfDataInRfRes > 0){
			bool isPure = true;
			const int iClass = sortedLabels[iActRfRes][0];
			for(int k = 1; k < amountOfDataInRfRes; ++k){
				if(sortedLabels[iActRfRes][k] != iClass){
					isPure = false;
					break;
				}
			}
			std::cout << "Ispure: "<< isPure << std::endl;
			m_isPure[iActRfRes] = isPure;
			if(isPure){
				continue;
			}
			Eigen::MatrixXd dataMat;
			dataMat.conservativeResize(sortedData[iActRfRes][0].rows(), amountOfDataInRfRes);
			int i = 0;
			for(Data::iterator it = sortedData[iActRfRes].begin(); it != sortedData[iActRfRes].end(); ++it){
				dataMat.col(i++) = *it;
			}
			m_gps[iActRfRes].resize(m_amountOfUsedClasses);
			for(int iActClass = 0; iActClass < m_amountOfUsedClasses; ++iActClass){
				std::cout << "Class: " << iActClass << std::endl;
				Eigen::VectorXd y(amountOfDataInRfRes);
				bool isThere = false;
				const int hyperPoints = 36;
				Eigen::MatrixXd dataHyper;
				dataHyper.conservativeResize(dataMat.rows(), hyperPoints);
				Eigen::VectorXd yHyper(hyperPoints);
				int oneCounter = 0, minusCounter = 0, counterHyper = 0;
				// copy the first 35 points of both TODO find way of randomly taking the values
				if(amountOfDataInRfRes > hyperPoints){
					for(int j = 0; j < amountOfDataInRfRes; ++j){
						if(sortedLabels[iActRfRes][j] == iActClass){
							y[j] = 1;
							isThere = true;
							if(oneCounter < hyperPoints / 2 && counterHyper < hyperPoints){
								dataHyper.col(counterHyper) = dataMat.col(j);
								yHyper[counterHyper] = 1;
								++counterHyper;
								++oneCounter;
							}
						}else{
							y[j] = -1;
							if(minusCounter < hyperPoints / 2 && counterHyper < hyperPoints){
								dataHyper.col(counterHyper) = dataMat.col(j);
								yHyper[counterHyper] = -1;
								++minusCounter;
								++counterHyper;
							}
						}
					}
				}
				if(minusCounter < hyperPoints / 2 || oneCounter < hyperPoints / 2 ){
					// reduce the number of hyperpoints, not enough counter parts there!
					dataHyper.resize(dataMat.rows(), minusCounter + oneCounter);
					yHyper.resize(minusCounter + oneCounter);
				}
				if(yHyper.rows() < hyperPoints * 0.75){
					std::cout << "to less points -> make it pure!" << std::endl;
					m_isPure[iActRfRes] = true; // save the best value TODO
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
					actGp.init(dataMat,y);
					actGp.trainWithoutKernelOptimize();
				}
			}
		}
		std::cout <<"i: " << m_isPure[iActRfRes] << std::endl;
	}
}

int RandomForestGaussianProcess::predict(const DataElement& point, std::vector<double>& prob) const {
	const int rfLabel = m_forest.predict(point);
	if(m_isPure[rfLabel]){
		prob = std::vector<double>(m_amountOfUsedClasses, 0.0);
		prob[rfLabel] = 1.0;
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

