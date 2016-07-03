/*
 * RandomForestGaussianProcess.cc
 *
 *  Created on: 29.06.2016
 *      Author: Max
 */

#include "../RandomForestGaussianProcess/RandomForestGaussianProcess.h"

#include "../GaussianProcess/GaussianProcessMultiClass.h"

#include "../Data/DataWriterForVisu.h"

RandomForestGaussianProcess::RandomForestGaussianProcess(const Data& data, const Labels& labels,
		const int heightOfTrees, const int amountOfTrees,
		const int amountOfUsedClasses) :
	m_data(data), m_labels(labels), m_heightOfTrees(heightOfTrees),
	m_amountOfTrees(amountOfTrees), m_amountOfUsedClasses(amountOfUsedClasses),
	m_forest(m_heightOfTrees, m_amountOfTrees, m_amountOfUsedClasses),
	m_isPure(m_amountOfUsedClasses, false){

	if(m_data.size() == 0){
		printError("No data given!");
		return;
	}

	Eigen::Vector2i minMaxUsedData;
	minMaxUsedData << data.size() * 0.2 , data.size() * 0.6 ;
	m_forest.train(m_data, m_labels, m_amountOfUsedClasses, minMaxUsedData);
	std::vector<int> guessedLabels;
	m_forest.predictData(m_data, guessedLabels);
	std::vector<int> countClasses(m_amountOfUsedClasses, 0);
	for(int i = 0; i < guessedLabels.size(); ++i){
		countClasses[guessedLabels[i]] = 0;
	}

	std::vector<Data > sortedData;
	std::vector<Labels > sortedLabels;

	sortedData.resize(m_amountOfUsedClasses);
	sortedLabels.resize(m_amountOfUsedClasses);
	for(int i = 0; i < m_data.size();  ++i){
		sortedData[guessedLabels[i]].push_back(m_data[i]);
		sortedLabels[guessedLabels[i]].push_back(m_labels[i]);
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
		if(amountOfDataInRfRes > 0){
			bool isPure = true;
			const int iClass = sortedLabels[iActRfRes][0];
			for(int k = 1; k < amountOfDataInRfRes; ++k){
				if(sortedLabels[iActRfRes][k] != iClass){
					isPure = false;
					break;
				}
			}
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

				Eigen::VectorXd y(amountOfDataInRfRes);
				bool isThere = false;
				for(int j = 0; j < amountOfDataInRfRes; ++j){
					if(sortedLabels[iActRfRes][j] == iActClass){
						y[j] = 1;
						isThere = true;
					}else{
						y[j] = -1;
					}
				}
				if(isThere){ // an element of this class is here, so a gp must be preformed!
					m_gps[iActRfRes][iActClass].init(dataMat,y);
					m_gps[iActRfRes][iActClass].train();
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

