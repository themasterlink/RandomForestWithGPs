/*
 * OnlineRandomForest.cc
 *
 *  Created on: 13.10.2016
 *      Author: Max
 */

#include "OnlineRandomForest.h"
#include "../Utility/Util.h"
#include "../Base/Settings.h"

OnlineRandomForest::OnlineRandomForest(OnlineStorage<ClassPoint*>& storage,
		const int maxDepth,
		const int amountOfTrees,
		const int amountOfUsedClasses):
		m_amountOfTrees(amountOfTrees),
		m_maxDepth(maxDepth),
		m_amountOfClasses(amountOfUsedClasses),
		m_amountOfPointsUntilRetrain(0),
		m_counterForRetrain(0),
		m_amountOfUsedDims(0),
		m_storage(storage){
	Settings::getValue("OnlineRandomForest.amountOfPointsUntilRetrain", m_amountOfPointsUntilRetrain);
	double val;
	Settings::getValue("OnlineRandomForest.minUsedDataFactor", val);
	m_minMaxUsedDataFactor[0] = val;
	Settings::getValue("OnlineRandomForest.maxUsedDataFactor", val);
	m_minMaxUsedDataFactor[1] = val;
}

OnlineRandomForest::~OnlineRandomForest(){
}

void OnlineRandomForest::train(){
	if(m_storage.size() < 2){
		printError("There must be at least two points!");
		return;
	}else if(m_storage.dim() < 2){
		printError("There should be at least 2 dimensions in the data");
	}else if(m_amountOfUsedDims > m_storage.dim()){
		printError("Amount of dims can't be bigger than the dimension size!");
		return;
	}
	std::vector<int> values(m_amountOfClasses, 0);
	const int seed = 0;
	const Eigen::Vector2i minMax = getMinMaxData();
	RandomNumberGeneratorForDT generator(m_storage.dim(), minMax[0],
			minMax[1], m_storage.size(), seed);
	for(unsigned int i = 0; i < m_amountOfTrees; ++i){
		m_trees.push_back(DynamicDecisionTree(m_storage, m_maxDepth, m_amountOfClasses));
		m_trees.back().train(m_amountOfUsedDims, generator);
	}
}

void OnlineRandomForest::update(Subject* caller, unsigned int event){
	switch(event){
		case OnlineStorage<ClassPoint*>::APPEND:{
			if(m_counterForRetrain >= m_amountOfPointsUntilRetrain){
				update();
				m_counterForRetrain = 0;
			}
			break;
		}
		case OnlineStorage<ClassPoint*>::APPENDBLOCK:{
			update();
			m_counterForRetrain = 0;
			break;
		}
		case OnlineStorage<ClassPoint*>::ERASE:{
			printError("This update type is not supported here!");
			break;
		}
		default: {
			printError("This update type is not supported here!");
			break;
		}
	}
}

void OnlineRandomForest::update(){
	double startAmountOfRight = 0;
	DecisionTreeIterator itWorst = findWorstPerformingTree(startAmountOfRight);
	if(startAmountOfRight > 90.){
		printDebug("No update needed!");
		return;
	}
	const int seed = 0;
	const Eigen::Vector2i minMax = getMinMaxData();
	RandomNumberGeneratorForDT generator(m_storage.dim(), minMax[0], minMax[1], m_storage.size(), seed);
	double lastWorstAmountOfRight = startAmountOfRight;
	for(unsigned int updateStep = 0; startAmountOfRight + 10 > lastWorstAmountOfRight && updateStep < m_trees.size() * 0.25; ++updateStep){
		itWorst->train(m_amountOfUsedDims, generator);
		itWorst = findWorstPerformingTree(startAmountOfRight);
	}
}

OnlineRandomForest::DecisionTreeIterator OnlineRandomForest::findWorstPerformingTree(double& correctAmount){
	int minCorrect = m_storage.size();
	DecisionTreeIterator itWorst = m_trees.end();
	for(DecisionTreeIterator itTree = m_trees.begin(); itTree != m_trees.end(); ++itTree){
		int correct = 0;
		for(OnlineStorage<ClassPoint*>::ConstIterator it = m_storage.begin(); it != m_storage.end(); ++it){
			ClassPoint& point = *(*it);
			if(point.getLabel() == itTree->predict(point)){
				++correct;
			}
		}
		if(minCorrect > correct){
			minCorrect = correct;
			itWorst = itTree;
		}
	}
	correctAmount = minCorrect / (double) m_storage.size() * 100.;
	return itWorst;
}

int OnlineRandomForest::predict(const DataPoint& point) const {
	std::vector<int> values(m_amountOfClasses, 0);
	for(DecisionTreeConstIterator it = m_trees.cbegin(); it != m_trees.cend(); ++it){
		++values[it->predict(point)];
	}
	return std::distance(values.cbegin(), std::max_element(values.cbegin(), values.cend()));
}

void OnlineRandomForest::predictData(const Data& points, Labels& labels) const{

}

void OnlineRandomForest::getLeafNrFor(std::vector<int>& leafNrs){
	leafNrs = std::vector<int>(m_amountOfClasses, 0);
	for(OnlineStorage<ClassPoint*>::ConstIterator it = m_storage.begin(); it != m_storage.end(); ++it){
		leafNrs[predict(**it)] += 1;
	}
}

Eigen::Vector2i OnlineRandomForest::getMinMaxData(){
	Eigen::Vector2i minMax;
	minMax[0] = m_minMaxUsedDataFactor[0] * m_storage.size();
	minMax[1] = m_minMaxUsedDataFactor[1] * m_storage.size();
	return minMax;
}
