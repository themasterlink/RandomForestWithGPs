/*
 * OnlineRandomForest.cc
 *
 *  Created on: 13.10.2016
 *      Author: Max
 */

#include "OnlineRandomForest.h"
#include "../Utility/Util.h"

OnlineRandomForest::OnlineRandomForest(const int maxDepth,
		const int amountOfTrees, const int amountOfUsedClasses):
		m_amountOfTrees(amountOfTrees),
		m_amountOfClasses(amountOfUsedClasses){
}

OnlineRandomForest::~OnlineRandomForest(){
}

void OnlineRandomForest::train(const ClassData& data, const int amountOfUsedDims,
		const Eigen::Vector2i minMaxUsedData){
	if(data.size() < 2){
		printError("There must be at least two points!");
		return;
	}else if(data[0]->rows() < 2){
		printError("There should be at least 2 dimensions in the data");
	}else if(amountOfUsedDims > data[0]->rows()){
		printError("Amount of dims can't be bigger than the dimension size!");
		return;
	}
	std::vector<int> values(m_amountOfClasses, 0);
	const int seed = 0;
	RandomNumberGeneratorForDT generator(data[0]->rows(), minMaxUsedData[0],
			minMaxUsedData[1], data.size(), seed);
	for(unsigned int i = 0; i < m_amountOfTrees; ++i){
		m_trees.push_back(DecisionTree(0,0));
		m_trees.back().train(data, amountOfUsedDims, generator);
	}
}

void OnlineRandomForest::update(const ClassData& data, const int amountOfUsedDims,
		const Eigen::Vector2i minMaxUsedData){
	double startAmountOfRight = 0;
	DecisionTreeIterator itWorst = findWorstPerformingTree(data, startAmountOfRight);
	if(startAmountOfRight > 90.){
		printDebug("No update needed!");
		return;
	}
	double lastWorstAmountOfRight = startAmountOfRight;
	for(unsigned int updateStep = 0; startAmountOfRight + 10 > lastWorstAmountOfRight && updateStep < m_trees.size() * 0.25; ++updateStep){

	}
}

OnlineRandomForest::DecisionTreeIterator OnlineRandomForest::findWorstPerformingTree(const ClassData& data, double& correctAmount){
	int minCorrect = data.size();
	DecisionTreeIterator itWorst = m_trees.end();
	for(DecisionTreeIterator itTree = m_trees.begin(); itTree != m_trees.end(); ++itTree){
		int correct = 0;
		for(ClassDataConstIterator it = data.cbegin(); it != data.cend(); ++it){
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
	correctAmount = minCorrect / (double) data.size() * 100.;
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

void OnlineRandomForest::getLeafNrFor(const ClassData& data, std::vector<int>& leafNrs){

}
