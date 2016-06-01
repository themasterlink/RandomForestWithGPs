/*
 * DecisionTree.cpp
 *
 *  Created on: 31.05.2016
 *      Author: Max
 */

#include "DecisionTree.h"
#include <cmath>
#include "../Utility/Util.h"
#include <random>
#include <algorithm>

DecisionTree::DecisionTree(const int maxDepth) :
m_maxDepth(maxDepth), m_maxNodeNr(pow(2, maxDepth + 1) - 1),
m_maxInternalNodeNr(pow(2, maxDepth) - 1), m_splitValues(m_maxInternalNodeNr + 1) { // + 1 -> no use of the first element
}

DecisionTree::~DecisionTree() {
	// TODO Auto-generated destructor stub
}

void DecisionTree::train(const Data& data, const Labels& labels, const int amountOfUsedDims, const Eigen::Vector2i minMaxUsedData){
	const bool verbose = false;
	if(data.size() != labels.size()){
		printError("Label and data size are not equal!"); return;
	}else if(data.size() < 2){
		printError("There must be at least two points!"); return;
	}else if(data[0].rows() > 1){
		printError("There should be at least 2 dimensions in the data");
	}else if(amountOfUsedDims > data[0].rows()){
		printError("Amount of dims can't be bigger than the dimension size!"); return;
	}

	std::default_random_engine generator;
	std::uniform_int_distribution<int> uniformDist_Data(0,data[0].rows() - 1); // 0 ... (dimension of data - 1)
	std::uniform_int_distribution<int> uniformDist_usedData(minMaxUsedData[0], minMaxUsedData[1]); // TODO add check!
	std::uniform_int_distribution<int> uniformDist_Dimension(0,amountOfUsedDims - 1); // 0 ... (dimension of data - 1)

	std::vector<int> usedDims(amountOfUsedDims);
	for(int i = 0; i < amountOfUsedDims; ++i){
		bool doAgain = false;
		do{
			const int randNr = uniformDist_Data(generator);  // generates number in the range 0...data.rows() - 1;
			for(int j = 0; j < i; ++j){
				if(randNr == usedDims[j]){
					doAgain = true; break;
				}
			}
			if(!doAgain){
				usedDims[i] = randNr;
			}
		}while(doAgain);
	}
	if(verbose)
		for(int i = 0; i < amountOfUsedDims; ++i){
			std::cout << usedDims[i] << ", " << std::endl;
		}

	for(int i = 1; i < m_maxInternalNodeNr + 1; ++i){ // first element is not used!
		// calc split value for each node
		
		// choose dimension for split
		const int randDim = uniformDist_Dimension(generator);  // generates number in the range 0...amountOfUsedDims - 1

		const int amountOfUsedData = uniformDist_usedData(generator);
		for(int j = 0; j < amountOfUsedData; ++j){
			const int randElementId = uniformDist_Data(generator);

		}


	}

}
