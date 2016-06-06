/*
 * RandomForestWriter.cc
 *
 *  Created on: 05.06.2016
 *      Author: Max
 */

#include "RandomForestWriter.h"
#include <fstream>

RandomForestWriter::RandomForestWriter()
{
	// TODO Auto-generated constructor stub

}

RandomForestWriter::~RandomForestWriter()
{
	// TODO Auto-generated destructor stub
}


void RandomForestWriter::writeToFile(const std::string& filePath, const RandomForest& forest){
	if(filePath.length() == 0){
		printError("File path is empty!");
		return;
	}else if(forest.getNrOfTrees() == 0){
		printError("Number of trees is zero -> writing not possible!");
		return;
	}
	const RandomForest::DecisionTreesContainer& trees = forest.getTrees();
	std::fstream file(filePath,std::ios::out|std::ios::binary);
	if(file.is_open()){
		file << (int) trees.size() << "\n";
		for(RandomForest::DecisionTreesContainer::const_iterator it = trees.cbegin(); it != trees.cend(); ++it){
			DecisionTreeData data;
			it->writeToData(data);
			file << data.height  << "\n";
			file << data.nrOfInternalNodes << "\n";
			file << data.nrOfLeaves << "\n";
			file << data.amountOfClasses << "\n";
			Utility::writeVecToStream(file, data.splitValues);
			Utility::writeVecToStream(file, data.dimValues);
			Utility::writeVecToStream(file, data.labelsOfWinningClassInLeaves);
		}
		file.close();
	}else{
		printError("The opening failed for: " << filePath);
		return;
	}
}

void RandomForestWriter::readFromFile(const std::string& filePath, RandomForest& forest){
	if(filePath.length() == 0){
		printError("File path is empty!");
		return;
	}
	std::fstream file(filePath,std::ios::binary| std::ios::in);
	if(file.is_open()){
		int treeSize;
		file >> treeSize;
		if(forest.getNrOfTrees() > 0){
			RandomForest temp(0,0,0);
			temp.init(treeSize);
			for(int i = 0; i < treeSize; ++i){
				DecisionTreeData data;
				file >> data.height;
				file >> data.nrOfInternalNodes;
				file >> data.nrOfLeaves;
				file >> data.amountOfClasses;
				Utility::readVecFromStream(file, data.splitValues);
				Utility::readVecFromStream(file, data.dimValues);
				Utility::readVecFromStream(file, data.labelsOfWinningClassInLeaves);
				temp.generateTreeBasedOnData(data, i);
			}
			forest.addForest(temp);
		}else{
			forest.init(treeSize);
			for(int i = 0; i < treeSize; ++i){
				DecisionTreeData data;
				file >> data.height;
				file >> data.nrOfInternalNodes;
				file >> data.nrOfLeaves;
				file >> data.amountOfClasses;
				Utility::readVecFromStream(file, data.splitValues);
				Utility::readVecFromStream(file, data.dimValues);
				Utility::readVecFromStream(file, data.labelsOfWinningClassInLeaves);
				forest.generateTreeBasedOnData(data, i);
			}
		}
		file.close();
	}else{
		printError("The opening failed for: " << filePath);
		return;
	}
}
