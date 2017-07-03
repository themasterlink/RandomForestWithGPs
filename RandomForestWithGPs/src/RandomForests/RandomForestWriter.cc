/*
 * RandomForestWriter.cc
 *
 *  Created on: 05.06.2016
 *      Author: Max
 */

#ifdef BUILD_OLD_CODE

#include "RandomForestWriter.h"
#include <fstream>
#include "../Utility/ReadWriterHelper.h"

RandomForestWriter::RandomForestWriter(){
}

RandomForestWriter::~RandomForestWriter(){
}


void RandomForestWriter::writeToFile(const std::string& filePath, const RandomForest& forest){
	if(filePath.length() == 0){
		printError("File path is empty!");
		return;
	}else if(forest.getNrOfTrees() == 0){
		printError("Number of trees is zero -> writing not possible!");
		return;
	}
	std::fstream file(filePath,std::ios::out|std::ios::binary);
	if(file.is_open()){
		writeToStream(file, forest);
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
		readFromStream(file, forest);
		file.close();
	}else{
		printError("The opening failed for: " << filePath);
		return;
	}
}

void RandomForestWriter::writeToStream(std::fstream& file, const RandomForest& forest){
	const RandomForest::DecisionTreesContainer& trees = forest.getTrees();
	const int treeSize = trees.size();
	file.write((char*) &treeSize, sizeof(int));
	for(auto it = trees.cbegin(); it != trees.cend(); ++it){
		DecisionTreeData data;
		it->writeToData(data);
		file.write((char*) &data.height, sizeof(int));
		file.write((char*) &data.nrOfInternalNodes, sizeof(int));
		file.write((char*) &data.nrOfLeaves, sizeof(int));
		file.write((char*) &data.amountOfClasses, sizeof(int));
		ReadWriterHelper::writeVector(file, data.splitValues);
		ReadWriterHelper::writeVector(file, data.dimValues);
		ReadWriterHelper::writeVector(file, data.labelsOfWinningClassInLeaves);
	}
}

void RandomForestWriter::readFromStream(std::fstream& file, RandomForest& forest){
	int treeSize;
	file.read((char*) &treeSize, sizeof(int));
	if(forest.getNrOfTrees() > 0){
		RandomForest temp(0,0,0);
		temp.init(treeSize);
		for(int i = 0; i < treeSize; ++i){
			DecisionTreeData data;
			file.read((char*) &data.height, sizeof(int));
			file.read((char*) &data.nrOfInternalNodes, sizeof(int));
			file.read((char*) &data.nrOfLeaves, sizeof(int));
			file.read((char*) &data.amountOfClasses, sizeof(int));
			ReadWriterHelper::readVector(file, data.splitValues);
			ReadWriterHelper::readVector(file, data.dimValues);
			ReadWriterHelper::readVector(file, data.labelsOfWinningClassInLeaves);
			temp.generateTreeBasedOnData(data, i);
		}
		forest.addForest(temp);
	}else{
		forest.init(treeSize);
		for(int i = 0; i < treeSize; ++i){
			DecisionTreeData data;
			file.read((char*) &data.height, sizeof(int));
			file.read((char*) &data.nrOfInternalNodes, sizeof(int));
			file.read((char*) &data.nrOfLeaves, sizeof(int));
			file.read((char*) &data.amountOfClasses, sizeof(int));
			ReadWriterHelper::readVector(file, data.splitValues);
			ReadWriterHelper::readVector(file, data.dimValues);
			ReadWriterHelper::readVector(file, data.labelsOfWinningClassInLeaves);
			forest.generateTreeBasedOnData(data, i);
		}
	}
}

#endif // BUILD_OLD_CODE