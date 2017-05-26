/*
 * ReadWriterHelper.cc
 *
 *  Created on: 14.07.2016
 *      Author: Max
 */

#include "ReadWriterHelper.h"
#include "../Data/ClassKnowledge.h"
#include<memory>

ReadWriterHelper::ReadWriterHelper() {
	// TODO Auto-generated constructor stub

}

ReadWriterHelper::~ReadWriterHelper() {
	// TODO Auto-generated destructor stub
}

void ReadWriterHelper::writeVector(std::fstream& stream, const VectorX& vector){
	VectorX::Index rows = vector.rows();
	stream.write((char*) (&rows), sizeof(VectorX::Index));
	stream.write((char*) vector.data(), rows*sizeof(VectorX::Scalar));
}

void ReadWriterHelper::readVector(std::fstream& stream, VectorX& vector){
	VectorX::Index rows=0;
	stream.read((char*) (&rows),sizeof(VectorX::Index));
	vector.resize(rows);
	stream.read( (char *) vector.data(), rows*sizeof(VectorX::Scalar));
}

void ReadWriterHelper::readPoint(std::fstream& stream, LabeledVectorX& vector){
	VectorX::Index rows=0;
	stream.read((char*) (&rows),sizeof(VectorX::Index));
	vector.resize(rows);
	stream.read( (char *) vector.data(), rows*sizeof(VectorX::Scalar));
	unsigned int label;
	stream.read( (char *) (&label), sizeof(unsigned int));
	vector.setLabel(label);
}

void ReadWriterHelper::writePoint(std::fstream& stream, const LabeledVectorX& vector){
	VectorX::Index rows = vector.rows();
	stream.write((char*) (&rows), sizeof(VectorX::Index));
	stream.write((char*) vector.data(), rows*sizeof(VectorX::Scalar));
	unsigned int label = vector.getLabel();
	stream.write((char*) (&label), sizeof(unsigned int));
}

void ReadWriterHelper::writeDynamicTree(std::fstream& stream, const DynamicDecisionTree& tree){
	// write standart information
	const unsigned int amountOfDims = ClassKnowledge::amountOfDims();
	const unsigned int amountOfClasses = ClassKnowledge::amountOfClasses();
	stream.write((char*) (&amountOfDims), sizeof(unsigned int));
	stream.write((char*) (&amountOfClasses), sizeof(unsigned int));
	stream.write((char*) (&tree.m_amountOfClasses), sizeof(unsigned int));
	stream.write((char*) (&tree.m_maxDepth), sizeof(unsigned int));
	writeVector(stream, tree.m_splitValues);
	writeVector(stream, tree.m_splitDim);
	writeVector(stream, tree.m_labelsOfWinningClassesInLeaves);
}

void ReadWriterHelper::readDynamicTree(std::fstream& stream, DynamicDecisionTree& tree){
	// read standart information
	const unsigned int amountOfDims = ClassKnowledge::amountOfDims();
	const unsigned int amountOfClasses = ClassKnowledge::amountOfClasses();
	unsigned int readDims = 0;
	unsigned int readClasses = 0;
	stream.read((char*) (&readDims), sizeof(unsigned int));
	stream.read((char*) (&readClasses), sizeof(unsigned int));
	if(readDims == amountOfDims && readClasses == amountOfClasses){ // check standart information
		unsigned int amountOfUsedClasses = 0;
		unsigned int depth = 0;
		stream.read((char*) (&amountOfUsedClasses), sizeof(unsigned int));
		stream.read((char*) (&depth), sizeof(unsigned int));
		tree.prepareForSetting(depth, amountOfUsedClasses);
		readVector(stream, tree.m_splitValues);
		readVector(stream, tree.m_splitDim);
		readVector(stream, tree.m_labelsOfWinningClassesInLeaves);
	}else{
		printError("The reading process failed the saved tree was not trained on this data set! Had dims: " << readDims << " and classes: " << readClasses);
	}
}

void ReadWriterHelper::writeBigDynamicTree(std::fstream& stream, const BigDynamicDecisionTree& tree){
	// read standart information
	const unsigned int amountOfDims = ClassKnowledge::amountOfDims();
	const unsigned int amountOfClasses = ClassKnowledge::amountOfClasses();
	stream.write((char*) (&amountOfDims), sizeof(unsigned int));
	stream.write((char*) (&amountOfClasses), sizeof(unsigned int));
	stream.write((char*) (&tree.m_amountOfClasses), sizeof(unsigned int));
	stream.write((char*) (&tree.m_maxDepth), sizeof(unsigned int));
	const unsigned int fastLayers = tree.m_fastInnerTrees.size();
	const unsigned int smallLayers = tree.m_smallInnerTrees.size();
	stream.write((char*) (&fastLayers), sizeof(unsigned int));
	stream.write((char*) (&smallLayers), sizeof(unsigned int));
	for(unsigned int i = 0; i < fastLayers; ++i){
		const unsigned int size = tree.m_fastInnerTrees[i].size();
		stream.write((char*) (&size), sizeof(unsigned int));
	}
	for(unsigned int i = 0; i < smallLayers; ++i){
		const unsigned int size = tree.m_smallInnerTrees[i].size();
		stream.write((char*) (&size), sizeof(unsigned int));
	}
	for(unsigned int i = 0; i < fastLayers; ++i){
		for(unsigned int j = 0; j < tree.m_fastInnerTrees[i].size(); ++j){
			const bool useThisTree = tree.m_fastInnerTrees[i][j] != nullptr;
			stream.write((char*) (&useThisTree), sizeof(bool));
			if(useThisTree){
				writeDynamicTree(stream, *tree.m_fastInnerTrees[i][j]);
			}
		}
	}
	for(unsigned int i = 0; i < smallLayers; ++i){
		for(const auto& innerTree : tree.m_smallInnerTrees[i]){
			stream.write((char*) (&innerTree.first), sizeof(unsigned int));
			writeDynamicTree(stream, *innerTree.second);
		}
	}
}

void ReadWriterHelper::readBigDynamicTree(std::fstream& stream, BigDynamicDecisionTree& tree){
	// read standart information
	const unsigned int amountOfDims = ClassKnowledge::amountOfDims();
	const unsigned int amountOfClasses = ClassKnowledge::amountOfClasses();
	unsigned int readDims = 0;
	unsigned int readClasses = 0;
	stream.read((char*) (&readDims), sizeof(unsigned int));
	stream.read((char*) (&readClasses), sizeof(unsigned int));
	if(readDims == amountOfDims && readClasses == amountOfClasses){ // check standart information
		unsigned int amountOfUsedClasses = 0;
		unsigned int depth = 0;
		stream.read((char*) (&amountOfUsedClasses), sizeof(unsigned int));
		stream.read((char*) (&depth), sizeof(unsigned int));
		unsigned int fastLayers, smallLayers;
		stream.read((char*) (&fastLayers), sizeof(unsigned int));
		stream.read((char*) (&smallLayers), sizeof(unsigned int));
		tree.prepareForSetting(depth, amountOfUsedClasses, fastLayers + smallLayers, fastLayers, smallLayers);
		// depth per layer is calculated in prepareForSetting
		for(unsigned int i = 0; i < fastLayers; ++i){
			unsigned int size;
			stream.read((char*) (&size), sizeof(unsigned int));
			tree.m_fastInnerTrees[i].resize(size);
		}
		std::vector<unsigned int> sizes(smallLayers, 0);
		for(unsigned int i = 0; i < smallLayers; ++i){
			unsigned int size;
			stream.read((char*) (&size), sizeof(unsigned int));
			sizes[i] = size;
		}
		for(unsigned int i = 0; i < fastLayers; ++i){
			for(unsigned int j = 0; j < tree.m_fastInnerTrees[i].size(); ++j){
				bool useThisTree;
				stream.read((char*) (&useThisTree), sizeof(bool));
				if(useThisTree){
					tree.m_fastInnerTrees[i][j] = new DynamicDecisionTree(tree.m_storage);
					readDynamicTree(stream, *tree.m_fastInnerTrees[i][j]);
				}else{
					tree.m_fastInnerTrees[i][j] = nullptr;
				}
			}
		}
		for(unsigned int i = 0; i < smallLayers; ++i){
			for(unsigned int j = 0; j < sizes[i]; ++j){
				unsigned int nr;
				stream.read((char*) (&nr), sizeof(unsigned int));
				DynamicDecisionTree* pTree = new DynamicDecisionTree(tree.m_storage);
				readDynamicTree(stream, *pTree);
				tree.m_smallInnerTrees[i].insert(BigDynamicDecisionTree::SmallTreeInnerPair(nr, pTree));
			}
		}

	}

}
