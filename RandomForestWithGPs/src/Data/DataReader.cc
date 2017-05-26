/*
 * DataReader.cc
 *
 *  Created on: 04.06.2016
 *      Author: Max
 */

#include "DataReader.h"
#include "DataBinaryWriter.h"
#include <iostream>
#include "../Utility/ReadWriterHelper.h"
#include "../Utility/Util.h"
#include "../Base/Settings.h"
#include "ClassKnowledge.h"
#include "DataConverter.h"

#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>

DataReader::DataReader(){
}

DataReader::~DataReader(){
}

void DataReader::readFromBinaryFile(LabeledData& data, const std::string& inputName, const unsigned int amountOfData){
	std::string line;
	std::fstream input(inputName);
	if(input.is_open()){
		long size;
		input.read((char*) &size, sizeof(long));
		const unsigned long lastSize = data.size();
		data.reserve((unsigned long) size + lastSize);
		printOnScreen("Read " << size << " points from binary: " << inputName);
		if(amountOfData > size && amountOfData != (unsigned int) INT_MAX){
			printWarning("The amount of data provided is smaller than the desired amount!");
		}
		for(unsigned long i = lastSize; i < size + lastSize; ++i){
			LabeledVectorX* p = new LabeledVectorX();
			ReadWriterHelper::readPoint(input, *p);
			if(!ClassKnowledge::hasClassName(p->getLabel())){
				ClassKnowledge::setNameFor(number2String(p->getLabel()), p->getLabel());
			}
			data.push_back(p);
		}
		if(data.size() > 0 && ClassKnowledge::amountOfDims() == 0){
			ClassKnowledge::setAmountOfDims((unsigned int) data[0]->rows());
		}
		input.close();
	}else{
		printError("File was not found: " << inputName);
	}
}

void DataReader::readFromFile(LabeledData& data, const std::string& inputName, const unsigned int amountOfData){
	std::string line;
	std::ifstream input(inputName);
	if(input.is_open()){
		while(std::getline(input, line)){
			std::vector<std::string> elements;
			std::stringstream ss(line);
			std::string item;
			while(std::getline(ss, item, ',')){
				elements.push_back(item);
			}
			LabeledVectorX* newEle = new LabeledVectorX((const int) (elements.size() - 1),
												(const unsigned int) std::stoi(elements.back()));
			for(int i = 0; i < (int) elements.size() - 1; ++i){
				(*newEle)[i] = (Real) std::stod(elements[i]);
			};
			data.push_back(newEle);
			if((unsigned int) data.size() == amountOfData){
				break;
			}
		}
		if(data.size() > 0 && ClassKnowledge::amountOfDims() == 0){
			ClassKnowledge::setAmountOfDims((unsigned int) data[0]->rows());
		}
		input.close();
	}else{
		printError("File was not found: " << inputName);
	}
}

void DataReader::readFromFile(LabeledData& data, const std::string& inputName,
		const unsigned int amountOfData, const unsigned int classNr, const bool readTxt,
							  const bool containsDegrees){
	std::string inputPath(inputName);
	if(boost::filesystem::exists(inputName + ".binary") && !readTxt){
		// is a binary file -> faster loading!
		inputPath += ".binary";
		std::fstream input(inputPath, std::fstream::in);
		if(input.is_open()){
			long size;
			input.read((char*) &size, sizeof(long));
			data.resize(std::min((unsigned long) amountOfData, (unsigned long) size));
			for(long i = 0; i < std::min((long) amountOfData,size); ++i){
				data[i] = new LabeledVectorX();
				ReadWriterHelper::readPoint(input, *data[i]);
				data[i]->setLabel(classNr);
			}
		}else{
			printError("The file could not be opened: " << inputPath);
		}
		input.close();
	}else if(boost::filesystem::exists(inputName + ".txt")){
		inputPath += ".txt";
		printOnScreen("Read txt from " + inputPath);
		std::ifstream input(inputPath);
		if(input.is_open()){
			std::string line;
			while(std::getline(input, line)){
				std::vector<std::string> elements;
				std::stringstream ss(line);
				std::string item;
				while(std::getline(ss, item, ' ')){
					elements.push_back(item);
				}
				LabeledVectorX* newEle = new LabeledVectorX(int(elements.size()), classNr);
				for(unsigned int i = 0; i < elements.size(); ++i){
					newEle->coeffRef(i) = (Real) std::stod(elements[i]);
				}
				data.push_back(newEle);
				if(data.size() == amountOfData){
					break;
				}
			}
			input.close();
			DataBinaryWriter::toFile(data, inputName + ".binary"); // create binary to avoid rereading .txt
		}else{
			printError("The file could not be opened: " << inputPath);
		}
	}else if(boost::filesystem::exists(inputName + ".csv")){
		inputPath += ".csv";
		printOnScreen("Read csv from " + inputPath);
		std::ifstream input(inputPath);
		if(input.is_open()){
			std::string line;
			std::set<unsigned int> classes;
			auto size = 100;
			while(std::getline(input, line)){
				std::vector<std::string> elements;
				elements.reserve(size);
				std::stringstream ss(line);
				std::string item;
				while(std::getline(ss, item, ',')){
					elements.push_back(item);
				}
				if(elements.size() > 0){
					auto label = (unsigned int) std::stoi(elements.front());
					auto maxSize = (unsigned int) elements.size();
					size = maxSize;
					unsigned int start = 1;
					if(containsDegrees){
						maxSize -= 3;
						// last element should be the label
						label = (unsigned int) std::stof(elements[maxSize]);
						start = 0;
					}
					classes.insert(label);
					// containsDegrees removes the last two degrees at the end of each line
					// minus 1 because the first element is the label
					LabeledVectorX* newEle = new LabeledVectorX((int)(maxSize) - start, label);
					for(unsigned int i = start; i < maxSize; ++i){
						newEle->coeffRef(i - start) = (Real) std::stod(elements[i]);
					}
					data.push_back(newEle);
					if(data.size() == amountOfData){
						break;
					}
				}
			}
			std::stringstream ss;
			for(auto c : classes){
				ss << c << ", ";
			}
			printOnScreen("Classes: " << ss.str());
			input.close();
//			if(data.size() > 0){
//				DataBinaryWriter::toFile(data, inputName + ".binary"); // create binary to avoid rereading .txt
//			}
		}else{
			printError("The file could not be opened: " << inputPath);
		}
	}else{
		printError("File was not found for .txt or .binary: " << inputName);
	}
	if(data.size() > 0 && ClassKnowledge::amountOfDims() == 0){
		ClassKnowledge::setAmountOfDims((unsigned int) data[0]->rows());
	}
}

void DataReader::readFromFiles(DataSets& dataSets, const std::string& folderLocation, const unsigned int amountOfData, const bool readTxt, bool& didNormalizeData){
	boost::filesystem::path targetDir(folderLocation);
	boost::filesystem::directory_iterator end_itr;
	// cycle through the directory
	unsigned int amountOfClasses = ClassKnowledge::amountOfClasses();
	std::string fakeDataLocation;
	Settings::getValue("TotalStorage.folderLocFake", fakeDataLocation);
	boost::filesystem::path fakeDataLoc(fakeDataLocation);
	int type = 0;
	if(targetDir.parent_path().parent_path().filename() == "mnistOrg"){
		type = 1;
	}else if(targetDir.parent_path().parent_path().filename() == "uspsOrg"){
		type = 2;
	}else if(targetDir.parent_path().filename() == "washington"){
		type = 3;
	}else if(targetDir.parent_path().filename() == "simon"){
		type = 4;
	}else if(targetDir.filename() == "washingtonData"){ // new data from Max Durner
		type = 5;
	}else if(targetDir.parent_path().filename() == fakeDataLoc.parent_path().filename()){
		type = 0;
	}else{
		printError("This type is not supported here: " << targetDir.parent_path().filename());
		quitApplication();
		return;
	}
	if(targetDir.parent_path().filename() == "mnist" && type == 0){
//		didNormalizeData = true; // did perform that before write out mnistOrg
	}
	switch(type){
	case 0:{
		for(boost::filesystem::directory_iterator itr(targetDir); itr != end_itr; ++itr){
			if(boost::filesystem::is_directory(itr->path())){
				const std::string name(itr->path().filename().c_str());
				LabeledData data;
				std::string filePath(itr->path().c_str());
				filePath += "/vectors";
				readFromFile(data, filePath, amountOfData, amountOfClasses, readTxt);
				ClassKnowledge::setNameFor(name, amountOfClasses);
				++amountOfClasses;
				dataSets.insert(DataSetPair(name, data));
			}
		}
		break;
	}
	case 1:{
		LabeledData data[10];
		std::vector<unsigned char> labels;
		const std::string inputPath(folderLocation);
		if(boost::filesystem::exists(inputPath + "/labels.mnist") && boost::filesystem::exists(inputPath + "/data.mnist")){
			for(std::string fileName : {"/labels.mnist", "/data.mnist"}){
				std::ifstream input(inputPath + fileName);
				if(input.is_open()){
					input.seekg(0);
					int_fast32_t magicNumber;
					input.read((char*) &magicNumber, 4);
					magicNumber = highEndian2LowEndian(magicNumber);
					if(magicNumber == 2051){
						int_fast32_t size;
						input.read((char*) &size, 4);
						size = highEndian2LowEndian(size);
						int_fast32_t rows;
						input.read((char*) &rows, 4);
						rows = highEndian2LowEndian(rows);
						int_fast32_t cols;
						input.read((char*) &cols, 4);
						cols = highEndian2LowEndian(cols);
						if(labels.size() != (unsigned long) size){
							printError("The labels should be read first!");
							getchar();
							return;
						}
						for(unsigned int i = 0; i < (unsigned int) size; ++i){
							LabeledVectorX* newEle = new LabeledVectorX(int(rows * cols), labels[i]);
							for(unsigned int r = 0; r < (unsigned int) rows; ++r){
								for(unsigned int c = 0; c < (unsigned int) cols; ++c){
									unsigned char ele;
									input.read((char*) &ele, 1);
									(*newEle).coeffRef(r * rows + c) = ele;
								}
							}
							data[labels[i]].push_back(newEle);
						}
					}else if(magicNumber == 2049){
						int_fast32_t size;
						input.read((char*) &size, 4);
						size = highEndian2LowEndian(size);
						labels.resize((unsigned long) size);
						for(unsigned int i = 0; i < (unsigned int) size; ++i){
							input.read((char*) &labels[i], 1);
						}
					}else{
						printError("This .mnist type is unknown!");
					}
				}
				input.close();
			}
		}else{
			printError("There is no data.mnist and labels.mnist file in this folder: " + inputPath);
		}


		if(data[0].size() == 0){
			printError("No data was read!");
			return;
		}
		const bool createNewClasses = ClassKnowledge::amountOfClasses() == 0;
		for(unsigned int i = 0; i < 10; ++i){
			if(createNewClasses){
				ClassKnowledge::setNameFor(number2String(i), i);
			}
			dataSets.insert( DataSetPair(number2String(i), data[i]));
		}
//		const unsigned int amountOfDim = data[0][0]->rows();
//		std::vector<Vector2> minMaxValues(amountOfDim);
//		for(unsigned int k = 0; k < amountOfDim; ++k){
//			minMaxValues[k][0] = REAL_MAX;
//			minMaxValues[k][1] = NEG_REAL_MAX;
//		}
//		for(DataSetsIterator it = dataSets.begin(); it != dataSets.end(); ++it){
//			for(unsigned int t = 0; t < it->second.size(); ++t){
//				LabeledVectorX& point = *it->second[t];
//				for(unsigned int k = 0; k < amountOfDim; ++k){
//					if(point.coeff(k) < minMaxValues[k].coeff(0)){
//						minMaxValues[k].coeffRef(0) = point.coeff(k);
//					}
//					if(point.coeff(k) > minMaxValues[k].coeff(1)){
//						minMaxValues[k].coeffRef(1) = point.coeff(k);
//					}
//				}
//			}
//		}
//		int newDim = amountOfDim;
////		bool diffIsEqual = false;
////		int lastDim = -1;
//		std::vector<bool> notUsed(amountOfDim, false);
//		for(int iActDim = amountOfDim - 1; iActDim >= 0; --iActDim){
//			printOnScreen("iActDim: " << iActDim);
//			if(minMaxValues[iActDim].coeff(0) == minMaxValues[iActDim].coeff(1)){
//				--newDim;
//				for(DataSetsIterator it = dataSets.begin(); it != dataSets.end(); ++it){
//					for(unsigned int t = 0; t < it->second.size(); ++t){
//						LabeledVectorX& point = *it->second[t];
//						for(unsigned int t = iActDim; t < newDim; ++t){
//							point.coeffRef(t) = point.coeff(t+1);
//						}
//					}
//				}
//				notUsed[iActDim] = true;
//			}
////			else if(minMaxValues[iActDim].coeff(0) != minMaxValues[iActDim].coeff(1) && diffIsEqual){
////				// remove this dimension in every pixel
////				int diff = lastDim - iActDim;
////				for(unsigned int i = iActDim; i < lastDim; ++i){
////					notUsed[i] = true;
////				}
////				printOnScreen("Remove dimension from exclusive " << iActDim << " to " << lastDim << " from all points");
////				++iActDim; // get to last value
////				newDim -= diff;
////				std::stringstream str2;
////				for(unsigned int l = iActDim; l < std::min((int) amountOfDim, lastDim + 5); ++l){
////					str2 << dataSets.begin()->second[0]->coeff(l) << ", ";
////				}
////				printOnScreen("From " << iActDim << " to " << std::min((int) amountOfDim, lastDim + 5) << ": " << str2.str());
////				for(DataSetsIterator it = dataSets.begin(); it != dataSets.end(); ++it){
////					for(unsigned int t = 0; t < it->second.size(); ++t){
////						ClassPoint& point = *it->second[t];
////						if(iActDim < lastDim - 1){ // not if the last element is removed
////							for(unsigned int t = iActDim; t < std::min(newDim, (int) amountOfDim - diff); ++t){
////								point.coeffRef(t) = point.coeff(t+diff);
////							}
////						}
////						point.resize(newDim);
////					}
////				}
////				std::stringstream str3;
////				for(unsigned int l = iActDim; l < std::min((int) amountOfDim, lastDim + 5); ++l){
////					str3 << dataSets.begin()->second[0]->coeff(l) << ", ";
////				}
////				printOnScreen("From " << iActDim << " to " << std::min((int) amountOfDim, lastDim + 5) << ": " << str3.str());
////				diffIsEqual = false;
////				--iActDim; // get back last value
////			}
//		}
//		for(DataSetsIterator it = dataSets.begin(); it != dataSets.end(); ++it){
//			for(unsigned int t = 0; t < it->second.size(); ++t){
//				(*it->second[t]).resize(newDim);
//			}
//		}
//		printOnScreen("Reduced from " << amountOfDim << " to " << newDim);
//		cv::Mat img(28, 28, CV_8UC3, cv::Scalar(0, 0, 0));
//		for(unsigned int r = 0; r < 28; ++r){
//			for(unsigned int c = 0; c < 28; ++c){
//				cv::Vec3b& color = img.at<cv::Vec3b>(r,c);
//				color[0] = (notUsed[r * 28 + c] ? 1 : 0) * 255;
//				color[1] = (notUsed[r * 28 + c] ? 1 : 0) * 255;
//				color[2] = (notUsed[r * 28 + c] ? 1 : 0) * 255;
//			}
//		}
//		cv::imwrite("test.png", img);
//		openFileInViewer("test.png");
//		ClassKnowledge::setAmountOfDims(newDim);

		ClassKnowledge::setAmountOfDims(28 * 28);
		for(DataSetsIterator it = dataSets.begin(); it != dataSets.end(); ++it){
			for(unsigned int t = 0; t < it->second.size(); ++t){
				LabeledVectorX& point = *it->second[t];
				for(unsigned int k = 0; k < ClassKnowledge::amountOfDims(); ++k){
					point.coeffRef(k) /= 255.;
				}
			}
		}

//		DataPoint center, var;
//		DataConverter::centerAndNormalizeData(dataSets, center, var);
		didNormalizeData = true;

//		std::string mnistFolder = targetDir.parent_path().parent_path().c_str();
//			mnistFolder += "/mnist/";
//		if(!boost::filesystem::exists(mnistFolder)){
//			system(("mkdir " + mnistFolder).c_str());
//		}
//		for(unsigned int i = 0; i < 10; ++i){
//			if(!boost::filesystem::exists(mnistFolder + number2String(i))){
//				system(("mkdir " + mnistFolder + number2String(i)).c_str());
//			}
//			DataBinaryWriter::toFile(data[i], mnistFolder + number2String(i) + "/vectors.binary"); // create binary to avoid rereading .txt
//		}
		break;
	}
	case 2:{
		LabeledData data[10];
		std::ifstream input(folderLocation + "data.txt");
		if(input.is_open()){
			std::string line;
			std::getline(input, line);
			if(line.length() != 0){
				std::vector<std::string> elements;
				std::stringstream ss(line);
				std::string item;
				while(std::getline(ss, item, ' ')){
					elements.push_back(item);
				}
				if(elements[0] != "10" || elements[1] != "256"){
					printError("The size or the dimension is wrong!");
					return;
				}
			}
			while(std::getline(input, line)){
				if(line.length() != 0){
					std::vector<std::string> elements;
					elements.reserve(260);
					std::stringstream ss(line);
					std::string item;
					while(std::getline(ss, item, ' ')){
						elements.push_back(item);
					}
					if(elements.size() == 257){
						LabeledVectorX* newEle = new LabeledVectorX(256, (const unsigned int) std::stoi(elements[0]));
						for(unsigned int i = 1; i < 257; ++i){
							newEle->coeffRef(i-1) = (Real) std::stod(elements[i]);
						}
						data[newEle->getLabel()].push_back(newEle);
					}else if(elements.size() > 0 && elements[0] == "-1"){
						break;
					}else{
						printError("Something went wrong!");
					}
				}
			}
		}else{
			printError("File: " << folderLocation << "data.txt" << " could not be opened!");
			return;
		}
		input.close();
		const bool firstTime = ClassKnowledge::amountOfClasses() == 0;
		for(unsigned int i = 0; i < 10; ++i){
			if(firstTime){
				ClassKnowledge::setNameFor(number2String(i), i);
			}
			dataSets.insert( DataSetPair(number2String(i), data[i]));
		}
		if(data[0].size() == 0){
			printError("Class 0 is not represented here!");
			return;
		}
		auto dimValue = (const unsigned int) data[0][0]->rows();
		ClassKnowledge::setAmountOfDims(dimValue);

//		std::string uspsFolder = targetDir.parent_path().parent_path().c_str();
//		uspsFolder += "/usps/";
//		if(!boost::filesystem::exists(uspsFolder)){
//			system(("mkdir " + uspsFolder).c_str());
//		}
//		for(unsigned int i = 0; i < 10; ++i){
//			if(!boost::filesystem::exists(uspsFolder + number2String(i))){
//				system(("mkdir " + uspsFolder + number2String(i)).c_str());
//			}
//			if(data[i].size() > 0)
//				DataBinaryWriter::toFile(data[i], uspsFolder + number2String(i) + "/vectors.binary"); // create binary to avoid rereading .txt
//		}
		break;
	}
	case 3: case 4: case 5: {
		LabeledData data;
		for(boost::filesystem::directory_iterator itr(targetDir); itr != end_itr; ++itr){
			if(boost::filesystem::is_regular_file(itr->path()) && boost::filesystem::extension(itr->path()) == ".csv"){
				const std::string inputPath(itr->path().c_str());
				readFromFile(data, inputPath.substr(0, inputPath.length() - 4), INT_MAX, UNDEF_CLASS_LABEL, true, type == 5);
				break;
			}
		}
		printOnScreen("first: " << data[0]->transpose());
		if(data.size() > 0){
			std::map<unsigned int, unsigned int> mapFromOldToNewLabels;
			for(unsigned int i = 0; i < data.size(); ++i){
				auto it = mapFromOldToNewLabels.find(data[i]->getLabel());
				if(it != mapFromOldToNewLabels.end()){ // this class was registered before
					dataSets.find(ClassKnowledge::getNameFor(it->second))->second.push_back(data[i]);
				}else{
					const unsigned int newNumber = ClassKnowledge::amountOfClasses();
					mapFromOldToNewLabels.insert(std::pair<unsigned int, unsigned int>(data[i]->getLabel(), newNumber));
					ClassKnowledge::setNameFor(number2String(newNumber), newNumber);
					LabeledData newData;
					dataSets.insert(DataSetPair(ClassKnowledge::getNameFor(newNumber), newData));
					dataSets.find(ClassKnowledge::getNameFor(newNumber))->second.push_back(data[i]);
				}
			}
			auto dimValue = static_cast<const unsigned int>(data[0]->rows());
			ClassKnowledge::setAmountOfDims(dimValue);
		}
		break;
	}
	default:{
		printError("This type was not defined before!");
		quitApplication();
	}
	}
	unsigned int totalSize = 0;
	for(DataSetsConstIterator it = dataSets.begin(); it != dataSets.end(); ++it){
		totalSize += it->second.size();
	}
	printOnScreen("Finished Reading all Folders for " << folderLocation << ", amount of points: " << totalSize);
}

