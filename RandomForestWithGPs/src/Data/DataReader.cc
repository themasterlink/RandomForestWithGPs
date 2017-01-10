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
#include "ClassKnowledge.h"
#include "DataConverter.h"
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>

DataReader::DataReader(){
}

DataReader::~DataReader(){
}

void DataReader::readFromBinaryFile(ClassData& data, const std::string& inputName, const int amountOfData){
	std::string line;
	std::fstream input(inputName);
	if(input.is_open()){
		long size;
		input.read((char*) &size, sizeof(long));
		data.resize(size);
		for(ClassDataIterator it = data.begin(); it != data.end(); ++it){
			*it = new ClassPoint();
			ReadWriterHelper::readPoint(input, **it);
			if(!ClassKnowledge::hasClassName((*it)->getLabel())){
				ClassKnowledge::setNameFor(number2String((*it)->getLabel()), (*it)->getLabel());
			}
		}
		if(data.size() > 0 && ClassKnowledge::amountOfDims() == 0){
			ClassKnowledge::setAmountOfDims(data[0]->rows());
		}
		input.close();
	}else{
		printError("File was not found: " << inputName);
	}
}

void DataReader::readFromFile(ClassData& data, const std::string& inputName, const int amountOfData){
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
			ClassPoint* newEle = new ClassPoint(elements.size() - 1, std::stoi(elements.back()));
			for(int i = 0; i < elements.size() - 1; ++i){
				(*newEle)[i] = std::stod(elements[i]);
			};
			data.push_back(newEle);
			if(data.size() == amountOfData){
				break;
			}
		}
		if(data.size() > 0 && ClassKnowledge::amountOfDims() == 0){
			ClassKnowledge::setAmountOfDims(data[0]->rows());
		}
		input.close();
	}else{
		printError("File was not found: " << inputName);
	}
}

void DataReader::readFromFile(ClassData& data, const std::string& inputName,
		const int amountOfData, const unsigned int classNr, const bool readTxt){
	std::string inputPath(inputName);
	if(boost::filesystem::exists(inputName + ".binary") && !readTxt){
		// is a binary file -> faster loading!
		inputPath += ".binary";
		std::fstream input(inputPath, std::fstream::in);
		if(input.is_open()){
			long size;
			input.read((char*) &size, sizeof(long));
			data.resize(min(amountOfData,(int)size));
			for(long i = 0; i < min(amountOfData,(int)size); ++i){
				data[i] = new ClassPoint();
				ReadWriterHelper::readPoint(input, *data[i]);
				data[i]->setLabel(classNr);
			}
		}else{
			printError("The file could not be opened: " << inputPath);
		}
		input.close();
	}else if(boost::filesystem::exists(inputName + ".txt")){
		inputPath += ".txt";
		printOnScreen("Read txt from " + inputName);
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
				ClassPoint* newEle = new ClassPoint(elements.size(), classNr);
				for(int i = 0; i < elements.size(); ++i){
					newEle->coeffRef(i) = std::stod(elements[i]);
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
		printOnScreen("Read csv from " + inputName);
		std::ifstream input(inputPath);
		if(input.is_open()){
			std::string line;
			while(std::getline(input, line)){
				std::vector<std::string> elements;
				std::stringstream ss(line);
				std::string item;
				while(std::getline(ss, item, ',')){
					elements.push_back(item);
				}
				if(elements.size() > 0){
					const unsigned int label = std::stoi(elements.front());
					ClassPoint* newEle = new ClassPoint(elements.size() - 1, label);
					for(int i = 1; i < elements.size(); ++i){
						newEle->coeffRef(i) = std::stod(elements[i]);
					}
					data.push_back(newEle);
					if(data.size() == amountOfData){
						break;
					}
				}
			}
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
		ClassKnowledge::setAmountOfDims(data[0]->rows());
	}
}

void DataReader::readFromFiles(DataSets& dataSets, const std::string& folderLocation, const int amountOfData, const bool readTxt, bool& didNormalizeData){
	boost::filesystem::path targetDir(folderLocation);
	boost::filesystem::directory_iterator end_itr;
	// cycle through the directory
	unsigned int amountOfClasses = ClassKnowledge::amountOfClasses();
	int type = 0;
	if(targetDir.parent_path().filename() == "mnistOrg"){
		type = 1;
	}else if(targetDir.parent_path().filename() == "uspsOrg"){
		type = 2;
	}else if(targetDir.parent_path().filename() == "washington"){
		type = 3;
	}
	if(targetDir.parent_path().filename() == "mnist" && type == 0){
//		didNormalizeData = true; // did perform that before write out mnistOrg
	}
	if(type == 0){
		for(boost::filesystem::directory_iterator itr(targetDir); itr != end_itr; ++itr){
			if(boost::filesystem::is_directory(itr->path())){
				const std::string name(itr->path().filename().c_str());
				ClassData data;
				std::string filePath(itr->path().c_str());
				filePath += "/vectors";
				readFromFile(data, filePath, amountOfData, amountOfClasses, readTxt);
				ClassKnowledge::setNameFor(name, amountOfClasses);
				++amountOfClasses;
				dataSets.insert(DataSetPair(name, data));
			}
		}
	}else if(type == 1){
		ClassData data[10];
		for(boost::filesystem::directory_iterator itr(targetDir); itr != end_itr; ++itr){
			std::vector<unsigned char> labels;
			if(boost::filesystem::is_directory(itr->path())){
				const std::string inputPath(itr->path().c_str());
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
								if(labels.size() != size){
									printError("The labels should be read first!");
									getchar();
									return;
								}
								for(unsigned int i = 0; i < size; ++i){
									ClassPoint* newEle = new ClassPoint(rows * cols, labels[i]);
									for(unsigned int r = 0; r < rows; ++r){
										for(unsigned int c = 0; c < cols; ++c){
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
								labels.resize(size);
								for(unsigned int i = 0; i < size; ++i){
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
			}
		}
		if(data[0].size() == 0){
			printError("Class 0 is not represented here!");
			return;
		}
		for(unsigned int i = 0; i < 10; ++i){
			ClassKnowledge::setNameFor(number2String(i), i);
			dataSets.insert( DataSetPair(number2String(i), data[i]));
		}
//		const unsigned int amountOfDim = data[0][0]->rows();
//		std::vector<Eigen::Vector2d > minMaxValues(amountOfDim);
//		for(unsigned int k = 0; k < amountOfDim; ++k){
//			minMaxValues[k][0] = DBL_MAX;
//			minMaxValues[k][1] = NEG_DBL_MAX;
//		}
//		for(DataSetsIterator it = dataSets.begin(); it != dataSets.end(); ++it){
//			for(unsigned int t = 0; t < it->second.size(); ++t){
//				ClassPoint& point = *it->second[t];
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
//						ClassPoint& point = *it->second[t];
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
				ClassPoint& point = *it->second[t];
				for(unsigned int k = 0; k < ClassKnowledge::amountOfDims(); ++k){
					point.coeffRef(k) /= 255.;
				}
			}
		}

//		DataPoint center, var;
//		DataConverter::centerAndNormalizeData(dataSets, center, var);
		didNormalizeData = true;

		std::string mnistFolder = targetDir.parent_path().parent_path().c_str();
			mnistFolder += "/mnist/";
		if(!boost::filesystem::exists(mnistFolder)){
			system(("mkdir " + mnistFolder).c_str());
		}
		for(unsigned int i = 0; i < 10; ++i){
			if(!boost::filesystem::exists(mnistFolder + number2String(i))){
				system(("mkdir " + mnistFolder + number2String(i)).c_str());
			}
			DataBinaryWriter::toFile(data[i], mnistFolder + number2String(i) + "/vectors.binary"); // create binary to avoid rereading .txt
		}
	}else if(type == 2){
		ClassData data[10];
		for(boost::filesystem::directory_iterator itr(targetDir); itr != end_itr; ++itr){
			if(boost::filesystem::is_regular_file(itr->path())){
				const std::string inputPath(itr->path().c_str());
				const std::string fileName = itr->path().filename().c_str();
				if(boost::filesystem::extension(itr->path()) == ".txt"){
					std::ifstream input(inputPath);
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
								std::stringstream ss(line);
								std::string item;
								while(std::getline(ss, item, ' ')){
									elements.push_back(item);
								}
								if(elements.size() == 257){
									ClassPoint* newEle = new ClassPoint(256, std::stoi(elements[0]));
									for(unsigned int i = 1; i < 257; ++i){
										(*newEle)[i-1] = std::stod(elements[i]);
									}
									data[newEle->getLabel()].push_back(newEle);
								}else if(elements.size() > 0 && elements[0] == "-1"){
									break;
								}else{
									printError("Something went wrong!");
								}
							}
						}
					}
					input.close();
				}
			}
		}
		for(unsigned int i = 0; i < 10; ++i){
			ClassKnowledge::setNameFor(number2String(i), i);
			dataSets.insert( DataSetPair(number2String(i), data[i]));
		}
		if(data[0].size() == 0){
			printError("Class 0 is not represented here!");
			return;
		}
		const unsigned int dimValue = data[0][0]->rows();
		ClassKnowledge::setAmountOfDims(dimValue);

		std::string uspsFolder = targetDir.parent_path().parent_path().c_str();
		uspsFolder += "/usps/";
		if(!boost::filesystem::exists(uspsFolder)){
			system(("mkdir " + uspsFolder).c_str());
		}
		for(unsigned int i = 0; i < 10; ++i){
			if(!boost::filesystem::exists(uspsFolder + number2String(i))){
				system(("mkdir " + uspsFolder + number2String(i)).c_str());
			}
			if(data[i].size() > 0)
				DataBinaryWriter::toFile(data[i], uspsFolder + number2String(i) + "/vectors.binary"); // create binary to avoid rereading .txt
		}

	}else if(type == 3){
		ClassData data;
		for(boost::filesystem::directory_iterator itr(targetDir); itr != end_itr; ++itr){
			if(boost::filesystem::is_regular_file(itr->path()) && boost::filesystem::extension(itr->path()) == ".csv"){
				const std::string inputPath(itr->path().c_str());
				readFromFile(data, inputPath.substr(0, inputPath.length() - 4), INT_MAX, UNDEF_CLASS_LABEL, true);
			}
		}
		if(data.size() > 0){
			std::map<unsigned int, unsigned int> mapFromOldToNewLabels;
			for(unsigned int i = 0; i < data.size(); ++i){
				std::map<unsigned int, unsigned int>::iterator it = mapFromOldToNewLabels.find(data[i]->getLabel());
				if(it != mapFromOldToNewLabels.end()){ // this class was registered before
					dataSets.find(ClassKnowledge::getNameFor(it->second))->second.push_back(data[i]);
				}else{
					const int newNumber = ClassKnowledge::amountOfClasses();
					mapFromOldToNewLabels.insert(std::pair<unsigned int, unsigned int>(data[i]->getLabel(), newNumber));
					ClassKnowledge::setNameFor(number2String(newNumber), newNumber);
					ClassData newData;
					dataSets.insert(DataSetPair(ClassKnowledge::getNameFor(newNumber), newData));
					dataSets.find(ClassKnowledge::getNameFor(newNumber))->second.push_back(data[i]);
				}
			}
			const unsigned int dimValue = data[0]->rows();
			ClassKnowledge::setAmountOfDims(dimValue);
		}
	}
	printOnScreen("Finished Reading all Folders");
}

