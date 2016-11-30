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
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>

DataReader::DataReader(){
}

DataReader::~DataReader(){
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
			ClassPoint* newEle = new ClassPoint(elements.size() - 1, std::stoi(elements.back()) > 0 ? 1 : 0);
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
		printOnScreen("Read txt from " + inputName);
		inputPath += ".txt";
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
					(*newEle)[i] = std::stod(elements[i]);
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
	}else{
		printError("File was not found for .txt or .binary: " << inputName);
	}
	if(data.size() > 0 && ClassKnowledge::amountOfDims() == 0){
		ClassKnowledge::setAmountOfDims(data[0]->rows());
	}
}

void DataReader::readFromFiles(DataSets& dataSets, const std::string& folderLocation, const int amountOfData, const bool readTxt){
	boost::filesystem::path targetDir(folderLocation);
	boost::filesystem::directory_iterator end_itr;
	// cycle through the directory
	unsigned int amountOfClasses = ClassKnowledge::amountOfClasses();
	int type = 0;
	if(targetDir.parent_path().filename() == "mnistOrg"){
		type = 1;
	}else if(targetDir.parent_path().filename() == "uspsOrg"){
		type = 2;
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
				dataSets.insert( DataSetPair(name, data));
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
											(*newEle)[r * rows + c] = ele;
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
					for(unsigned int i = 0; i < 10; ++i){
						ClassKnowledge::setNameFor(number2String(i), i);
						dataSets.insert( DataSetPair(number2String(i), data[i]));
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
		const unsigned int dimValue = data[0][0]->rows();
		std::vector<Eigen::Vector2d > minMaxValues(dimValue);
		for(unsigned int k = 0; k < dimValue; ++k){
			minMaxValues[k][0] = DBL_MAX;
			minMaxValues[k][1] = -DBL_MAX;
		}
		for(DataSetsIterator it = dataSets.begin(); it != dataSets.end(); ++it){
			for(unsigned int t = 0; t < it->second.size(); ++t){
				ClassPoint& point = *it->second[t];
				for(unsigned int k = 0; k < dimValue; ++k){
					if(point[k] < minMaxValues[k][0]){
						minMaxValues[k][0] = point[k];
					}
					if(point[k] > minMaxValues[k][1]){
						minMaxValues[k][1] = point[k];
					}
				}
			}
		}
		int newDim = dimValue;
		bool diffIsEqual = false;
		int lastDim = -1;
		std::vector<bool> notUsed(dimValue, false);
		for(int k = dimValue - 1; k >= 0; --k){
			if(minMaxValues[k][0] == minMaxValues[k][1] && !diffIsEqual){
				lastDim = k;
				diffIsEqual = true;
			}else if(minMaxValues[k][0] != minMaxValues[k][1] && diffIsEqual){
				// remove this dimension in every pixel
				int diff = lastDim - k;
				for(unsigned int i = k; i < lastDim; ++i){
					notUsed[i] = true;
				}
				printOnScreen("Remove dimension from exclusive " << k << " to " << lastDim << " from all points");
				++k; // get to last value
				newDim -= diff;
				for(DataSetsIterator it = dataSets.begin(); it != dataSets.end(); ++it){
					for(unsigned int t = 0; t < it->second.size(); ++t){
						ClassPoint& point = *it->second[t];
						if(k < dimValue - 1){ // not if the last element is removed
							for(unsigned int t = k; t < newDim; ++t){
								point[t] = point[t+diff];
							}
						}
						point.resize(newDim);
					}
				}
				diffIsEqual = false;
				--k; // get back last value
			}
		}
		cv::Mat img(28, 28, CV_8UC3, cv::Scalar(0, 0, 0));
		for(unsigned int r = 0; r < 28; ++r){
			for(unsigned int c = 0; c < 28; ++c){
				cv::Vec3b& color = img.at<cv::Vec3b>(r,c);
				color[0] = (notUsed[r * 28 + c] ? 1 : 0) * 255;
				color[1] = (notUsed[r * 28 + c] ? 1 : 0) * 255;
				color[2] = (notUsed[r * 28 + c] ? 1 : 0) * 255;
			}
		}
		cv::imwrite("test.png", img);
		openFileInViewer("test.png");
		ClassKnowledge::setAmountOfDims(newDim);

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

	}
	printOnScreen("Finished Reading all Folders");
	sleep(1);
}

