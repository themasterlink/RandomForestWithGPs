/*
 * DataSet.h
 *
 *  Created on: 31.05.2016
 *      Author: Max
 */

#ifndef RANDOMFORESTS_DATA_H_
#define RANDOMFORESTS_DATA_H_

#include <Eigen/Dense>
#include "../Utility/Util.h"
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

typedef Eigen::VectorXd DataElement;
typedef std::vector<DataElement> Data;
typedef std::vector<Eigen::VectorXd> ComplexLabels; // could be that the data elements have continous labels
typedef std::vector<int> SimpleLabels; // could be that the data elements have continous labels
typedef SimpleLabels Labels;

class DataReader{

public:
	static void readTrainingFromFile(Data& data, Labels& label, const std::string& inputName){
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
				DataElement newEle(elements.size() - 1);
				for(int i = 0; i < elements.size() - 1; ++i){
					newEle[i] = std::stod(elements[i]);
				}
				label.push_back(std::stoi(elements.back()) > 0 ? 1 : 0);
				data.push_back(newEle);
			}
			input.close();
		}else{
			printError("File was not found: " << inputName);
		}
	}

	static void readTestFromFile(Data& data, Labels& label, const std::string& inputName){
		std::string line;
		std::ifstream input(inputName);
		if(input.is_open()){
			bool stillData = true;
			while(std::getline(input, line)){
				if(line.length() < 2){
					stillData = false;
					continue;
				}
				if(stillData){
					std::vector<std::string> elements;
					std::stringstream ss(line);
					std::string item;
					while(std::getline(ss, item, ',')){
						elements.push_back(item);
					}
					DataElement newEle(elements.size());
					for(int i = 0; i < elements.size(); ++i){
						newEle[i] = std::stod(elements[i]);
					}
					data.push_back(newEle);
				}else{
					label.push_back(std::stoi(line) > 0 ? 1 : 0);
				}
			}
			input.close();
		}else{
			printError("File was not found: " << inputName);
		}
	}

private:
	DataReader();
	~DataReader();
};

#endif /* RANDOMFORESTS_DATA_H_ */
