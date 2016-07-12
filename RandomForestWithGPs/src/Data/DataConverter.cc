/*
 * DataConverter.cc
 *
 *  Created on: 06.07.2016
 *      Author: Max
 */

#include "DataConverter.h"
#include <list>

DataConverter::DataConverter()
{
	// TODO Auto-generated constructor stub

}

DataConverter::~DataConverter()
{
	// TODO Auto-generated destructor stub
}


void DataConverter::toDataMatrix(const Data& data, Eigen::MatrixXd& result, const int ele){
	const int min = ele < data.size() ? ele : data.size();
	result.conservativeResize(data[0].rows(), min);
	int i = 0;
	for(Data::const_iterator it = data.begin(); it != data.end() && i < min; ++it){
		result.col(i++) = *it;
	}
}


void DataConverter::toRandDataMatrix(const Data& data, const Labels& labels, Eigen::MatrixXd& result, Eigen::VectorXd& y, const int ele){
	if(ele == data.size()){
		toDataMatrix(data, result, ele);
		y.conservativeResize(data.size());
		for(int i = 0; i < data.size(); ++i){
			y[i] = labels[i] != 0 ? 1 : -1;
		}
		return;
	}
	const int min = ele < data.size() ? ele : data.size();
	result.conservativeResize(data[0].rows(), min);
	y.conservativeResize(min);
	std::list<int> alreadyUsed;
	for(int i = 0; i < min; ++i){
		bool usedBefore = false;
		int randEle = 0;
		do{
			usedBefore = false;
			randEle = ((double) rand() / (RAND_MAX)) * data.size();
			for(std::list<int>::const_iterator it = alreadyUsed.begin(); it != alreadyUsed.end(); ++it){
				if(*it == randEle){
					usedBefore = true;
					break;
				}
			}
		}while(usedBefore);
		alreadyUsed.push_back(randEle);
		result.col(i) = data[randEle];
		y[i] = labels[randEle] != 0 ? 1 : -1;
	}
}

void DataConverter::toRandUniformDataMatrix(const Data& data, const Labels& labels, const std::vector<int>& classCounts, Eigen::MatrixXd& result, Eigen::VectorXd& y, const int ele, const int actClass){
	if(ele >= data.size()){ // use all
		toDataMatrix(data, result, ele);
		y.conservativeResize(data.size());
		for(int i = 0; i < data.size(); ++i){
			y[i] = labels[i] == actClass ? 1 : -1;
		}
		return;
	}
	const int amountOfClasses = classCounts.size();
	result.conservativeResize(data[0].rows(), ele);
	y.conservativeResize(ele);
	std::list<int> alreadyUsed;
	std::vector<bool> useWholeClass(amountOfClasses, false);
	for(int i = 0; i < amountOfClasses; ++i){
		if(classCounts[i] < (int)(ele / (double) amountOfClasses )){
			useWholeClass[i] = true;
		}
	}
	int iResCounter = 0;
	std::vector<bool> usedBeforeEle(data.size(), false);
	// get a proper distribution in it
	for(int iActClass = 0; iActClass < amountOfClasses; ++iActClass){
		if(useWholeClass[iActClass]){ // copy whole class into result
			for(int iEle = 0; iEle < data.size(); ++iEle){
				if(labels[iEle] == iActClass){ // should copy this
					result.col(iResCounter) = data[iEle];
					y[iResCounter] = labels[iEle] == actClass ? 1 : -1;
					usedBeforeEle[iEle] = true;
					iResCounter += 1;
				}
			}
		}else{
			// make it full
			const int offset = (int)( ele / (double) amountOfClasses) + iResCounter;
			if(offset - iResCounter > classCounts[iActClass]){
				printError("Something went wrong! Class count: " << classCounts[iActClass] << ", use:" << offset - iResCounter);
			}
			for(;iResCounter < offset; ++iResCounter){
				int randEle = 0;
				bool usedBefore;
				do{
					usedBefore = false;
					randEle = ((double) rand() / (RAND_MAX)) * data.size();
					if(!usedBeforeEle[randEle] && labels[randEle] == iActClass){
						usedBeforeEle[randEle] = true;
						usedBefore = true;
						result.col(iResCounter) = data[randEle];
						y[iResCounter] = labels[randEle]  == actClass ? 1 : -1;
					}
				}while(!usedBefore);
			}
		}
	}
	// fill up with random values
	for(;iResCounter < ele; ++iResCounter){
		int randEle = 0;
		bool usedBefore;
		do{
			usedBefore = false;
			randEle = ((double) rand() / (RAND_MAX)) * data.size();
			if(!usedBeforeEle[randEle]){
				usedBeforeEle[randEle] = true;
				usedBefore = true;
				result.col(iResCounter) = data[randEle];
				y[iResCounter] = labels[randEle] == actClass ? 1 : -1;
			}
		}while(!usedBefore);
	}
}
