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
