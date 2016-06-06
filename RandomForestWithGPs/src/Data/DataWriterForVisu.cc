/*
 * DataWriterForVisu.cc
 *
 *  Created on: 04.06.2016
 *      Author: Max
 */

#include "DataWriterForVisu.h"

DataWriterForVisu::DataWriterForVisu()
{
	// TODO Auto-generated constructor stub

}

DataWriterForVisu::~DataWriterForVisu()
{
	// TODO Auto-generated destructor stub
}

void DataWriterForVisu::writeData(const std::string& fileName, const Data& data, const Labels& labels,
		const int x, const int y){
	if(data.size() > 0){
		if(!(data[0].rows() > x && data[0].rows() > y && x != y && y >= 0 && x >= 0)){
			printError("These axis x: " << x << ", y: " << y << " aren't printable!");
			return;
		}
		std::ofstream file;
		file.open(fileName);
		if(file.is_open()){
			int i = 0;
			for(Data::const_iterator it = data.cbegin(); it != data.cend(); ++it){
				file << (*it)[x] << " " << (*it)[y] << " " << labels[i++] << "\n";
			}
		}
		file.close();
	}else{
		printError("No data -> no writting!");
	}
}

void DataWriterForVisu::generateGrid(const std::string& fileName, const RandomForest& forest,
		const double amountOfPointsOnOneAxis, const Data& data,
		const int x, const int y){
	if(data.size() == 0){
		printError("No data is given, this data is needed to find min and max!");
		return;
	}
	const int dim = data[0].rows();
	Eigen::Vector2d dimVec;
	dimVec << x,y;
	Eigen::Vector2d min, max;
	for(int i = 0; i < 2; ++i){
		min[i] = 1000000;
		max[i] = -1000000;
	}
	for(Data::const_iterator it = data.cbegin(); it != data.cend(); ++it){
		for(int i = 0; i < 2; ++i){
			int j = dimVec[i];
			if(min[i] > (*it)[j]){
				min[i] = (*it)[j];
			}
			if(max[i] < (*it)[j]){
				max[i] = (*it)[j];
			}
		}
	}
	Eigen::Vector2d stepSize = (1. / amountOfPointsOnOneAxis) * (max - min);
	std::ofstream file;
	file.open(fileName);
	Data points;
	points.reserve(amountOfPointsOnOneAxis * (amountOfPointsOnOneAxis + 1));
	int amount = 0;
	for(double xVal = max[0]; xVal >= min[0]; xVal -= stepSize[0]){
		for(double yVal = min[1]; yVal < max[1]; yVal+= stepSize[1]){
			DataElement ele(dim);
			for(int i = 0; i < dim; ++i){
				if(i == x){
					ele[i] = xVal;
				}else if(i == y){
					ele[i] = yVal;
				}else{
					ele[i] = 0;
				}
			}
			points.push_back(ele);
			++amount;
		}
	}
	Labels labels;
	forest.predictData(points, labels);
	for(int i = 0; i < amount; ++i){
		file << points[i][0] << " " << points[i][1] << " " << labels[i] << "\n";
	}
	file.close();
}
