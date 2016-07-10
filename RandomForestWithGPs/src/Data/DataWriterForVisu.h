/*
 * DataWriterForVisu.h
 *
 *  Created on: 04.06.2016
 *      Author: Max
 */

#ifndef DATA_DATAWRITERFORVISU_H_
#define DATA_DATAWRITERFORVISU_H_

#include "../RandomForests/RandomForest.h"
#include "../RandomForestGaussianProcess/RandomForestGaussianProcess.h"

class DataWriterForVisu{
public:

	static void writeData(const std::string& fileName, const Data& data, const Labels& labels, const int x = 0, const int y = 1);

	static void generateGrid(const std::string& fileName, const RandomForest& forest,
			const double amountOfPointsOnOneAxis, const Data& dataForMinMax, const int x = 0, const int y = 1);

	static void generateGrid(const std::string& fileName, const RandomForestGaussianProcess& rfgp,
			const double amountOfPointsOnOneAxis, const Data& dataForMinMax, const int x = 0, const int y = 1);

	static void generateGrid(const std::string& fileName, const GaussianProcessBinary& gp,
			const double amountOfPointsOnOneAxis, const Data& dataForMinMax, const int x = 0, const int y = 1);

private:
	DataWriterForVisu();
	virtual ~DataWriterForVisu();
};

#endif /* DATA_DATAWRITERFORVISU_H_ */
