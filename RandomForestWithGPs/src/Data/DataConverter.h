/*
 * DataConverter.h
 *
 *  Created on: 06.07.2016
 *      Author: Max
 */

#ifndef DATA_DATACONVERTER_H_
#define DATA_DATACONVERTER_H_

#include "Data.h"

class DataConverter{
public:
	static void toDataMatrix(const Data& data, Eigen::MatrixXd& result, const int ele);

	static void toRandDataMatrix(const Data& data, const Labels& labels, Eigen::MatrixXd& result, Eigen::VectorXd& y, const int ele);

	static void toRandUniformDataMatrix(const Data& data, const Labels& labels, const std::vector<int>& classCounts, Eigen::MatrixXd& result, Eigen::VectorXd& y, const int ele, const int actClass);


private:
	DataConverter();
	virtual ~DataConverter();
};

#endif /* DATA_DATACONVERTER_H_ */
