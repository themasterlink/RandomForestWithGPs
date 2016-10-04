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

	static void toDataMatrix(const Data& data, const Labels& labels, Eigen::MatrixXd& result, Eigen::VectorXd& y, const int ele);

	static void toRandDataMatrix(const Data& data, const Labels& labels, Eigen::MatrixXd& result, Eigen::VectorXd& y, const int ele);

	static void toRandUniformDataMatrix(const Data& data, const Labels& labels, const std::vector<int>& classCounts, Eigen::MatrixXd& result, Eigen::VectorXd& y, const int ele, const int actClass);

	static void toRandClassAndHalfUniformDataMatrix(const Data& data, const Labels& labels, const std::vector<int>& classCounts, Eigen::MatrixXd& result, Eigen::VectorXd& y, const int ele, const int actClass, std::vector<bool>& usedElements, const std::vector<bool>& blockElements);

	static void getMinMax(const Data& data, double& min, double& max, const bool ignoreDBL_MAX_NEG = false);

	static void getMinMax(const Eigen::MatrixXd& mat, double& min, double& max, const bool ignoreDBL_MAX_NEG = false);

	static void getMinMax(const Eigen::VectorXd& vec, double& min, double& max, const bool ignoreDBL_MAX_NEG = false);

	static void getMinMaxIn2D(const Data& data, Eigen::Vector2d& min, Eigen::Vector2d& max, const Eigen::Vector2i& dim);

private:
	DataConverter();
	virtual ~DataConverter();
};

#endif /* DATA_DATACONVERTER_H_ */
