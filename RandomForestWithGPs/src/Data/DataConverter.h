/*
 * DataConverter.h
 *
 *  Created on: 06.07.2016
 *      Author: Max
 */

#ifndef DATA_DATACONVERTER_H_
#define DATA_DATACONVERTER_H_

#include "Data.h"
#include "DataPoint.h"
#include "ClassData.h"
#include "DataSets.h"

class DataConverter{
public:
	static void centerAndNormalizeData(Data& data, DataPoint& center, DataPoint& var);

	static void centerAndNormalizeData(ClassData& data, DataPoint& center, DataPoint& var);

	static void centerAndNormalizeData(DataSets& data, DataPoint& center, DataPoint& var);

	static void toDataMatrix(const Data& data, Eigen::MatrixXd& result, const int ele);

	static void toDataMatrix(const ClassData& data, Eigen::MatrixXd& result, Eigen::VectorXd& y, const int ele);

	static void toDataMatrix(const DataSets& datas, Eigen::MatrixXd& result,
			Eigen::VectorXd& labels, Eigen::MatrixXd& testResult, Eigen::VectorXd& testLabels, const int trainAmount);

	static void toRandDataMatrix(const ClassData& data, Eigen::MatrixXd& result, Eigen::VectorXd& y, const int ele);

	static void toRandUniformDataMatrix(const ClassData& data, const std::vector<int>& classCounts, Eigen::MatrixXd& result,
			Eigen::VectorXd& y, const int ele, const unsigned int actClass);

	static void toRandClassAndHalfUniformDataMatrix(const ClassData& data, const std::vector<int>& classCounts, Eigen::MatrixXd& result,
			Eigen::VectorXd& y, const int ele, const unsigned int actClass, std::vector<bool>& usedElements, const std::vector<bool>& blockElements);

	static void getMinMax(const Data& data, double& min, double& max, const bool ignoreDBL_MAX_NEG = false);

	static void getMinMax(const Eigen::MatrixXd& mat, double& min, double& max, const bool ignoreDBL_MAX_NEG = false);

	static void getMinMax(const Eigen::VectorXd& vec, double& min, double& max, const bool ignoreDBL_MAX_NEG = false);

	static void getMinMax(const std::list<double>& list, double& min, double& max, const bool ignoreDBL_MAX_NEG = false);

	static void getMinMaxIn2D(const std::list<Eigen::Vector2d>& list, Eigen::Vector2d& min, Eigen::Vector2d& max, const bool ignoreDBL_MAX_NEG = false);

	static void getMinMaxIn2D(const Data& data, Eigen::Vector2d& min, Eigen::Vector2d& max, const Eigen::Vector2i& dim);

	static void getMinMaxIn2D(const ClassData& data, Eigen::Vector2d& min, Eigen::Vector2d& max, const Eigen::Vector2i& dim);

	static void setToData(const DataSets& set, ClassData& data);

private:
	DataConverter();
	virtual ~DataConverter();
};

#endif /* DATA_DATACONVERTER_H_ */
