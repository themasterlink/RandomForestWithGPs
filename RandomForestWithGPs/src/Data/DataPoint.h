/*
 * DataPoint.h
 *
 *  Created on: 13.10.2016
 *      Author: Max
 */

#ifndef DATA_DATAPOINT_H_
#define DATA_DATAPOINT_H_

#include <Eigen/Dense>

typedef typename Eigen::VectorXd DataPoint;
/*
class DataPoint : public Eigen::VectorXd {
public:
	DataPoint();

	DataPoint(const int size);

	DataPoint(const int size, const double& element);

	DataPoint& operator=(const DataPoint& point);

	virtual ~DataPoint();
};*/

#endif /* DATA_DATAPOINT_H_ */
