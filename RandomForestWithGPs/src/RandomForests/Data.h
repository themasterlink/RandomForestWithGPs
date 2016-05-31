/*
 * DataSet.h
 *
 *  Created on: 31.05.2016
 *      Author: Max
 */

#ifndef RANDOMFORESTS_DATA_H_
#define RANDOMFORESTS_DATA_H_

#include <Eigen/Dense>
#include <vector>

typedef std::vector<Eigen::VectorXd> Data;
typedef std::vector<Eigen::VectorXd> Labels; // could be that the data elements have continous labels

#endif /* RANDOMFORESTS_DATA_H_ */
