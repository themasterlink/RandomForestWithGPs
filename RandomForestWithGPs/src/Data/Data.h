/*
 * Data.h
 *
 *  Created on: 04.06.2016
 *      Author: Max
 */

#ifndef DATA_DATA_H_
#define DATA_DATA_H_

#include <Eigen/Dense>
#include "../Utility/Util.h"
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "DataPoint.h"

static const auto UNDEF_CLASS_LABEL = 99999999u;

using Data = std::vector<DataPoint*>;

using DataIterator = Data::iterator;

using DataConstIterator = Data::const_iterator;

using Labels = std::vector<unsigned int>;

using DiagMatrixXd = Eigen::DiagonalWrapper<const Eigen::MatrixXd> ;

/*typedef Eigen::VectorXd DataElement;
typedef std::vector<DataElement> Data;
typedef std::map<std::string, Data > DataSets;
typedef std::vector<Eigen::VectorXd> ComplexLabels; // could be that the data elements have continous labels
typedef std::vector<int> SimpleLabels; // could be that the data elements have continous labels
typedef SimpleLabels Labels;
typedef std::vector<double> DoubleLabels; // could be that the data elements have continous labels
*/


#endif /* DATA_DATA_H_ */
