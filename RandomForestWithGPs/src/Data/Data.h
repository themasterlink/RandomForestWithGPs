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

#define UNDEF_CLASS_LABEL 99999999

typedef typename std::vector<DataPoint*> Data;

typedef typename Data::iterator DataIterator;

typedef typename Data::const_iterator DataConstIterator;

typedef typename std::vector<unsigned int> Labels;

typedef typename Eigen::DiagonalWrapper<const Eigen::MatrixXd> DiagMatrixXd;

/*typedef Eigen::VectorXd DataElement;
typedef std::vector<DataElement> Data;
typedef std::map<std::string, Data > DataSets;
typedef std::vector<Eigen::VectorXd> ComplexLabels; // could be that the data elements have continous labels
typedef std::vector<int> SimpleLabels; // could be that the data elements have continous labels
typedef SimpleLabels Labels;
typedef std::vector<double> DoubleLabels; // could be that the data elements have continous labels
*/


#endif /* DATA_DATA_H_ */
