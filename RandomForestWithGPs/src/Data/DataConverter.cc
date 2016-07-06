/*
 * DataConverter.cc
 *
 *  Created on: 06.07.2016
 *      Author: Max
 */

#include "DataConverter.h"

DataConverter::DataConverter()
{
	// TODO Auto-generated constructor stub

}

DataConverter::~DataConverter()
{
	// TODO Auto-generated destructor stub
}


void DataConverter::toDataMatrix(const Data& data, Eigen::MatrixXd& result){
	result.conservativeResize(data[0].rows(), data.size());
	int i = 0;
	for(Data::const_iterator it = data.begin(); it != data.end(); ++it){
		result.col(i++) = *it;
	}
}

