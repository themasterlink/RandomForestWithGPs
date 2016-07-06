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
	static void toDataMatrix(const Data& data, Eigen::MatrixXd& result);

private:
	DataConverter();
	virtual ~DataConverter();
};

#endif /* DATA_DATACONVERTER_H_ */
