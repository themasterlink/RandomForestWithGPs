/*
 * ClassPoint.h
 *
 *  Created on: 13.10.2016
 *      Author: Max
 */

#ifndef DATA_CLASSPOINT_H_
#define DATA_CLASSPOINT_H_

#include "DataPoint.h"

class ClassPoint : public DataPoint {
public:
	ClassPoint();

	ClassPoint(const int size, const unsigned int label);

	ClassPoint(const int size, const double& element, const unsigned int label);

	virtual ~ClassPoint();

	void setLabel(const unsigned int label);

	unsigned int getLabel() const;

private:
	unsigned int m_label;
};

#endif /* DATA_CLASSPOINT_H_ */
