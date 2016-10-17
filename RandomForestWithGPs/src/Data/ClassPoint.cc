/*
 * ClassPoint.cc
 *
 *  Created on: 13.10.2016
 *      Author: Max
 */

#include "ClassPoint.h"


ClassPoint::ClassPoint(): m_label(0) {

}

ClassPoint::ClassPoint(const int size, const unsigned int label):
	DataPoint(size), m_label(label) {
}

ClassPoint::ClassPoint(const int size, const double& element, const unsigned int label):
	DataPoint(DataPoint::Constant(size, element)), m_label(label) {
}

ClassPoint::~ClassPoint() {
}

void ClassPoint::setLabel(const unsigned int label){
	m_label = label;
}

unsigned int ClassPoint::getLabel() const{
	return m_label;
}
