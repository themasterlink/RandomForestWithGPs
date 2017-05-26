/*
 * LabeledVectorX.cc
 *
 *  Created on: 13.10.2016
 *      Author: Max
 */

#include "LabeledVectorX.h"
#include "../Utility/Util.h"

LabeledVectorX::LabeledVectorX(): m_label(UNDEF_CLASS_LABEL) {
}

LabeledVectorX::LabeledVectorX(const int size, const unsigned int label):
	VectorX(size), m_label(label) {
}

LabeledVectorX::LabeledVectorX(const int size, const float& element, const unsigned int label):
	VectorX(VectorX::Constant(VectorX::Index(size), element)), m_label(label) {
}

LabeledVectorX::~LabeledVectorX() = default;

