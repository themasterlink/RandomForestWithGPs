/*
 * LabeledVectorX.h
 *
 *  Created on: 13.10.2016
 *      Author: Max
 */

#ifndef DATA_LABELEDVECTOR_H_
#define DATA_LABELEDVECTOR_H_

#include "../Base/Types.h"

class LabeledVectorX : public VectorX {
public:
	LabeledVectorX();

	LabeledVectorX(const int size, const unsigned int label);

	LabeledVectorX(const int size, const float& element, const unsigned int label);

	virtual ~LabeledVectorX();

	void setLabel(const unsigned int label);

	unsigned int getLabel() const noexcept;

private:
	unsigned int m_label;
};

inline
void LabeledVectorX::setLabel(const unsigned int label){
	m_label = label;
}

inline
unsigned int LabeledVectorX::getLabel() const noexcept{
	return m_label;
}

using LabeledData = std::vector<LabeledVectorX*>;

using LabeledDataIterator = LabeledData::iterator;

using LabeledDataConstIterator = LabeledData::const_iterator;

using DataSets = std::map< std::string, LabeledData>;

using DataSetsIterator = DataSets::iterator;

using DataSetsConstIterator = DataSets::const_iterator;

using DataSetPair = std::pair<std::string, LabeledData >;

#endif /* DATA_LABELEDVECTOR_H_ */
