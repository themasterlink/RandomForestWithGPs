/*
 * DataConverter.h
 *
 *  Created on: 06.07.2016
 *      Author: Max
 */

#ifndef DATA_DATACONVERTER_H_
#define DATA_DATACONVERTER_H_

#include "LabeledVectorX.h"

class DataConverter{
public:
	static void centerAndNormalizeData(Data& data, VectorX& center, VectorX& var);

	static void centerAndNormalizeData(LabeledData& data, VectorX& center, VectorX& var);

	static void centerAndNormalizeData(DataSets& data, VectorX& center, VectorX& var);

	static void toDataMatrix(const Data& data, Matrix& result, const int ele);

	static void toDataMatrix(const LabeledData& data, Matrix& result, VectorX& y, const int ele);

	static void toDataMatrix(const DataSets& datas, Matrix& result,
			VectorX& labels, Matrix& testResult, VectorX& testLabels, const int trainAmount);

	static void toRandDataMatrix(const LabeledData& data, Matrix& result, VectorX& y, const int ele);

	static void toRandUniformDataMatrix(const LabeledData& data, const std::vector<int>& classCounts, Matrix& result,
			VectorX& y, const int ele, const unsigned int actClass);

	static void toRandClassAndHalfUniformDataMatrix(const LabeledData& data, const std::vector<int>& classCounts, Matrix& result,
			VectorX& y, const int ele, const unsigned int actClass, std::vector<bool>& usedElements, const std::vector<bool>& blockElements);

	static void getMinMax(const Data& data, Real& min, Real& max, const bool ignoreREAL_MAX_NEG = false);

	static void getMinMax(const Matrix& mat, Real& min, Real& max, const bool ignoreREAL_MAX_NEG = false);

	static void getMinMax(const VectorX& vec, Real& min, Real& max, const bool ignoreREAL_MAX_NEG = false);

	static void getMinMax(const std::list<Real>& list, Real& min, Real& max, const bool ignoreREAL_MAX_NEG = false);

	static void getMinMaxIn2D(const std::list<Vector2>& list, Vector2& min, Vector2& max, const bool ignoreDBL_MAX_NEG = false);

	static void getMinMaxIn2D(const Data& data, Vector2& min, Vector2& max, const Vector2i& dim);

	static void getMinMaxIn2D(const LabeledData& data, Vector2& min, Vector2& max, const Vector2i& dim);

	static void setToData(const DataSets& set, LabeledData& data);

private:
	DataConverter();
	virtual ~DataConverter();
};

#endif /* DATA_DATACONVERTER_H_ */
