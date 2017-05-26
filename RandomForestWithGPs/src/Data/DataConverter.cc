/*
 * DataConverter.cc
 *
 *  Created on: 06.07.2016
 *      Author: Max
 */

#include "DataConverter.h"
#include "../Utility/Util.h"

DataConverter::DataConverter()
{
	// TODO Auto-generated constructor stub

}

DataConverter::~DataConverter()
{
	// TODO Auto-generated destructor stub
}

void DataConverter::centerAndNormalizeData(DataSets& datas, VectorX& center, VectorX& var){
	if(datas.size() > 0){
		const unsigned int dim = datas.begin()->second[0]->rows();
		if(center.rows() != dim && var.rows() != dim){ // calc center and var first
			center = VectorX::Zero(dim);
			VectorX counter = VectorX::Ones(dim);
			for(DataSetsIterator it = datas.begin(); it != datas.end(); ++it){
				for(LabeledDataIterator itData = it->second.begin(); itData != it->second.end(); ++itData){
					LabeledVectorX& ele = **itData;
					for(unsigned int i = 0; i < dim; ++i){
						const real fac = 1. / counter.coeff(i);
						center.coeffRef(i) = fac * ele.coeff(i) + (1. - fac) * center.coeff(i);
						++counter.coeffRef(i);
					}
				}
			}
			counter = VectorX::Ones(dim);
			var = VectorX::Zero(dim);
			for(DataSetsIterator it = datas.begin(); it != datas.end(); ++it){
				for(LabeledDataIterator itData = it->second.begin(); itData != it->second.end(); ++itData){
					LabeledVectorX& ele = **itData;
					for(unsigned int i = 0; i < dim; ++i){
						const real fac = 1. / counter.coeff(i);
						const real newVal = ele.coeff(i) - center.coeff(i);
						var.coeffRef(i) = fac * (newVal * newVal) + (1. - fac) * var.coeff(i);
						++counter.coeffRef(i);
					}
				}
			}
			for(unsigned int i = 0; i < var.rows(); ++i){
				var.coeffRef(i) = sqrt((real) var.coeff(i));
				if(var.coeff(i) <= EPSILON){
					var.coeffRef(i) = 1.; // no change
				}
			}
		}
		for(DataSetsIterator it = datas.begin(); it != datas.end(); ++it){
			for(LabeledDataIterator itData = it->second.begin(); itData != it->second.end(); ++itData){
				LabeledVectorX& ele = **itData;
				for(unsigned int i = 0; i < dim; ++i){
					ele.coeffRef(i) = (ele.coeff(i) - center.coeff(i)) / var.coeff(i);
				}
			}
		}
	}
}

void DataConverter::centerAndNormalizeData(Data& data, VectorX& center, VectorX& var){
	if(data.size() > 0){
		if(center.rows() != data[0]->rows() && var.rows() != data[0]->rows()){ // calc center and var first
			center = VectorX::Zero(data[0]->rows());
			VectorX counter = VectorX::Ones(data[0]->rows());
			for(DataIterator it = data.begin(); it != data.end(); ++it){
				VectorX& ele = *(*it);
				for(unsigned int i = 0; i < ele.rows(); ++i){
					const real fac = 1. / counter.coeff(i);
					center.coeffRef(i) = fac * ele.coeff(i) + (1. - fac) * center.coeff(i);
					++counter.coeffRef(i);
				}
			}
			counter = VectorX::Ones(data[0]->rows());
			var = VectorX::Zero(data[0]->rows());
			for(DataIterator it = data.begin(); it != data.end(); ++it){
				VectorX& ele = *(*it);
				for(unsigned int i = 0; i < ele.rows(); ++i){
					const real fac = 1. / counter.coeff(i);
					const real newVal = ele.coeff(i) - center.coeff(i);
					var.coeffRef(i) = fac * (newVal * newVal) + (1. - fac) * var.coeff(i);
					++counter.coeffRef(i);
				}
			}
			for(unsigned int i = 0; i < var.rows(); ++i){
				var.coeffRef(i) = sqrt((real) var.coeff(i));
				if(var.coeff(i) <= EPSILON){
					var.coeffRef(i) = 1.; // no change
				}
			}
		}
		for(DataIterator it = data.begin(); it != data.end(); ++it){
			VectorX& ele = *(*it);
			for(unsigned int i = 0; i < ele.rows(); ++i){
				ele.coeffRef(i) = (ele.coeff(i) - center.coeff(i)) / var.coeff(i);
			}
		}
	}
}

void DataConverter::centerAndNormalizeData(LabeledData& data, VectorX& center, VectorX& var){
	if(data.size() > 0){
		if(center.rows() != data[0]->rows() && var.rows() != data[0]->rows()){ // calc center and var first
			center = VectorX::Zero(data[0]->rows());
			VectorX counter = VectorX::Ones(data[0]->rows());
			for(LabeledDataIterator it = data.begin(); it != data.end(); ++it){
				LabeledVectorX& ele = *(*it);
				for(unsigned int i = 0; i < ele.rows(); ++i){
					const real fac = 1. / counter.coeff(i);
					center.coeffRef(i) = fac * ele.coeff(i) + (1. - fac) * center.coeff(i);
					++counter.coeffRef(i);
				}
			}
			counter = VectorX::Ones(data[0]->rows());
			var = VectorX::Zero(data[0]->rows());
			for(LabeledDataIterator it = data.begin(); it != data.end(); ++it){
				LabeledVectorX& ele = *(*it);
				for(unsigned int i = 0; i < ele.rows(); ++i){
					const real fac = 1. / counter.coeff(i);
					const real newVal = ele.coeff(i) - center.coeff(i);
					var.coeffRef(i) = fac * (newVal * newVal) + (1. - fac) * var.coeff(i);
					++counter.coeffRef(i);
				}
			}
			for(unsigned int i = 0; i < var.rows(); ++i){
				var.coeffRef(i) = sqrt((real) var.coeff(i));
				if(var.coeff(i) <= EPSILON){
					var.coeffRef(i) = 1.; // no change
				}
			}
		}
		for(LabeledDataIterator it = data.begin(); it != data.end(); ++it){
			LabeledVectorX& ele = *(*it);
			for(unsigned int i = 0; i < ele.rows(); ++i){
				ele.coeffRef(i) = (ele.coeff(i) - center.coeff(i)) / var.coeff(i);
			}
		}
	}
}

void DataConverter::toDataMatrix(const Data& data, Matrix& result, const int ele){
	const int min = ele < (int) data.size() ? ele : (int) data.size();
	result.conservativeResize(data[0]->rows(), min);
	int i = 0;
	for(DataConstIterator it = data.begin(); it != data.end() && i < min; ++it){
		result.col(i++) = *(*it);
	}
}

void DataConverter::toDataMatrix(const LabeledData& data, Matrix& result, VectorX& y, const int ele){
	const int min = ele < (int) data.size() ? ele : (int) data.size();
	result.conservativeResize((long) data[0]->rows(), min);
	y.conservativeResize(min);
	int i = 0;
	for(LabeledDataConstIterator it = data.begin(); it != data.end() && i < min; ++it, ++i){
		result.col(i) = *(*it);
		y[i] = (*it)->getLabel() == 0 ? 1 : -1;
	}
}

void DataConverter::toDataMatrix(const DataSets& datas, Matrix& result,
		VectorX& labels, Matrix& testResult, VectorX& testLabels, const int trainAmount){
	if(datas.size() <= 1){
		printError("The amount of data sets must be bigger than 1!"); return;
	}
	int eleCount = 0;
	const int classAmount = 2;
	int labelCounter = 0;
	for(DataSetsConstIterator itData = datas.begin(); itData != datas.end(); ++itData, ++labelCounter){
		if(itData->second.size() == 0){
			printError("The class \"" << itData->first << "\" has no elements!"); return;
		}
		if(classAmount == labelCounter){
			break;
		}
		eleCount += itData->second.size();
	}
	const int dim = datas.begin()->second[0]->rows();
	if(dim == 0){
		printError("The first element of the class \"" << datas.begin()->first << "\" has no values!"); return;
	}
	if(trainAmount == 0){
		printError("The train amount can not be zero!"); return;
	}
	labelCounter = 0;
	int trainCounter = 0;
	int testCounter = 0;
	result.conservativeResize(dim, trainAmount * 2);
	labels.conservativeResize(trainAmount * 2);
	if(eleCount <= trainAmount){
		printError("The train amount is to high!"); return;
	}
	testResult.conservativeResize(dim, eleCount - trainAmount);
	testLabels.conservativeResize(eleCount - trainAmount);
	for(DataSetsConstIterator itData = datas.begin(); itData != datas.end(); ++itData, ++labelCounter){
		if(classAmount == labelCounter){
			break;
		}
		const int amountOfElements = itData->second.size();
		for(int i = 0; i < amountOfElements; ++i){
			if(i < trainAmount){
				// train data
				result.col(trainCounter) = *itData->second[i];
				labels[trainCounter] = labelCounter == 0 ? 1 : -1;
				++trainCounter;
			}else{ //  if(i < (fac) * amountOfElements + 200)
				// test data
				testResult.col(testCounter) = *itData->second[i];
				testLabels[testCounter] = labelCounter == 0 ? 1 : -1;
				++testCounter;
			}
		}
	}
}

void DataConverter::toRandDataMatrix(const LabeledData& data, Matrix& result, VectorX& y, const int ele){
	if(ele == (int) data.size()){
		toDataMatrix(data, result, y, ele);
		return;
	}
	const int min = ele < (int) data.size() ? ele : (int) data.size();
	result.conservativeResize(data[0]->rows(), min);
	std::list<int> alreadyUsed;
	y.conservativeResize(min);
	for(int i = 0; i < min; ++i){
		bool usedBefore = false;
		int randEle = 0;
		do{
			usedBefore = false;
			randEle = ((real) rand() / (RAND_MAX)) * data.size();
			for(std::list<int>::const_iterator it = alreadyUsed.begin(); it != alreadyUsed.end(); ++it){
				if(*it == randEle){
					usedBefore = true;
					break;
				}
			}
		}while(usedBefore);
		alreadyUsed.push_back(randEle);
		result.col(i) = *data[randEle];
		y[i] = data[randEle]->getLabel() == 0 ? 1 : -1;
	}
}

void DataConverter::toRandUniformDataMatrix(const LabeledData& data, const std::vector<int>& classCounts,
		Matrix& result, VectorX& y, const int ele, const unsigned int actClass){
	if(ele >= (int) data.size()){ // use all
		toDataMatrix(data, result, y, ele);
		return;
	}
	const unsigned int amountOfClasses = classCounts.size();
	result.conservativeResize(data[0]->rows(), ele);
	y.conservativeResize(ele);
	std::vector<bool> useWholeClass(amountOfClasses, false);
	for(unsigned int i = 0; i < amountOfClasses; ++i){
		if(classCounts[i] < (int)(ele / (real) amountOfClasses )){
			useWholeClass[i] = true;
		}
	}
	int iResCounter = 0;
	std::vector<bool> usedBeforeEle(data.size(), false);
	// get a proper distribution in it
	for(unsigned int iActClass = 0; iActClass < amountOfClasses; ++iActClass){
		if(useWholeClass[iActClass]){ // copy whole class into result
			for(unsigned int iEle = 0; iEle < (unsigned int) data.size(); ++iEle){
				if(data[iEle]->getLabel() == iActClass){ // should copy this
					result.col(iResCounter) = *data[iEle];
					y[iResCounter] = data[iEle]->getLabel() == actClass ? 1 : -1;
					usedBeforeEle[iEle] = true;
					iResCounter += 1;
				}
			}
		}else{
			// make it full
			const int offset = (int)( ele / (real) amountOfClasses) + iResCounter;
			if(offset - iResCounter > classCounts[iActClass]){
				printError("Something went wrong! Class count: " << classCounts[iActClass] << ", use:" << offset - iResCounter);
			}
			for(;iResCounter < offset; ++iResCounter){
				int randEle = 0;
				bool usedBefore;
				do{
					usedBefore = false;
					randEle = ((real) rand() / (RAND_MAX)) * data.size();
					if(!usedBeforeEle[randEle] && data[randEle]->getLabel() == iActClass){
						usedBeforeEle[randEle] = true;
						usedBefore = true;
						result.col(iResCounter) = *data[randEle];
						y[iResCounter] = data[randEle]->getLabel()  == actClass ? 1 : -1;
					}
				}while(!usedBefore);
			}
		}
	}
	// fill up with random values
	for(;iResCounter < ele; ++iResCounter){
		int randEle = 0;
		bool usedBefore;
		do{
			usedBefore = false;
			randEle = ((real) rand() / (RAND_MAX)) * data.size();
			if(!usedBeforeEle[randEle]){
				usedBeforeEle[randEle] = true;
				usedBefore = true;
				result.col(iResCounter) = *data[randEle];
				y[iResCounter] = data[randEle]->getLabel() == actClass ? 1 : -1;
			}
		}while(!usedBefore);
	}
}

void DataConverter::toRandClassAndHalfUniformDataMatrix(const LabeledData& data,
		const std::vector<int>& classCounts, Matrix& result, VectorX& y,
		const int ele, const unsigned int actClass, std::vector<bool>& usedElements, const std::vector<bool>& blockElements){
	if(ele >= (int) data.size()){ // use all
		toDataMatrix(data, result, y, ele);
		return;
	}
	const unsigned int amountOfClasses = classCounts.size();
	result.conservativeResize(data[0]->rows(), ele);
	y.conservativeResize(ele);
	std::vector<bool> useWholeClass(amountOfClasses, false);
	const int amountForActClass = std::min(classCounts[actClass], ele / 2); // should use the half of the actclass or if
	const int amountForRestClass = ele - amountForActClass;
	useWholeClass[actClass] = classCounts[actClass] <= ele / 2;
	for(unsigned int i = 0; i < amountOfClasses; ++i){
		if(i != actClass && classCounts[i] < (int)(amountForRestClass / (real) amountOfClasses )){
			useWholeClass[i] = true;
		}
	}
	int iResCounter = 0;
	if(usedElements.size() == 0){ // init if not used before
		usedElements = std::vector<bool>(data.size(), false);
	}
	if(blockElements.size() != data.size()){
		printError("Block elements must have the same size as the data!");
		return;
	}
	//std::vector<bool> usedBeforeEle(data.size(), false);
	// get a proper distribution in it
	const unsigned int amountOfRestClasses = amountOfClasses - 1;
	for(unsigned int iActClass = 0; iActClass < amountOfClasses; ++iActClass){
		const int actualLabel = iActClass == actClass ? 1 : -1;
		if(useWholeClass[iActClass]){ // copy whole class into result
			for(unsigned int iEle = 0; iEle < data.size(); ++iEle){
				if(data[iEle]->getLabel() == iActClass && !blockElements[iEle]){ // should copy this
					result.col(iResCounter) = *data[iEle];
					y[iResCounter] = actualLabel;
					usedElements[iEle] = true;
					iResCounter += 1;
				}
			}
		}else{
			// make it full
			int offset = (int)( amountForRestClass / (real) amountOfRestClasses) + iResCounter;
			if(iActClass == actClass){
				offset = amountForActClass + iResCounter;
			}
			if(offset - iResCounter > classCounts[iActClass]){
				printError("Something went wrong! Class count: " << classCounts[iActClass] << ", use:" << offset - iResCounter);
			}
			for(;iResCounter < offset; ++iResCounter){
				int randEle = 0;
				bool usedBefore;
				do{
					usedBefore = false;
					randEle = ((real) rand() / (RAND_MAX)) * data.size();
					if(!usedElements[randEle] && data[randEle]->getLabel() == iActClass && !blockElements[randEle]){
						usedElements[randEle] = true;
						usedBefore = true;
						result.col(iResCounter) = *data[randEle];
						y[iResCounter] = actualLabel;
					}
				}while(!usedBefore);
			}
		}
	}
	// fill up with random values
	for(;iResCounter < ele; ++iResCounter){
		int randEle = 0;
		bool usedBefore;
		do{
			usedBefore = false;
			randEle = ((real) rand() / (RAND_MAX)) * data.size();
			if(!usedElements[randEle] && !blockElements[randEle]){
				usedElements[randEle] = true;
				usedBefore = true;
				result.col(iResCounter) = *data[randEle];
				y[iResCounter] = data[randEle]->getLabel() == actClass ? 1 : -1;
			}
		}while(!usedBefore);
	}
}

void DataConverter::getMinMax(const Data& data, real& min, real& max, const bool ignoreREAL_MAX_NEG){
	min = REAL_MAX; max = NEG_REAL_MAX;
	if(ignoreREAL_MAX_NEG){
		for(DataConstIterator it = data.begin(); it != data.end(); ++it){
			VectorX& ele = **it;
			for(unsigned int i = 0; i < ele.rows(); ++i){
				const real val = ele[i];
				if(val < min && val > NEG_REAL_MAX){
					min = val;
				}
				if(val > max){
					max = val;
				}
			}
		}
	}else{
		for(DataConstIterator it = data.begin(); it != data.end(); ++it){
			VectorX& ele = **it;
			for(unsigned int i = 0; i < ele.rows(); ++i){
				const real val = ele[i];
				if(val < min){
					min = val;
				}
				if(val > max){
					max = val;
				}
			}
		}
	}
}

void DataConverter::getMinMax(const Matrix& mat, real& min, real& max, const bool ignoreREAL_MAX_NEG){
	min = REAL_MAX; max = NEG_REAL_MAX;
	if(ignoreREAL_MAX_NEG){
		for(unsigned int i = 0; i < mat.rows(); ++i){
			for(unsigned int j = 0; j < mat.cols(); ++j){
				const real ele = mat(i,j);
				if(ele < min && ele > NEG_REAL_MAX){
					min = ele;
				}
				if(ele > max){
					max = ele;
				}
			}
		}
	}else{
		for(unsigned int i = 0; i < mat.rows(); ++i){
			for(unsigned int j = 0; j < mat.cols(); ++j){
				const real ele = mat(i,j);
				if(ele < min){
					min = ele;
				}
				if(ele > max){
					max = ele;
				}
			}
		}
	}
}

void DataConverter::getMinMax(const VectorX& vec, real& min, real& max, const bool ignoreREAL_MAX_NEG){
	min = REAL_MAX; max = NEG_REAL_MAX;
	if(ignoreREAL_MAX_NEG){
		for(unsigned int i = 0; i < vec.rows(); ++i){
			const real ele = vec[i];
			if(ele < min && ele > NEG_REAL_MAX){
				min = ele;
			}
			if(ele > max){
				max = ele;
			}
		}
	}else{
		for(unsigned int i = 0; i < vec.rows(); ++i){
			const real ele = vec[i];
			if(ele < min){
				min = ele;
			}
			if(ele > max){
				max = ele;
			}
		}
	}
}

void DataConverter::getMinMax(const std::list<real>& list, real& min, real& max, const bool ignoreREAL_MAX_NEG){
	min = REAL_MAX; max = NEG_REAL_MAX;
	if(ignoreREAL_MAX_NEG){
		for(std::list<real>::const_iterator it = list.begin(); it != list.end(); ++it){
			const real ele = *it;
			if(ele < min && ele > NEG_REAL_MAX){
				min = ele;
			}
			if(ele > max){
				max = ele;
			}
		}
	}else{
		for(std::list<real>::const_iterator it = list.begin(); it != list.end(); ++it){
			const real ele = *it;
			if(ele < min){
				min = ele;
			}
			if(ele > max){
				max = ele;
			}
		}
	}
}

void DataConverter::getMinMaxIn2D(const std::list<Vector2>& list, Vector2& min, Vector2& max, const bool ignoreREAL_MAX_NEG){
	min[0] = min[1] =  REAL_MAX;
	max[0] = max[1] = NEG_REAL_MAX;
	for(std::list<Vector2>::const_iterator it = list.cbegin(); it != list.cend(); ++it){
		for(unsigned int i = 0; i < 2; ++i){
			if(ignoreREAL_MAX_NEG){
				if((*it)[i] < min[i] && (*it)[i] > NEG_REAL_MAX){
					min[i] = (*it)[i];
				}
			}else{
				if((*it)[i] < min[i]){
					min[i] = (*it)[i];
				}
			}
			if((*it)[i] > max[i]){
				max[i] = (*it)[i];
			}
		}
	}
}

void DataConverter::getMinMaxIn2D(const LabeledData& data, Vector2& min, Vector2& max, const Vector2i& dim){
	min[0] = min[1] =  REAL_MAX;
	max[0] = max[1] = NEG_REAL_MAX;
	if(data.size() > 0){
		if(dim[0] < data[0]->rows() && dim[1] < data[0]->rows()){
			for(LabeledDataConstIterator it = data.begin(); it != data.end(); ++it){
				const real first = (**it)[dim[0]];
				const real second = (**it)[dim[1]];
				if(first < min[0]){
					min[0] = first;
				}
				if(first > max[0]){
					max[0] = first;
				}
				if(second < min[1]){
					min[1] = second;
				}
				if(second > max[1]){
					max[1] = second;
				}
			}
		}
	}
}

void DataConverter::getMinMaxIn2D(const Data& data, Vector2& min, Vector2& max, const Vector2i& dim){
	min[0] = min[1] =  REAL_MAX;
	max[0] = max[1] = NEG_REAL_MAX;
	if(data.size() > 0){
		if(dim[0] < data[0]->rows() && dim[1] < data[0]->rows()){
			for(DataConstIterator it = data.begin(); it != data.end(); ++it){
				const real first = (**it)[dim[0]];
				const real second = (**it)[dim[1]];
				if(first < min[0]){
					min[0] = first;
				}
				if(first > max[0]){
					max[0] = first;
				}
				if(second < min[1]){
					min[1] = second;
				}
				if(second > max[1]){
					max[1] = second;
				}
			}
		}
	}
}

void DataConverter::setToData(const DataSets& set, LabeledData& data){
	long size = 0;
	for(DataSetsConstIterator it = set.begin(); it != set.end(); ++it){
		size += it->second.size();
	}
	data.resize(size);
	int i = 0;
	for(DataSetsConstIterator it = set.begin(); it != set.end(); ++it){
		for(LabeledDataConstIterator itEle = it->second.begin(); itEle != it->second.end(); ++itEle){
			data[i] = *itEle;
			++i;
		}
	}
}

