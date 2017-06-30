/*
 * ConfusionMatrixPrinter.cc
 *
 *  Created on: 23.07.2016
 *      Author: Max
 */

#include "ConfusionMatrixPrinter.h"
#include "../Data/ClassKnowledge.h"

ConfusionMatrixPrinter::ConfusionMatrixPrinter() {
	// TODO Auto-generated constructor stub

}

ConfusionMatrixPrinter::~ConfusionMatrixPrinter() {
	// TODO Auto-generated destructor stub
}



void ConfusionMatrixPrinter::print(const Matrix& conv){
	if(conv.rows() != ClassKnowledge::instance().amountOfClasses() ||
	   conv.cols() != ClassKnowledge::instance().amountOfClasses()){
		printError("The amount of rows or cols does not correspond to the amount of names: ("
						   << conv.rows() << "," << conv.cols() << ") != "
						   << ClassKnowledge::instance().amountOfClasses());
		return;
	}
	int maxLength = 0;
	for(unsigned int i = 0; i < ClassKnowledge::instance().amountOfClasses(); ++i){
		if((int) ClassKnowledge::instance().getNameFor(i).length() > maxLength){
			maxLength = ClassKnowledge::instance().getNameFor(i).length();
		}
	}
	std::vector<int> maxSize(conv.cols(), 0);
	for(int i = 0; i < conv.cols(); ++i){
		maxSize[i] = ClassKnowledge::instance().getNameFor(i).length();
		for(int j = 0; j < conv.rows(); ++j){
			const int res = amountOfDigits((int)conv.coeff(j,i));
			if(res > maxSize[i]){
				maxSize[i] = res;
			}
		}
		maxSize[i] += 2;
	}
	maxLength += 2;
	std::stringstream stream;
	// offset before the first name
	for(int i = 0; i < maxLength; ++i){
		stream << " ";
	}
	int l = 0;
	for(unsigned int t = 0; t < ClassKnowledge::instance().amountOfClasses(); ++t){
		for(int i = ClassKnowledge::instance().getNameFor(t).length(); i < maxSize[l]; ++i)
			stream << " ";
		stream << ClassKnowledge::instance().getNameFor(t);
		++l;
	}
	ScreenOutput::instance().print(stream.str());
		for(int j = 0; j < conv.rows(); ++j){
		std::stringstream stream2;
			stream2 << ClassKnowledge::instance().getNameFor(j);
			for(int k = ClassKnowledge::instance().getNameFor(j).length(); k < maxLength; ++k)
			stream2 << " ";
		for(int i = 0; i < conv.cols(); ++i){
			const int covMaxSize = maxSize[i];
			const int amountOfAct = amountOfDigits((int)conv.coeff(j,i));
			for(int k = 0; k < covMaxSize - amountOfAct; ++k){
				stream2 << " ";
			}
			stream2 << (int) conv.coeff(j,i);
		}
			ScreenOutput::instance().print(stream2.str());
	}
}

int ConfusionMatrixPrinter::amountOfDigits(int x) {
	const int minus = x < 0 ? 1 : 0;
	x = abs(x);
	if(x < 100){
		if(x < 10){
			return 1 + minus;
		}
		return 2 + minus;
	}else if(x < 1000){
		return 3 + minus;
	}
	if (x >= 10000) {
		if (x >= 10000000) {
			if (x >= 100000000) {
				if (x >= 1000000000)
					return 10 + minus;
				return 9 + minus;
			}
			return 8 + minus;
		}
		if (x >= 100000) {
			if (x >= 1000000)
				return 7 + minus;
			return 6 + minus;
		}
		return 5 + minus;
	}
	return 4 + minus;
}
