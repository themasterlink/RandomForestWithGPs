/*
 * ConfusionMatrixPrinter.cc
 *
 *  Created on: 23.07.2016
 *      Author: Max
 */

#include "ConfusionMatrixPrinter.h"

ConfusionMatrixPrinter::ConfusionMatrixPrinter() {
	// TODO Auto-generated constructor stub

}

ConfusionMatrixPrinter::~ConfusionMatrixPrinter() {
	// TODO Auto-generated destructor stub
}



void ConfusionMatrixPrinter::print(const Eigen::MatrixXd& conv, const std::vector<std::string>& names, std::ostream& stream){
	if(conv.rows() != names.size() || conv.cols() != names.size()){
		printError("The amount of rows or cols does not correspond to the amount of names!");
		return;
	}
	int maxLength = 0;
	for(std::vector<std::string>::const_iterator it = names.begin(); it != names.end(); ++it){
		if(it->length() > maxLength){
			maxLength = it->length();
		}
	}
	std::vector<int> maxSize(conv.cols(), 0);
	for(int i = 0; i < conv.cols(); ++i){
		maxSize[i] = names[i].length();
		for(int j = 0; j < conv.rows(); ++j){
			const int res = amountOfDigits((int)conv(i,j));
			if(res > maxSize[i]){
				maxSize[i] = res;
			}
		}
		maxSize[i] += 2;
	}
	maxLength += 2;
	// offset before the first name
	for(int i = 0; i < maxLength; ++i){
		stream << " ";
	}
	int l = 0;
	for(std::vector<std::string>::const_iterator it = names.begin(); it != names.end(); ++it){
		for(int i = it->length(); i < maxSize[l]; ++i)
			stream << " ";
		stream << *it;
		++l;
	}
	stream << "\n";
	for(int i = 0; i < conv.cols(); ++i){
		stream << names[i];
		for(int k = names[i].length(); k < maxLength; ++k)
			stream << " ";
		for(int j = 0; j < conv.rows(); ++j){
			const int covMaxSize = max(maxSize[j], (int) names[j].length());
			const int amountOfAct = amountOfDigits((int)conv(i,j));
			for(int k = 0; k < covMaxSize - amountOfAct; ++k){
				stream << " ";
			}
			stream << (int) conv(i,j);
		}
		stream << "\n";
	}
	flush(stream);
}

int ConfusionMatrixPrinter::amountOfDigits(int x) {
	if(x < 100){
		if(x < 10){
			return 1;
		}
		return 2;
	}else if(x < 1000){
		return 3;
	}
	if (x >= 10000) {
		if (x >= 10000000) {
			if (x >= 100000000) {
				if (x >= 1000000000)
					return 10;
				return 9;
			}
			return 8;
		}
		if (x >= 100000) {
			if (x >= 1000000)
				return 7;
			return 6;
		}
		return 5;
	}
	return 4;
}
