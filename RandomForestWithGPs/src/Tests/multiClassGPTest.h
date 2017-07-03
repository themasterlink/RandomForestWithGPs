/*
 * multiClassGPTest.h
 *
 *  Created on: 15.09.2016
 *      Author: Max
 */

#ifndef TESTS_MULTICLASSGPTEST_H_
#define TESTS_MULTICLASSGPTEST_H_

#include "../Data/DataReader.h"
#include "../GaussianProcess/GaussianProcessMultiClass.h"

#ifdef BUILD_OLD_CODE

void executeForMultiClass(const std::string& path){

	LabeledData data;
	DataReader::readFromFile(data, path, 500);

	const unsigned int dataPoints = data.size();
	Matrix dataMat;

	dataMat.conservativeResize(data[0]->rows(), data.size());
	int i = 0;
	for(LabeledDataIterator it = data.begin(); it != data.end(); ++it, ++i){
		dataMat.col(i) = **it;
	}
	const int amountOfClass = 2;
	/*std::vector<Data> dataPerClass(amountOfClass);
	for(int i = 0; i < data.size(); ++i){
		dataPerClass[labels[i]].push_back(data[i]);
	}*/
	VectorX y(VectorX(dataPoints * amountOfClass));
	for(unsigned int i = 0; i < data.size(); ++i){
		y[data[i]->getLabel() * dataPoints + i] = 1;
	}
	std::fstream f("t.txt", std::ios::out);
	//f << "dataMat:\n" << dataMat << std::endl;
	std::vector<Matrix> cov;

	Matrix covariance;
	GaussianKernel kernel;
	kernel.init(dataMat, true, false);
	kernel.calcCovariance(covariance);
	f << "covariance: \n" << covariance << std::endl;
	f << "labels: \n";
	for(unsigned int i = 0; i < data.size(); ++i){
		f << "           " << data[i]->getLabel();
	}
	f << std::endl;
	for(unsigned int i = 0; i < amountOfClass; ++i){ // calc the covariance matrix for each f_c
		Matrix cov_c = covariance; //  * y.segment(i*dataPoints, dataPoints).transpose();
		for(unsigned int j = 0; j < dataPoints; ++j){
			if(i == data[j]->getLabel()){
				for(unsigned int k = 0; k < dataPoints; ++k){
					cov_c(j,k) = 0;
					cov_c(k,j) = 0;
				}
			}
		}
		f << "cov_c: \n" << cov_c << std::endl;
		cov.push_back(cov_c);
	}

	f << std::endl;
	f << std::endl;
	f << std::endl;
	//f << cov << std::endl;
	f << y.transpose() << std::endl;
	f.close();
	GaussianProcessMultiClass::magicFunc(amountOfClass,dataPoints, cov, y);
}

#endif // BUILD_OLD_CODE

#endif /* TESTS_MULTICLASSGPTEST_H_ */
