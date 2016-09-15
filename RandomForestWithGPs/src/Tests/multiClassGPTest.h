/*
 * multiClassGPTest.h
 *
 *  Created on: 15.09.2016
 *      Author: Max
 */

#ifndef TESTS_MULTICLASSGPTEST_H_
#define TESTS_MULTICLASSGPTEST_H_

#include <Eigen/Dense>
#include "../Data/DataReader.h"
#include "../GaussianProcess/GaussianProcessMultiClass.h"
#include "../GaussianProcess/Kernel.h"

void executeForMultiClass(const std::string& path){

	Data data;
	Labels labels;
	DataReader::readFromFile(data, labels, path);

	const int dataPoints = data.size();
	Eigen::MatrixXd dataMat;

	dataMat.conservativeResize(data[0].rows(), data.size());
	int i = 0;
	for(Data::iterator it = data.begin(); it != data.end(); ++it){
		dataMat.col(i++) = *it;
	}
	const int amountOfClass = 2;
	/*std::vector<Data> dataPerClass(amountOfClass);
	for(int i = 0; i < data.size(); ++i){
		dataPerClass[labels[i]].push_back(data[i]);
	}*/
	Eigen::VectorXd y(Eigen::VectorXd(dataPoints * amountOfClass));
	for(int i = 0; i < labels.size(); ++i){
		y[labels[i] * dataPoints + i] = 1;
	}
	std::fstream f("t.txt", std::ios::out);
	//f << "dataMat:\n" << dataMat << std::endl;
	std::vector<Eigen::MatrixXd> cov;

	Eigen::MatrixXd covariance;
	Kernel kernel;
	kernel.init(dataMat);
	kernel.calcCovariance(covariance);
	f << "covariance: \n" << covariance << std::endl;
	f << "labels: \n";
	for(int i = 0; i < labels.size(); ++i){
		f << "           " << labels[i];
	}
	f << std::endl;
	for(int i = 0; i < amountOfClass; ++i){ // calc the covariance matrix for each f_c
		Eigen::MatrixXd cov_c = covariance; //  * y.segment(i*dataPoints, dataPoints).transpose();
		for(int j = 0; j < dataPoints; ++j){
			if(i == labels[j]){
				for(int k = 0; k < dataPoints; ++k){
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



#endif /* TESTS_MULTICLASSGPTEST_H_ */
