//============================================================================
// Name        : RandomForestWithGPs.cpp
// Author      : 
// Version     :
// Copyright   : 
// Description :
//============================================================================

#include <iostream>
#include <Eigen/Dense>
#include "kernelCalc.h"
#include "Utility/Settings.h"
#include "Data/DataReader.h"
#include "Data/DataWriterForVisu.h"
#include "RandomForests/RandomForest.h"
#include "RandomForests/RandomForestWriter.h"
#include <cmath>
#include <Eigen/Cholesky>

// just for testing

typedef Eigen::DiagonalWrapper<const Eigen::MatrixXd> DiagMatrixXd;


void calcCovariance(Eigen::MatrixXd& cov, const Eigen::MatrixXd& dataMat){
	Eigen::MatrixXd centered = dataMat.rowwise() - dataMat.colwise().mean();
	cov = centered.adjoint() * centered;
}

void calcPhiBasedOnF(const Eigen::VectorXd& f, Eigen::VectorXd& pi, const int amountOfClasses, const int dataPoints){
	const int amountOfEle = dataPoints * amountOfClasses;
	if(f.rows() != amountOfEle){
		printError("Amount of rows in f is wrong!");
	}
	pi = Eigen::VectorXd::Zero(amountOfEle);
	for(int i = 0; i < amountOfClasses; ++i){
		double normalizer = 0.;
		for(int j = 0; j < amountOfClasses; ++j){
			normalizer += exp((double) f[i * amountOfClasses + j]);
		}
		normalizer = 1.0 / normalizer;
		for(int j = 0; j < dataPoints; ++j){
			const int iActEle = i * dataPoints + j;
			pi[iActEle] = normalizer * exp((double) f[iActEle]);
		}
	}

}


void magicFunc(const int amountOfClasses, const int dataPoints, const std::vector<Eigen::MatrixXd>& K_c, const Eigen::VectorXd& y){
	const int amountOfEle = dataPoints * amountOfClasses;
	const Eigen::MatrixXd eye(Eigen::MatrixXd::Identity(dataPoints,dataPoints));
	Eigen::MatrixXd R(Eigen::MatrixXd::Zero(amountOfEle, dataPoints));			// R <-- compute R (just a giant stacked identy matrix)
	for(int j = 0; j < dataPoints; ++j){ // todo find faster way
		for(int i = 0; i < amountOfClasses; ++i){
			R(i*dataPoints + j,j) = 1;
		}
	}
	Eigen::VectorXd f = Eigen::VectorXd::Zero(amountOfEle); 						// f <-- init with zeros
	// R and f were check! -> should be right
	bool converged = false;
	while(!converged){
		std::fstream f2("t2.txt", std::ios::out);
		Eigen::VectorXd lastF = f;													// lastF <- save f for converge controll
		Eigen::VectorXd pi; 														// pi
		calcPhiBasedOnF(f, pi, amountOfClasses, dataPoints);
		// pi was checked! -> should be right
		Eigen::VectorXd sqrtPi(pi);													// sqrtPi
		for(int i = 0; i < amountOfEle; ++i){
			sqrtPi[i] = sqrt((double) sqrtPi[i]);
		}
		// sqrtPi was checked! -> should be right
		const Eigen::MatrixXd D(pi.asDiagonal().toDenseMatrix());					// D
		//Eigen::DiagonalWrapper<const Eigen::MatrixXd> DSqrt(sqrtPi.asDiagonal()); 	// DSqrt
		//std::vector<DiagMatrixXd*> DSqrt_c(amountOfClasses, NULL);					//	DSqrt_c
		std::vector<Eigen::MatrixXd> E_c(amountOfClasses);							// E_c

		//std::vector<Eigen::MatrixXd> K_c;											// K_c
		/*Eigen::MatrixXd F(amountOfClasses, dataPoints); 							// F just to calc covariances
		for(int i = 0; i < dataPoints; ++i){ // todo find better way
			for(int j = 0; j < amountOfClasses; ++j){
				F(j,i) = (double) f(i*amountOfClasses + j);
			}
		}

		for(int i = 0; i < amountOfClasses; ++i){ // calc the covariance matrix for each f_c
			const Eigen::MatrixXd centered = F.colwise() - F.rowwise().mean();
			K_c.push_back(centered.adjoint() * centered);
		}
*/
		// TODO find way to construct bigPi in a nice an efficient way ...
		Eigen::MatrixXd bigPi(amountOfEle, dataPoints);
		for(int i = 0; i < amountOfClasses / 2; i+=2){
			bigPi << pi.segment(i*dataPoints, dataPoints).asDiagonal().toDenseMatrix(),
					pi.segment((i+1)*dataPoints, dataPoints).asDiagonal().toDenseMatrix();
		}
		// bigPi was checked! -> should be right (checked only for classAmount = 2) (check for C > 2 && C is uneven

		Eigen::MatrixXd E_sum;
		Eigen::VectorXd z(amountOfClasses);
		//std::vector<DiagMatrixXd*>::iterator it = DSqrt_c.begin();
		for(int i = 0; i < amountOfClasses; ++i){
			//delete(*it); // free last iteration, in init it is null
			//it = DSqrt_c.insert(it, sqrtPi.segment(i*dataPoints, dataPoints).asDiagonal());
			//DiagMatrixXd* pDSqrt_c= *it;
			//if(pDSqrt_c == NULL){
			//	printError("NULL");
			//}
			const DiagMatrixXd DSqrt_c(sqrtPi.segment(i*dataPoints, dataPoints));
			// DSqrt_c was checked! -> should be right
			Eigen::MatrixXd C = (DSqrt_c * K_c[i] * DSqrt_c) + eye;
			// C was checked -> might be right (didn't try to calc it)
			Eigen::MatrixXd L = Eigen::LLT<Eigen::MatrixXd>(C).matrixL();
			f2 << "C:\n" << C << std::endl;
			f2 << "\n\n";
			f2 << "L*L^T:\n" << L * L.transpose() << std::endl;
			f2 << "\n\n";
			f2 << "L:\n" << L << std::endl;
			f2 << "\n\n";
			Eigen::MatrixXd nenner = L.triangularView<Eigen::Lower>().solve(DSqrt_c.toDenseMatrix());
			f2 << "nenner:\n" << nenner << std::endl;
			f2 << "\n\n";
			E_c[i] = DSqrt_c * L.transpose().triangularView<Eigen::Upper>().solve(nenner);
			for(int j = 0; j < dataPoints; ++j){
				z[i] += log((double) L(j,j));
			}
			if(i == 0){
				E_sum = E_c[i];
			}else{
				E_sum += E_c[i];
			}
		}

		Eigen::MatrixXd M = Eigen::LLT<Eigen::MatrixXd>(E_sum).matrixL();

		Eigen::VectorXd b = (D - (bigPi * bigPi.transpose())) * f + y - pi;							// b

		Eigen::VectorXd c(amountOfEle);																// c
		for(int i = 0; i < amountOfClasses; ++i){
			const Eigen::VectorXd k = E_c[i] * K_c[i] * b.segment(i*dataPoints, dataPoints);
			for(int j = 0; j < dataPoints; ++j){ // todo rewrite -> faster
				c[i*dataPoints + j] = k[j];
			}
		}
		Eigen::MatrixXd E(amountOfEle, amountOfEle);
		for(int i = 0; i < amountOfClasses; ++i){
			for(int j = 0; j < dataPoints; ++j){
				for(int k = 0; k < dataPoints; ++k){
					E(i*dataPoints + j, i*dataPoints + k) = E_c[i](j,k);
				}
			}
		}
		Eigen::MatrixXd res = M.triangularView<Eigen::Lower>().solve(R.transpose() * c); 			// M^-1 * (R^T* c)
		f2 << b.transpose() << "\n\n\n\n\n";
		f2 << E_c[0]<< "\n\n\n\n\n";
		f2 << E_c[1]<< "\n\n\n\n\n";
		f2 << M << "\n\n\n\n\n";
		f2 << M.triangularView<Eigen::Lower>().toDenseMatrix() << "\n\n\n\n\n";
		f2 << M.transpose().triangularView<Eigen::Upper>().toDenseMatrix() << "\n\n\n\n\n";
		f2 << (M.transpose().triangularView<Eigen::Upper>().solve(res)) << "\n\n\n\n\n";
		f2 << K_c[0]<< "\n\n\n\n\n";
		f2 << K_c[1]<< "\n\n\n\n\n";

		f2.close();
		std::cout << "b: " << b.transpose() << std::endl;
		std::cout << "\n\n";
		std::cout << "c: " << c.transpose() << std::endl;
		std::cout << "\n\n";
		const Eigen::VectorXd a = b - c + E * R * (M.transpose().triangularView<Eigen::Upper>().solve(res)); // b-c + E * R * ((M^T)^-1 * (res))
		std::cout << "a: " << a.transpose() << std::endl;
		std::cout << "\n\n";
		std::cout << "before f: " << f.transpose() << std::endl;
		for(int i = 0; i < amountOfClasses; ++i){

			const Eigen::VectorXd k = K_c[i] * a.segment(i*dataPoints, dataPoints);
			std::cout << "k: " << k.transpose() << std::endl;
			for(int j = 0; j < dataPoints; ++j){ // todo rewrite -> faster
				f[i*dataPoints + j] = k[j];
			}
		}
		std::cout << "\n\n";
		std::cout << "after f: " << f.transpose() << std::endl;
		std::cout << "new mean: " << (f-lastF).mean() << std::endl;
		converged = false; // fabs((f-lastF).mean()) < 0.0001;
		getchar();
	}
}


int main(){

	std::cout << "Start" << std::endl;
	// read in Settings
	Settings::init("../Settings/init.json");
	Data data;
	Labels labels;
	std::string path;
	Settings::getValue("Training.path", path);
	DataReader::readFromFile(data, labels, path);

	const int dataPoints = data.size();
	Eigen::MatrixXd dataMat;

	dataMat.conservativeResize(data[0].rows(), data.size());
	int i = 0;
	for(Data::iterator it = data.begin(); it != data.end(); ++it){
		dataMat.col(i++) = *it;
	}
	const int amountOfClass = 2;
	std::vector<Data> dataPerClass(amountOfClass);
	for(int i = 0; i < data.size(); ++i){
		dataPerClass[labels[i]].push_back(data[i]);
	}
	Eigen::VectorXd y(Eigen::VectorXd::Zero(dataPoints * amountOfClass));
	for(int i = 0; i < labels.size(); ++i){
		y[labels[i] * dataPoints + i] = 1;
	}
	std::fstream f("t.txt", std::ios::out);
	//f << "dataMat:\n" << dataMat << std::endl;
	std::vector<Eigen::MatrixXd> cov;

	Eigen::MatrixXd covariance;
	calcCovariance(covariance, dataMat);
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
	magicFunc(amountOfClass,dataPoints, cov, y);

	std::cout << "finish" << std::endl;
	return 0;
	bool useFixedValuesForMinMaxUsedData;
	Settings::getValue("MinMaxUsedData.useFixedValuesForMinMaxUsedData", useFixedValuesForMinMaxUsedData);
	Eigen::Vector2i minMaxUsedData;
	if(useFixedValuesForMinMaxUsedData){
		int minVal = 0, maxVal = 0;
		Settings::getValue("MinMaxUsedData.minValue", minVal);
		Settings::getValue("MinMaxUsedData.maxValue", maxVal);
		minMaxUsedData << minVal, maxVal;
	}else{
		double minVal = 0, maxVal = 0;
		Settings::getValue("MinMaxUsedData.minValueFraction", minVal);
		Settings::getValue("MinMaxUsedData.maxValueFraction", maxVal);
		minMaxUsedData << (int) (minVal * data.size()),  (int) (maxVal * data.size());
	}
	std::cout << "Min max used data, min: " << minMaxUsedData[0] << " max: " << minMaxUsedData[1] << "\n";

	Data testData;
	Labels testLabels;
	Settings::getValue("Test.path", path);
	DataReader::readFromFile(testData, testLabels, path);

	std::cout << "Finished reading" << std::endl;
	int dim = 2;
	if(data.size() > 0){
		dim = data[0].rows();
	}

	int height;
	int amountOfTrees;
	Settings::getValue("Forest.Trees.height", height, 7);
	Settings::getValue("Forest.amountOfTrees", amountOfTrees, 1000);
	std::cout << "Amount of trees: " << amountOfTrees << " with height: " << height << std::endl;

	RandomForest forest(height, amountOfTrees, dim);
	forest.train(data, labels, dim, minMaxUsedData);

	Labels guessedLabels;
	forest.predictData(testData, guessedLabels);

	int wrong = 0;
	for(int i = 0; i < testData.size(); ++i){
		if(guessedLabels[i] != testLabels[i]){
			++wrong;
		}
	}
	std::cout << "Other: Amount of test size: " << testData.size() << std::endl;
	std::cout << "Other: Amount of wrong: " << wrong / (double) testData.size() << std::endl;



	bool doWriting;
	Settings::getValue("WriteBinarySaveOfTrees.doWriting", doWriting, false);
	if(doWriting){
		std::string path;
		Settings::getValue("WriteBinarySaveOfTrees.path", path);
		RandomForestWriter::writeToFile(path, forest);
	}
	RandomForest newForest(0,0,0);
	RandomForestWriter::readFromFile("../testData/trees2.binary", newForest);
	Labels guessedLabels2;
	newForest.addForest(forest);
	newForest.predictData(testData, guessedLabels2);

	wrong = 0;
	for(int i = 0; i < testData.size(); ++i){
		if(guessedLabels2[i] != testLabels[i]){
			++wrong;
		}
	}
	std::cout << "Amount of combined trees: " << newForest.getNrOfTrees() << std::endl;
	std::cout << "Read: Amount of test size: " << testData.size() << std::endl;
	std::cout << "Read: Amount of wrong: " << wrong / (double) testData.size() << std::endl;


	int printX, printY;
	Settings::getValue("Write2D.doWriting", doWriting, false);
	Settings::getValue("Write2D.printX", printX, 0);
	Settings::getValue("Write2D.printY", printY, 1);
	if(doWriting){
		Settings::getValue("Write2D.gridPath", path);
		StopWatch sw;
		DataWriterForVisu::generateGrid(path, forest, 200, data, printX, printY);
		Settings::getValue("Write2D.testPath", path);
		DataWriterForVisu::writeData(path, testData, testLabels, printX, printY);
		std::cout << "Time for write: " << sw.elapsedSeconds() << std::endl;
		std::cout << "End Reached" << std::endl;
		system("../PythonScripts/plotData.py");
	}



	return 0;
}

