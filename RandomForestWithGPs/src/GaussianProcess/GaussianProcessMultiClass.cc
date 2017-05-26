/*
 * GaussianProcessMultiClass.cc
 *
 *  Created on: 15.06.2016
 *      Author: Max
 */

#include "GaussianProcessMultiClass.h"
#include <cmath>

GaussianProcessMultiClass::GaussianProcessMultiClass(){
	// TODO Auto-generated constructor stub
}

GaussianProcessMultiClass::~GaussianProcessMultiClass()
{
	// TODO Auto-generated destructor stub
}


void GaussianProcessMultiClass::calcPhiBasedOnF(const VectorX& f, VectorX& pi, const int amountOfClasses, const int dataPoints){
	const int amountOfEle = dataPoints * amountOfClasses;
	if(f.rows() != amountOfEle){
		printError("Amount of rows in f is wrong!");
	}
	pi = VectorX::Zero(amountOfEle);
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

void GaussianProcessMultiClass::magicFunc(const int amountOfClasses, const int dataPoints, const std::vector<Matrix>& K_c, const VectorX& y){
	const int amountOfEle = dataPoints * amountOfClasses;
	const Matrix eye(Matrix::Identity(dataPoints,dataPoints));
	Matrix R(Matrix::Zero(amountOfEle, dataPoints));			// R <-- compute R (just a giant stacked identy matrix)
	for(int j = 0; j < dataPoints; ++j){ // todo find faster way
		for(int i = 0; i < amountOfClasses; ++i){
			R(i*dataPoints + j,j) = 1;
		}
	}
	VectorX f = VectorX::Zero(amountOfEle); 						// f <-- init with zeros
	// R and f were check! -> should be right
	bool converged = false;
	while(!converged){
		std::fstream f2("t2.txt", std::ios::out);
		VectorX lastF = f;													// lastF <- save f for converge controll
		VectorX pi; 														// pi
		calcPhiBasedOnF(f, pi, amountOfClasses, dataPoints);
		// pi was checked! -> should be right
		VectorX sqrtPi(pi);													// sqrtPi
		for(int i = 0; i < amountOfEle; ++i){
			sqrtPi[i] = sqrt((double) sqrtPi[i]);
		}
		// sqrtPi was checked! -> should be right
		const Matrix D(pi.asDiagonal().toDenseMatrix());					// D
		//Eigen::DiagonalWrapper<const Matrix> DSqrt(sqrtPi.asDiagonal()); 	// DSqrt
		//std::vector<DiagMatrixXd*> DSqrt_c(amountOfClasses, NULL);					//	DSqrt_c
		std::vector<Matrix> E_c(amountOfClasses);							// E_c

		//std::vector<Matrix> K_c;											// K_c
		/*Matrix F(amountOfClasses, dataPoints); 							// F just to calc covariances
		for(int i = 0; i < dataPoints; ++i){ // todo find better way
			for(int j = 0; j < amountOfClasses; ++j){
				F(j,i) = (double) f(i*amountOfClasses + j);
			}
		}

		for(int i = 0; i < amountOfClasses; ++i){ // calc the covariance matrix for each f_c
			const Matrix centered = F.colwise() - F.rowwise().mean();
			K_c.push_back(centered.adjoint() * centered);
		}
*/
		// TODO find way to construct bigPi in a nice an efficient way ...
		Matrix bigPi(amountOfEle, dataPoints);
		for(int i = 0; i < amountOfClasses / 2; i+=2){
			bigPi << pi.segment(i*dataPoints, dataPoints).asDiagonal().toDenseMatrix(),
					pi.segment((i+1)*dataPoints, dataPoints).asDiagonal().toDenseMatrix();
		}
		// bigPi was checked! -> should be right (checked only for classAmount = 2) (check for C > 2 && C is uneven

		Matrix E_sum;
		VectorX z(amountOfClasses);
		//std::vector<DiagMatrixXd*>::iterator it = DSqrt_c.begin();
		for(int i = 0; i < amountOfClasses; ++i){
			//SAVE_DELETE(*it); // free last iteration, in init it is null
			//it = DSqrt_c.insert(it, sqrtPi.segment(i*dataPoints, dataPoints).asDiagonal());
			//DiagMatrixXd* pDSqrt_c= *it;
			//if(pDSqrt_c == NULL){
			//	printError("NULL");
			//}
			const DiagMatrixXd DSqrt_c(sqrtPi.segment(i*dataPoints, dataPoints));
			// DSqrt_c was checked! -> should be right
			Matrix C = (DSqrt_c * K_c[i] * DSqrt_c) + eye;
			// C was checked -> might be right (didn't try to calc it)
			Matrix L = Eigen::LLT<Matrix>(C).matrixL();
			f2 << "C:\n" << C << std::endl;
			f2 << "\n\n";
			f2 << "L*L^T:\n" << L * L.transpose() << std::endl;
			f2 << "\n\n";
			f2 << "L:\n" << L << std::endl;
			f2 << "\n\n";
			Matrix nenner = L.triangularView<Eigen::Lower>().solve(DSqrt_c.toDenseMatrix());
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

		Matrix M = Eigen::LLT<Matrix>(E_sum).matrixL();

		VectorX b = (D - (bigPi * bigPi.transpose())) * f + y - pi;							// b

		VectorX c(amountOfEle);																// c
		for(int i = 0; i < amountOfClasses; ++i){
			const VectorX k = E_c[i] * K_c[i] * b.segment(i*dataPoints, dataPoints);
			for(int j = 0; j < dataPoints; ++j){ // todo rewrite -> faster
				c[i*dataPoints + j] = k[j];
			}
		}
		Matrix E(amountOfEle, amountOfEle);
		for(int i = 0; i < amountOfClasses; ++i){
			for(int j = 0; j < dataPoints; ++j){
				for(int k = 0; k < dataPoints; ++k){
					E(i*dataPoints + j, i*dataPoints + k) = E_c[i](j,k);
				}
			}
		}
		Matrix res = M.triangularView<Eigen::Lower>().solve(R.transpose() * c); 			// M^-1 * (R^T* c)
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
		const VectorX a = b - c + E * R * (M.transpose().triangularView<Eigen::Upper>().solve(res)); // b-c + E * R * ((M^T)^-1 * (res))
		std::cout << "a: " << a.transpose() << std::endl;
		std::cout << "\n\n";
		std::cout << "before f: " << f.transpose() << std::endl;
		for(int i = 0; i < amountOfClasses; ++i){

			const VectorX k = K_c[i] * a.segment(i*dataPoints, dataPoints);
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
