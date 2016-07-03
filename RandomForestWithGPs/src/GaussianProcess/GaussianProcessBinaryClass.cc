/*
 * GaussianProcessBinaryClass.cc
 *
 *  Created on: 15.06.2016
 *      Author: Max
 */

#include "GaussianProcessBinaryClass.h"
#include "GaussianProcessMultiClass.h"
#include <iomanip>
#include "../Data/Data.h"

GaussianProcessBinaryClass::GaussianProcessBinaryClass(): m_repetitionStepFactor(1.0), m_dataPoints(0), m_init(false), m_trained(false){
}

GaussianProcessBinaryClass::~GaussianProcessBinaryClass(){
}

void GaussianProcessBinaryClass::init(const Eigen::MatrixXd& dataMat, const Eigen::VectorXd& y){
	m_dataMat = dataMat;
	m_y = y;
	m_dataPoints = m_dataMat.cols();
	m_repetitionStepFactor = 1.0;
	m_init = true;
	m_kernel.init(m_dataMat);
}

void GaussianProcessBinaryClass::updatePis(const int dataPoints, const Eigen::VectorXd& y, const Eigen::VectorXd& t){
	for(int i = 0; i < dataPoints; ++i){
		m_pi[i] = 1.0 / (1.0 + exp((double) -y[i] * (double) m_f[i]));
		m_dLogPi[i] = t[i] - m_pi[i];
		m_ddLogPi[i] = -(-m_pi[i] * (1 - m_pi[i])); // first minus to get -ddlog(p_i|f_i)
		m_sqrtDDLogPi[i] = sqrt((double) m_ddLogPi[i]);
	}
}


void GaussianProcessBinaryClass::train(){
	if(!m_init){
		printError("Init must be performed before gp can be trained!");
		return;
	}
	std::cout << "Start train wiht: " << m_dataPoints << " points." << std::endl;
	Status status = NANORINFERROR;
	int nanCounter = 0;
	while(status == NANORINFERROR){
		status = train(m_dataPoints, m_dataMat, m_y);
		if(status == NANORINFERROR){
			srand(nanCounter);
			m_kernel.newRandHyperParams();
			std::cout << "\rNan or inf case: " << nanCounter << std::endl;
			std::cout << "len: " << m_kernel.len() << "\t, sigF: " << m_kernel.sigmaF() << "\t, sigN: " << m_kernel.sigmaN() << std::endl;
			++nanCounter;
			m_kernel.init(m_dataMat);
			m_choleskyLLT.compute(Eigen::MatrixXd::Identity(2,2));
			getchar();

		}
	}
	m_trained = true;
}

GaussianProcessBinaryClass::Status GaussianProcessBinaryClass::trainLM(double& logZ, std::vector<double>& dLogZ){
	Eigen::MatrixXd K;
	const Eigen::VectorXd ones = Eigen::VectorXd::Ones(m_dataPoints);
	m_kernel.calcCovariance(K);
	Status status = trainF(m_dataPoints, K, m_y);
	if(status == NANORINFERROR){
		return NANORINFERROR;
	}
	const Eigen::VectorXd diag = m_choleskyLLT.matrixL().toDenseMatrix().diagonal(); // TOdo more efficient?
	double sum = 0;
	for(int i = 0; i < diag.rows(); ++i){
		sum += log((double)diag[i]);
	}
	const double prob = (1.0 + exp(-(double) (m_y.dot(m_f))));
	const double tol = 1e-7;
	const double offsetVal = prob > tol && prob < 1 - tol ? prob : (prob < tol ? tol : 1 -tol );
	logZ = -0.5 * (double) (m_a.dot(m_f)) - log(offsetVal) + sum;
	const DiagMatrixXd WSqrt(m_sqrtDDLogPi);
	const Eigen::MatrixXd R = WSqrt * m_choleskyLLT.solve( m_choleskyLLT.solve(WSqrt.toDenseMatrix()));
	Eigen::MatrixXd C = m_choleskyLLT.solve(WSqrt * K);
	const Eigen::VectorXd s2 = -0.5 * (K.diagonal() - (C.transpose() * C).diagonal()) + (2.0 * m_pi - ones); // (2.0 * m_pi - ones) == dddLogPi
	int i = 0;
	for(int type = Kernel::LENGTH; type != Kernel::NNOISE; ++type){
		const Kernel::ParamType paramType = static_cast<Kernel::ParamType>(type);
		m_kernel.calcCovarianceDerivative(C, paramType);
		const double s1 = 0.5 * ((double) m_a.dot(C * m_a)) - (double) (R * C).trace();
		const Eigen::VectorXd b = C * m_dLogPi;
		const Eigen::VectorXd s3 = b - (K * (R * b));
		dLogZ[i] = s1 + s2.dot(s3);
		if(isnan(dLogZ[i])){
			std::cout << "dlogZ["<< i << "] is nan" << std::endl;
			std::cout << "s1: " << s1 << std::endl;
			std::cout << "s1: " << s1 << std::endl;
			std::cout << "s2: " << s2.transpose() << std::endl;
			std::cout << "s3: " << s3.transpose() << std::endl;
			std::cout << "b: " << b.transpose() << std::endl;
			return NANORINFERROR;
		}
		++i;
	}
	return ALLFINE;
}

GaussianProcessBinaryClass::Status GaussianProcessBinaryClass::train(const int dataPoints,
		const Eigen::MatrixXd& dataMat, const Eigen::VectorXd& y){

	Eigen::Vector2d min,max;
	min << 0,0;
	max << 2.0,2.0;
	double stepSize2 = 0.1;
	double logZ2;
	std::vector<double> dLogZ2;
	dLogZ2.reserve(3);
	m_kernel.setHyperParams(0.38, 1.3, 0.95);
	Status status = trainLM(logZ2, dLogZ2);
	std::cout << "LogZ: " << logZ2 << std::endl;
	return ALLFINE;
	/*
	//plot different weights
	std::ofstream file;
	file.open("out.txt");
	for(double xVal = max[0]; xVal >= min[0]; xVal -= stepSize2){
		for(double yVal = min[1]; yVal < max[1]; yVal+= stepSize2){
			m_kernel.setHyperParams(xVal, yVal, 0.95);
			Status status = trainLM(logZ2, dLogZ2);
			if(status == NANORINFERROR){
				logZ2 = 0.0;
			}
			file << xVal << " " << yVal << " " << logZ2 << "\n";

		}
		std::cout << "Done: " << xVal * 100.0 << "%" << std::endl;
	}
	file.close();
	return ALLFINE;*/
	//std::cout << std::setiosflags(std::ios::fixed) << std::setprecision(6);
	std::vector<double> dLogZ;
	dLogZ.reserve(3);
	double logZ;
	const int amountOfFirstSamples = 100;
	std::vector<double> val;
	val.reserve(3);
	val[0] = val[1] = val[2] = 0;
	double bestLogZ = -10000000000;
	for(int i = 0; i < amountOfFirstSamples; ++i){
		m_kernel.newRandHyperParams();
		m_kernel.setHyperParams(m_kernel.len(),m_kernel.sigmaF(),1);
		Status status = trainLM(logZ, dLogZ);
		std::cout << "logZ: " << logZ << ", status: " << status << ", bestlogZ: " << bestLogZ << ", " << (bestLogZ > fabs(logZ))<<std::endl;
		if(bestLogZ < logZ && status != NANORINFERROR){
			bestLogZ = logZ;
			m_kernel.getHyperParams(val);
			std::cout << "\rnew optimal len: " << m_kernel.len() << "\t, sigF: "
						<< m_kernel.sigmaF() << "\t, sigN: " << m_kernel.sigmaN() << "\tavg time; " << m_sw.elapsedAvgAsPrettyTime() << "          ";
			flush(std::cout);
		}else if(status == NANORINFERROR && bestLogZ == -10000000000){
			return NANORINFERROR;
		}
	}
	m_kernel.setHyperParams(val);
	if(bestLogZ == 0){
		return NANORINFERROR;
	}
	std::cout << "\nstart guess len: " << m_kernel.len() << "\t, sigF: " << m_kernel.sigmaF()<< "\t, sigN: " << m_kernel.sigmaN() << std::endl;
	int counter = 0;
	bool converged = false;
	double stepSize = 0.0001 / (m_dataPoints * m_dataPoints) * m_repetitionStepFactor;
	std::vector<double> vt;
	vt.reserve(3);
	vt[0] = vt[1] = vt[2] = 0.0;
	double lastSumHypParams = 0;
	double convergingRate = 1;
	double lastDLogZ = 0.0;
	while(!converged){
		Status status = trainLM(logZ, dLogZ);
		if(status == NANORINFERROR){
			return NANORINFERROR;
		}
		std::cout << "\r                                                                                                                                 ";
		std::cout << "\rLogZ: " << logZ << ",\tdLogZ: " << dLogZ[0] << ",\t" << dLogZ[1]
			<< ",\t" << dLogZ[2] << " \tlen: " << m_kernel.len() << "\t, sigF: " << m_kernel.sigmaF()<< "\t, sigN: " << m_kernel.sigmaN()  << "\tavg time; " << m_sw.elapsedAvgAsPrettyTime() << "\tstepsize: " << stepSize <<"     ";
		const double dLogZSum = fabs(dLogZ[0]) + fabs(dLogZ[1]); // + fabs(dLogZ[2]);
		const double sumHyperParams = fabs(m_kernel.len()) + fabs(m_kernel.sigmaF());
		std::cout << std::endl;
		//flush(std::cout);
		for(int j = 0; j < 3; ++j){
			vt[j] =  dLogZ[j] * stepSize + 0.9 * vt[j];
		}
		m_kernel.addHyperParams(vt);
		if(dLogZSum < 0.1){
			converged = true;
		}

		/*convergingRate = fabs(sumHyperParams - lastSumHypParams);
		if(convergingRate < 0.01){ // diff to slow
			stepSize *= 2;
		}else if(convergingRate > 0.01){
			stepSize *= 0.1;
		}*/
		/*convergingRate = fabs(lastDLogZ - dLogZSum);
		if(convergingRate < 0.1){ // diff to slow
			stepSize *= 2;
		}else if(convergingRate > 0.1){
			stepSize *= 0.1;
		}*/


		/*if(counter % 50 == 0 && counter > 0){
			stepSize *= 2;
		}*/
		counter++;
		lastSumHypParams = sumHyperParams;
		lastDLogZ = dLogZSum;
	}
	return ALLFINE;
}

GaussianProcessBinaryClass::Status GaussianProcessBinaryClass::trainF(const int dataPoints, const Eigen::MatrixXd& K, const Eigen::VectorXd& y){
	// find suited f:
	m_sw.startTime();
	m_f = Eigen::VectorXd::Zero(dataPoints); 						// f <-- init with zeros
	const Eigen::MatrixXd eye(Eigen::MatrixXd::Identity(dataPoints,dataPoints));
	bool converged = false;
	int j = 0;
	m_pi = Eigen::VectorXd::Zero(dataPoints);
	m_dLogPi = Eigen::VectorXd::Zero(dataPoints);
	m_ddLogPi = Eigen::VectorXd::Zero(dataPoints);
	m_sqrtDDLogPi = Eigen::VectorXd::Zero(dataPoints);
	const Eigen::VectorXd t = (y + Eigen::VectorXd::Ones(y.rows())) * 0.5; // Todo uneccessary often executed
	double lastObjective = 100000000000;
	double stepSize = 0.5;
	while(!converged){
		// calc - log p(y_i| f_i) -> -
		updatePis(dataPoints,y,t);

		const Eigen::MatrixXd WSqrt( DiagMatrixXd(m_sqrtDDLogPi).toDenseMatrix()); // TODO more efficient
		//std::cout << "K: \n" << K << std::endl;
		//std::cout << "inner: \n" << eye + (WSqrt * K * WSqrt) << std::endl;
		m_choleskyLLT.compute(eye + (WSqrt * K * WSqrt));
		const Eigen::VectorXd b = m_ddLogPi.cwiseProduct(m_f) + m_dLogPi;
		m_a = b - m_ddLogPi.cwiseProduct(m_choleskyLLT.solve( m_choleskyLLT.solve(m_ddLogPi.cwiseProduct(K * b)))); // WSqrt * == m_ddLogPi.cwiseProduct(...)
		const double firstPart = -0.5 * (double) (m_a.transpose() * m_f);
		const double prob = 1.0 / (1.0 + exp(-(double) (y.transpose() * m_f)));
		const double tol = 1e-7;
		const double offsetVal = prob > tol && prob < 1 - tol ? prob : (prob < tol ? tol : 1 -tol );
		const double objective = firstPart + log(offsetVal);
		//std::cout << "\rError in " << j <<": " << fabs(lastObjective / objective - 1.0) << ", from: " << lastObjective << ", to: " << objective <<  ", log: " << log(offsetVal) << "                    " << std::endl;
		if(isnan(objective)){
			std::cout << "Objective is nan!" << std::endl;
			return NANORINFERROR;
		}
		if(objective > lastObjective && j != 0){
			break;
			// decrease the step size and try again!
			//m_f = (m_f - lastF) * stepSize + lastF;
			stepSize /= 2.0;
			std::cout << "\rDecrease step size to: " << stepSize << ", isNan: "<< isnan(objective) << ",\tfrom: " << lastObjective << ",\tto: " << objective  << "              " << std::endl;
			flush(std::cout);
			if(stepSize < 0.005){
				if(j > 2){
					//m_f = lastF;
					updatePis(dataPoints,y,t);
					break;
				}
				m_f = Eigen::VectorXd::Zero(dataPoints); // start again!
				stepSize = 0.5;
				j = 0;
			}
			continue;
		}else{
			m_f = (K * m_a);
			stepSize = 0.5;
		}

		converged = fabs(lastObjective / objective - 1.0) < 0.005 && j > 3;
		lastObjective = objective;
		//converged = m_f.mean() < 100 && m_f.mean() > 50;//fabs((m_f-lastF).mean()) < 0.0001;
		++j;
	//	lastF = m_f;
	}
	m_sw.recordActTime();
	return ALLFINE;
}

double GaussianProcessBinaryClass::predict(const DataElement& newPoint) const{
	if(!m_init || !m_trained){
		return -1.0;
	}
	const DiagMatrixXd WSqrt(m_sqrtDDLogPi);
	Eigen::VectorXd kXStar;
	m_kernel.calcKernelVector(newPoint, m_dataMat, kXStar);
	const double fStar = (double) (kXStar.transpose() * (m_dLogPi));
	const Eigen::VectorXd v = m_choleskyLLT.solve(WSqrt * kXStar);
	const double vFStar = fabs((m_kernel.sigmaN() * m_kernel.sigmaN() + 1) - v.transpose() * v);

	const int amountOfSamples = 50000;
	const double start = fStar - vFStar * 3;
	const double end = fStar + vFStar * 3;
	const double stepSize = (end- start) / amountOfSamples;
	double prob = 0;
	for(double p = start; p < end; p+=stepSize){
		const double x = rand()/static_cast<double>(RAND_MAX);
		const double y = rand()/static_cast<double>(RAND_MAX);
		const double result = cos(2.0 * M_PI * x)*sqrt(-2*log(y)); // random gaussian after box mueller
		const double height = 1.0 / (1.0 + exp(p)) * (result * vFStar + fStar);
		prob += height * fabs(stepSize); // gives the integral
	}
	if(prob < 0){
		return 0;
	}else if(prob > 1){
		return 1;
	}
	return prob;
}
