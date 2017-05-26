/*
 * GaussianProcessBinaryClass.cc
 *
 *  Created on: 15.06.2016
 *      Author: Max
 */

#include "GaussianProcess.h"
#include <iomanip>
#include <algorithm>

GaussianProcess::GaussianProcess(): m_repetitionStepFactor(1.0), m_dataPoints(0), m_init(false), m_trained(false), m_fastPredict(false), m_fastPredictVFStar(0){
}

GaussianProcess::~GaussianProcess(){
}

void GaussianProcess::init(const Matrix& dataMat, const VectorX& y){
	m_dataMat = dataMat;
	m_y = y;
	m_dataPoints = m_dataMat.cols();
	m_t = (m_y + VectorX::Ones(m_y.rows())) * 0.5;
	m_f = VectorX(m_dataPoints);
	m_pi = VectorX(m_dataPoints);
	m_dLogPi = VectorX(m_dataPoints);
	m_ddLogPi = VectorX(m_dataPoints);
	m_sqrtDDLogPi = VectorX(m_dataPoints);
	m_repetitionStepFactor = 1.0;
	m_init = true;
	m_innerOfLLT = Matrix(m_dataPoints,m_dataPoints);
	m_kernel.init(m_dataMat, m_kernel.hasLengthMoreThanOneDim(), false);
	m_fastPredict = false;
}

void GaussianProcess::updatePis(){
	for(int i = 0; i < m_dataPoints; ++i){
		m_pi[i] = 1.0 / (1.0 + exp((Real) - m_y[i] * (Real) m_f[i]));
		m_dLogPi[i] = m_t[i] - m_pi[i];
		m_ddLogPi[i] = -(-m_pi[i] * (1 - m_pi[i])); // first minus to get -ddlog(p_i|f_i)
		m_sqrtDDLogPi[i] = sqrtReal((Real) m_ddLogPi[i]);
	}
}

void GaussianProcess::train(){
	if(!m_init){
		printError("Init must be performed before gp can be trained!");
		return;
	}
	std::cout << "Start train with " << m_dataPoints << " points, with dim: " << m_dataMat.col(0).rows() << std::endl;
	Status status = Status::NANORINFERROR;
	int nanCounter = 0;
	while(status == Status::NANORINFERROR){
		status = train(m_dataPoints, m_dataMat, m_y);
		if(status == Status::NANORINFERROR){
			srand(nanCounter);
			m_kernel.newRandHyperParams();
			std::cout << "\rNan or inf case: " << nanCounter << std::endl;
			std::cout << m_kernel.prettyString() << std::endl;
			++nanCounter;
			m_kernel.init(m_dataMat, m_kernel.hasLengthMoreThanOneDim(), false);
			m_choleskyLLT.compute(Matrix::Identity(2,2));
			m_repetitionStepFactor *= 0.5;
			//getchar();
		}
	}
	m_trained = true;
}

GaussianProcess::Status GaussianProcess::trainBayOpt(Real& logZ, const Real lambda){
	UNUSED(lambda);
	Matrix K;
	//const VectorX ones = VectorX::Ones(m_dataPoints);
	m_kernel.calcCovariance(K);
	//std::cout << "K: \n" << K << std::endl;
	Status status = trainF(K);
	if(status == Status::NANORINFERROR){
		return Status::NANORINFERROR;
	}
	const VectorX diag = m_choleskyLLT.matrixL().toDenseMatrix().diagonal(); // TOdo more efficient?
	Real sum = 0;
	for(int i = 0; i < diag.rows(); ++i){
		sum += log((Real)diag[i]);
	}
	const Real prob = (1.0 + exp(-(Real) (m_y.dot(m_f)))); // should be very small!
	const Real logVal = prob < 1e100 ? -log(prob) : -100;
	const Real aDotF = -0.5 * (Real) (m_a.dot(m_f));
	//std::cout << "Prob: " << prob << std::endl;
	//std::cout << "Mean: " << m_f.mean() << std::endl;
	//std::cout << "a: " << m_a.transpose() << std::endl;
	//std::cout << "f: " << m_f.transpose() << std::endl;
	//sumF /= m_f.rows();
	logZ = aDotF + logVal - sum;
	std::cout << CYAN << "LogZ elements a * f: " << aDotF << ", log: " << logVal << ", sum: " << -sum << ", logZ: " << logZ << RESET << std::endl;
	// -0.5 * (Real) (m_a.dot(m_f)) - 0.5 *sumF
	return Status::ALLFINE;
}

GaussianProcess::Status GaussianProcess::trainLM(Real& logZ, std::vector<Real>& dLogZ){
	Matrix K;
	const VectorX ones = VectorX::Ones(m_dataPoints);
	m_kernel.calcCovariance(K);
	//std::cout << "K: \n" << K << std::endl;
	Status status = trainF(K);
	if(status == Status::NANORINFERROR){
		return Status::NANORINFERROR;
	}
	const VectorX diag = m_choleskyLLT.matrixL().toDenseMatrix().diagonal(); // TOdo more efficient?
	Real sum = 0;
	if(diag.rows() != m_f.rows()){
		printError("calc of cholesky failed!");
	}
	for(int i = 0; i < diag.rows(); ++i){
		sum += log((Real)diag[i]);
	}
	const Real prob = (1.0 + exp(-(Real) (m_y.dot(m_f)))); // should be very small!
	const Real logVal = prob < 1e100 ? -log(prob) : -100;
	//std::cout << "Prob: " << prob << std::endl;
	//std::cout << "Mean: " << m_f.mean() << std::endl;
	//std::cout << "a: " << m_a.transpose() << std::endl;
	//std::cout << "f: " << m_f.transpose() << std::endl;
	//std::cout << CYAN << "LogZ elements a * f: " << -0.5 * (Real) (m_a.dot(m_f)) << ", log: " << logVal << ", sum: " << sum << RESET << std::endl;
	logZ = -0.5 * (Real) (m_a.dot(m_f)) + logVal - sum;
	const DiagMatrixXd WSqrt(m_sqrtDDLogPi);
	const Matrix R = WSqrt * m_choleskyLLT.solve( m_choleskyLLT.solve(WSqrt.toDenseMatrix()));
	Matrix C = m_choleskyLLT.solve(WSqrt * K);
	const VectorX s2 = -0.5 * (K.diagonal() - (C.transpose() * C).diagonal()) + (2.0 * m_pi - ones); // (2.0 * m_pi - ones) == dddLogPi
	GaussianKernelElementLength len(m_kernel.hasLengthMoreThanOneDim());
	m_kernel.calcCovarianceDerivative(C, &len);
	Real s1 = 0.5 * ((Real) m_a.dot(C * m_a)) - (Real) (R * C).trace();
	VectorX b = C * m_dLogPi;
	VectorX s3 = b - (K * (R * b));
	dLogZ[0] = s1 + s2.dot(s3);
	GaussianKernelElementFNoise fNoise;
	m_kernel.calcCovarianceDerivative(C, &fNoise);
	s1 = 0.5 * ((Real) m_a.dot(C * m_a)) - (Real) (R * C).trace();
	b = C * m_dLogPi;
	s3 = b - (K * (R * b));
	dLogZ[1] = s1 + s2.dot(s3);
	return Status::ALLFINE;
}

void GaussianProcess::trainWithoutKernelOptimize(){
	m_fastPredict = false;
	Matrix K;
	m_kernel.calcCovariance(K);
	trainF(K);
	m_trained = true;
}

GaussianProcess::Status GaussianProcess::train(const int dataPoints,
		const Matrix& dataMat, const VectorX& y){
	UNUSED(dataPoints);
	UNUSED(dataMat);
	UNUSED(y);
/*
	Vector2 min,max;
	min << 0,0;
	max << 2.0,2.0;
	Real stepSize2 = 0.1;
	Real logZ2;
	std::vector<Real> dLogZ2;
	dLogZ2.reserve(3);
	m_kernel.setHyperParams(0.075994, 2.0588, 1.0);
	Status status = trainLM(logZ2, dLogZ2);
	std::cout << "LogZ: " << logZ2 << std::endl;
	return Status::ALLFINE;

	//plot different weights
	std::ofstream file;
	file.open("out.txt");
	for(Real xVal = max[0]; xVal >= min[0]; xVal -= stepSize2){
		for(Real yVal = min[1]; yVal < max[1]; yVal+= stepSize2){
			m_kernel.setHyperParams(xVal, yVal, 0.95);
			Status status = trainLM(logZ2, dLogZ2);
			if(status == Status::NANORINFERROR){
				logZ2 = 0.0;
			}
			file << xVal << " " << yVal << " " << logZ2 << "\n";

		}
		std::cout << "Done: " << xVal * 100.0 << "%" << std::endl;
	}
	file.close();
	return Status::ALLFINE;*/
	std::cout << std::setiosflags(std::ios::fixed) << std::setprecision(6);
	std::vector<Real> dLogZ;
	dLogZ.reserve(3);
	Real logZ;
	const int amountOfFirstSamples = 30;
	GaussianKernelParams bestParams(true);
	Real bestLogZ = -10000000000;
	for(int i = 0; i < amountOfFirstSamples; ++i){
		m_kernel.newRandHyperParams();
		m_kernel.setSNoise(0.95);
		Status status = trainLM(logZ, dLogZ);
		std::cout << "logZ: " << logZ << ", status: " << static_cast<int>(status) << ", bestlogZ: " << bestLogZ << ", " << (bestLogZ > fabs(logZ)) << ", with " << m_kernel.prettyString() << std::endl;
		if(bestLogZ < logZ && status != Status::NANORINFERROR && logZ < 1000){
			bestLogZ = logZ;
			m_kernel.getCopyOfParams(bestParams);
			std::cout << "\rnew optimal " << m_kernel.prettyString() << "\tavg time; " << m_sw.elapsedAvgAsPrettyTime() << "          ";
			flush(std::cout);
			std::cout << std::endl;
		}else if(status == Status::NANORINFERROR && bestLogZ == -10000000000){
			return Status::NANORINFERROR;
		}
	}
	std::cout << std::endl;
	m_kernel.setHyperParamsWith(bestParams);
	//return Status::ALLFINE;
	if(bestLogZ == 0){
		return Status::NANORINFERROR;
	}
	std::cout << "\nstart guess: " << m_kernel.prettyString() << std::endl;
	int counter = 0;
	bool converged = false;
	//Real stepSize = 0.0001 / (m_dataPoints * m_dataPoints) * m_repetitionStepFactor;
	std::vector<Real> stepSize;
	stepSize.reserve(3);
	Real lastDLogZ = 0.0;

	std::vector<Real> gradient;
	gradient.reserve(3);
	gradient[0] = gradient[1] = gradient[2] = 0.0;
	std::vector<Real> ESquared;
	ESquared.reserve(3);
	ESquared[0] = ESquared[1] = ESquared[2] = 0;
	while(!converged){
		Status status = trainLM(logZ, dLogZ);
		if(status == Status::NANORINFERROR){
			return Status::NANORINFERROR;
		}
		const Real dLogZSum = fabs(dLogZ[0]) + fabs(dLogZ[1]); // + fabs(dLogZ[2]);
		std::cout << std::endl;
		//flush(std::cout);
		for(int j = 0; j < 3; ++j){
			const Real lastLearningRate = sqrtReal(EPSILON + ESquared[j]); // 0,001
			ESquared[j] = 0.9 * ESquared[j] + 0.1 * dLogZ[j] * dLogZ[j]; // 0,0000000099856
			const Real actLearningRate = sqrtReal(EPSILON + ESquared[j]);  // 0,0001413704354
			const Real fac = std::max(0.0,(Real) (-counter + 100.0) / 100.0);
			if(j == 1){
				stepSize[j] = 0.00001; // 0,001222106928
			}else{
				stepSize[j] = 0.00001 * fac + (1.0-fac) * 0.01 * m_repetitionStepFactor * lastLearningRate / actLearningRate; // 0,001222106928
			}
			gradient[j] = stepSize[j] * dLogZ[j];
		}
		m_kernel.addHyperParams(gradient[0], gradient[1], gradient[2]);
		//std::cout << "\r                                                                                                                                 ";
		std::cout << "gradient: " << gradient[0] << ",\t" << gradient[1] << ",\t" << gradient[2] << std::endl;
		std::cout << "\rLogZ: " << logZ << ",\tdLogZ: " << dLogZ[0] << ",\t" << dLogZ[1]
				  << ",\t" << dLogZ[2] << " \t" << m_kernel.prettyString() << "\tavg time; "
				  << m_sw.elapsedAvgAsPrettyTime() << "\tstepsize len: " << stepSize[0] << "\tstepsize sigmaF: " << stepSize[1] <<"         " << std::endl;

		if(lastDLogZ * 0.9 > dLogZSum){
			m_kernel.setHyperParamsWith(bestParams); // start again!
			break;
		}
		if(dLogZSum < 0.01){
			converged = true;
		}

		/*convergingRate = fabs(sumHyperParams - lastSumHypParams);
		if(convergingRate < 0.01){ // diff to slow
			stepSize *= 2;
		}else if(convergingRate > 0.01){
			stepSize *= 0.1;
		}
		convergingRate = fabs(lastDLogZ - dLogZSum);
		if(convergingRate < 0.1){ // diff to slow
			stepSize *= 2;
		}else if(convergingRate > 0.1){
			stepSize *= 0.1;
		}*/


		/*if(counter % 50 == 0 && counter > 0){
			stepSize *= 2;
		}*/
		counter++;
		lastDLogZ = dLogZSum;
		m_kernel.getCopyOfParams(bestParams);
	}
	return Status::ALLFINE;
}

GaussianProcess::Status GaussianProcess::trainF(const Matrix& K){
	// find suited f:
	m_sw.startTime();
	m_fastPredict = false;
	// m_f = VectorX::Zero(m_dataPoints); 						// f <-- init with zeros
	m_f.fill(0);														// f <-- init with zeros
	bool converged = false;
	int j = 0;
	Real lastObjective = 100000000000;
	Real stepSize = 0.5;
	VectorX oldA = m_a;
	VectorX oldF = m_f;
	while(!converged){
		// calc - log p(y_i| f_i) -> -
		updatePis();
		//const Matrix WSqrt( DiagMatrixXd(m_sqrtDDLogPi).toDenseMatrix()); // TODO more efficient
		//std::cout << "K: \n" << K << std::endl;
		//std::cout << "inner: \n" << eye + (WSqrt * K * WSqrt) << std::endl;
		//m_innerOfLLT = eye + (WSqrt * K * WSqrt);
		for(int i = 0; i < m_dataPoints; ++i){
			for(int j = 0; j < m_dataPoints; ++j){
				m_innerOfLLT(i,j) = m_sqrtDDLogPi[i] * m_sqrtDDLogPi[j] * K(i,j);
			}
			m_innerOfLLT(i,i) += 1;
		}
		m_choleskyLLT.compute(m_innerOfLLT);
		const VectorX b = m_ddLogPi.cwiseProduct(m_f) + m_dLogPi;
		m_a = b - m_ddLogPi.cwiseProduct(m_choleskyLLT.solve( m_choleskyLLT.solve(m_ddLogPi.cwiseProduct(K * b)))); // WSqrt * == m_ddLogPi.cwiseProduct(...)
		const Real firstPart = -0.5 * (Real) (m_a.dot(m_f));
		const Real prob = 1.0 / (1.0 + exp(-(Real) (m_y.dot(m_f))));
		const Real tol = 1e-7;
		const Real offsetVal = prob > tol && prob < 1 - tol ? prob : (prob < tol ? tol : 1 - tol );
		const Real objective = firstPart + log(offsetVal);
		//std::cout << "\rError in " << j <<": " << fabs(lastObjective / objective - 1.0) << ", from: " << lastObjective << ", to: " << objective <<  ", log: " << log(offsetVal) << "                    " << std::endl;
		if(isnan(objective)){
			//std::cout << "Objective is nan!" << std::endl;
			return Status::NANORINFERROR;
		}
		if(objective > lastObjective && j != 0){
			m_a = oldA;
			m_f = oldF;
			updatePis();
			//const Matrix WSqrt( DiagMatrixXd(m_sqrtDDLogPi).toDenseMatrix()); // TODO more efficient
			//std::cout << "K: \n" << K << std::endl;
			//std::cout << "inner: \n" << eye + (WSqrt * K * WSqrt) << std::endl;
			//m_innerOfLLT = eye + (WSqrt * K * WSqrt);
			for(int i = 0; i < m_dataPoints; ++i){
				for(int j = 0; j < m_dataPoints; ++j){
					m_innerOfLLT(i,j) = m_sqrtDDLogPi[i] * m_sqrtDDLogPi[j] * K(i,j);
				}
				m_innerOfLLT(i,i) += 1;
			}
			m_choleskyLLT.compute(m_innerOfLLT);
			break;
			// decrease the step size and try again!
			//m_f = (m_f - lastF) * stepSize + lastF;
			stepSize /= 2.0;
			std::cout << "\rDecrease step size to: " << stepSize << ", isNan: "<< isnan(objective) << ",\tfrom: " << lastObjective << ",\tto: " << objective  << "              " << std::endl;
			flush(std::cout);
			if(stepSize < 0.005){
				if(j > 2){
					//m_f = lastF;
					updatePis();
					break;
				}
				m_f.fill(0); // start again!
				stepSize = 0.5;
				j = 0;
			}
			continue;
		}else{
			m_f = (K * m_a);
			stepSize = 0.5;
		}

		converged = fabs(lastObjective / objective - 1.0) < 0.0001 && j > 3;
		lastObjective = objective;
		oldA = m_a;
		oldF = m_f;
		//converged = m_f.mean() < 100 && m_f.mean() > 50;//fabs((m_f-lastF).mean()) < 0.0001;
		++j;
	//	lastF = m_f;
	}
	m_sw.recordActTime();
	return Status::ALLFINE;
}

Real GaussianProcess::predict(const VectorX& newPoint, const int sampleSize) const{
	if(!m_init || !m_trained){
		printError("GP was not init: " << m_init << ", or trained: " << m_trained);
		return -1.0;
	}
	Real fStar = 0, vFStar = 0;
	VectorX kXStar;
	m_kernel.calcKernelVector(newPoint, m_dataMat, kXStar);
	fStar = (Real) (kXStar.dot(m_dLogPi));
	if(m_fastPredict){
		vFStar = m_fastPredictVFStar;
	}else{
		const Real vNorm = m_choleskyLLT.solve(m_sqrtDDLogPi.cwiseProduct(kXStar)).squaredNorm();
		const Real leftTermVFStar = fabs(m_kernel.getHyperParams().m_sNoise.getSquaredValue() + 1);
		vFStar = leftTermVFStar - vNorm;
		if(fabs(vNorm / leftTermVFStar) < 0.01){ // is vNorm smaller than 1 % of the whole term, than just forget it -> speeds it up!
			m_fastPredict = true;
			m_fastPredictVFStar = vFStar; // will be nearly always the same!
		}
	}
//std::cout << " Fstar is " << fStar << std::endl;
	/*if(isnan(vFStar) || vFStar > 1e200){
		std::cout << "Kernel: " << m_kernel.prettyString() << std::endl;
	}*/
	const int amountOfSamples = sampleSize;
	const Real start = fStar - vFStar * 3;
	const Real end = fStar + vFStar * 3;
	const Real stepSize = (end- start) / amountOfSamples;
	Real prob = 0;
	for(Real p = start; p < end; p+=stepSize){
		const Real x = rand()/static_cast<Real>(RAND_MAX);
		const Real y = rand()/static_cast<Real>(RAND_MAX);
		const Real result = cos(2.0 * M_PI * x)*sqrtReal(-2*log(y)); // random gaussian after box mueller
		const Real height = 1.0 / (1.0 + exp(-p)) * (result * vFStar + fStar);
		prob += height * stepSize; // gives the integral
	}
	/*if(isnan(prob)){
		std::cout << RED << fStar << ", " << vFStar << RESET << std::endl;
	}*/
	if(prob < 0){
		return 0;
	}
	/*else if(prob > 1){
		return 1;
	}*/
	return prob;
}

Real GaussianProcess::predict(const VectorX& point) const{
	return predict(point, 5000);
}

void GaussianProcess::setKernelParams(const Real len, const Real fNoise, const Real sNoise){
	m_kernel.setHyperParams(len, fNoise, sNoise);
}

void GaussianProcess::setKernelParams(const std::vector<Real>& lens, const Real fNoise, const Real sNoise){
	m_kernel.setHyperParams(lens, fNoise, sNoise);
}

