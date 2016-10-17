/*
 * IVM.cc
 *
 *  Created on: 27.09.2016
 *      Author: Max
 */

#include "IVM.h"
#include <boost/math/special_functions/erf.hpp>
#include "../Data/DataWriterForVisu.h"
#include "../Utility/Settings.h"

#define LOG2   0.69314718055994528623
#define LOG2PI 1.8378770664093453391
#define SQRT2  1.4142135623730951455

IVM::IVM(): m_logZ(0), m_derivLogZ(2), m_dataPoints(0), m_numberOfInducingPoints(0), m_bias(0), m_lambda(0), m_doEPUpdate(false), m_desiredFraction(0) {
}

IVM::~IVM() {
}

void IVM::init(const Matrix& dataMat, const Vector& y, const unsigned int numberOfInducingPoints, const bool doEPUpdate){
	m_dataMat = dataMat;
	m_y = y;
	m_doEPUpdate = doEPUpdate;
	if(m_y.rows() != m_dataMat.cols()){
		printError("Amount of data points and labels must be the same!");
		return;
	}
	m_dataPoints = m_dataMat.cols();
	setNumberOfInducingPoints(numberOfInducingPoints);
	StopWatch sw;
	m_kernel.init(m_dataMat);
	std::cout << "Time: " << sw.elapsedAsPrettyTime() << std::endl;
	int amountOfOneClass = 0;
	for(unsigned int i = 0; i < m_dataPoints; ++i){
		if(m_y[i] == -1){
			++amountOfOneClass;
		}
	}
	//std::cout << "Frac: " << (double) amountOfOneClass / (double) m_dataPoints << std::endl;
	m_bias = boost::math::cdf(boost::math::complement(m_logisticNormal, (double) amountOfOneClass / (double) m_dataPoints));
	Settings::getValue("IVM.lambda", m_lambda);
	Settings::getValue("IVM.desiredFraction", m_desiredFraction);
}

void IVM::setNumberOfInducingPoints(unsigned int nr){
	m_numberOfInducingPoints = std::min(nr, m_dataPoints);
	m_nuTilde = Vector(m_numberOfInducingPoints);
	m_tauTilde = Vector(m_numberOfInducingPoints);
}

double IVM::cumulativeDerivLog(const double x){
	return -(LOG2PI + x * x) * 0.5;
}

double IVM::cumulativeLog(const double x){
	return boost::math::erfc(-x / SQRT2) - LOG2;
}

bool IVM::train(bool clearActiveSet, const int verboseLevel){
	if(m_kernel.calcDiagElement() == 0){
		if(verboseLevel != 0)
			printError("The kernel diagonal is 0, this kernel params are invalid:" << m_kernel.prettyString());
		return false;
	}
	if(m_numberOfInducingPoints <= 0){
		if(verboseLevel != 0)
			printError("The number of inducing points is equal or below zero: " << m_numberOfInducingPoints);
		return false;
	}
	if(!m_kernel.isInit() || isnan(m_kernel.kernelFunc(0,0)) || m_kernel.calcDiagElement() != m_kernel.kernelFunc(0,0)){
		if(verboseLevel != 0)
			printError("The kernel was not initalized!");
		return false;
	}
	if(verboseLevel == 2)
		std::cout << "Diff: " << m_kernel.getDifferences(0,0) << ", " << m_kernel.getDifferences(1,0) << std::endl;
	Vector m = Vector::Zero(m_dataPoints);
	Vector beta = Vector::Zero(m_dataPoints);
	Vector mu = Vector::Zero(m_dataPoints);
	Vector zeta = Vector(m_dataPoints);
	if(clearActiveSet){
		m_I.clear();
	}else{
		if(m_I.size() != m_numberOfInducingPoints){
			printError("The active set size is not correct! Reset active set!");
			m_I.clear();
			clearActiveSet = true;
		}
	}
	m_J.clear();
	Eigen::Vector2i amountOfPointsPerClass;
	amountOfPointsPerClass[0] = amountOfPointsPerClass[1] = 0;
	for(unsigned int i = 0; i < m_dataPoints; ++i){
		zeta[i] = m_kernel.calcDiagElement();
		m_J.push_back(i);
		++amountOfPointsPerClass[(m_y[i] == 1 ? 0 : 1)];
	}
	Vector g = Vector(m_numberOfInducingPoints);
	Vector nu = Vector(m_numberOfInducingPoints);
	Vector delta = Vector(m_numberOfInducingPoints);
	StopWatch updateMat, findPoints;
	findPoints.startTime();
	double fraction = 0.;
	//std::cout << "bias: " << m_bias << std::endl;
	List<int>::const_iterator itOfActiveSet = m_I.begin();
	List<double> deltaValues;
	List<std::string> colors;
	for(unsigned int k = 0; k < m_numberOfInducingPoints; ++k){
		int argmax = -1;
		//List<Pair<int, double> > pointEntropies;
		delta[k] = -DBL_MAX;
		if(clearActiveSet){
			for(List<int>::const_iterator itOfJ = m_J.begin(); itOfJ != m_J.end(); ++itOfJ){
				double gForJ, nuForJ;
				const double deltaForJ = calcInnerOfFindPointWhichDecreaseEntropyMost(*itOfJ, zeta, mu, gForJ, nuForJ, fraction, amountOfPointsPerClass, verboseLevel);
				if(deltaForJ > delta[k]){
					argmax = *itOfJ;
					delta[k] = deltaForJ;
					g[k] = gForJ;
					nu[k] = nuForJ;
				}
			}
		}else{
			argmax = *itOfActiveSet;
			double gForArgmax, nuForArgmax;
			delta[k] = calcInnerOfFindPointWhichDecreaseEntropyMost(argmax, zeta, mu, gForArgmax, nuForArgmax, fraction, amountOfPointsPerClass, verboseLevel);
			g[k] = gForArgmax;
			nu[k] = nuForArgmax;
			++itOfActiveSet;
		}
		deltaValues.push_back((double) delta[k]);
		colors.push_back(std::string(m_y[argmax] == 1 ? "red" : "blue"));
		if(argmax == -1 && m_J.size() > 0){
			if(verboseLevel != 0){
				for(List<int>::const_iterator it = m_I.begin(); it != m_I.end(); ++it){
					std::cout << "(" << *it << ", " << (double) m_y[*it] << ")" << std::endl;
				}
				printError("No new inducing point was found and there are still points over!");
			}
			return false;
		}else if(argmax == -1){
			if(verboseLevel != 0)
				printError("No new inducing point was found, because no points are left to process, number of inducing points: "
						<< m_numberOfInducingPoints << ", size: " << m_dataPoints);
			return false;
		}
		fraction = ((fraction * k) + (m_y[argmax] == 1 ? 1 : 0)) / (double) (k + 1);
		if(verboseLevel == 2)
			printDebug("Next i is: " << argmax << " has label: " << (double) m_y[argmax]);
		// refine site params, posterior params & M, L, K
		if(fabs((double)g[k]) < EPSILON){
			m[argmax] = mu[argmax];
		}else if(fabs((double)nu[k]) > EPSILON){
			m[argmax] = g[k] / nu[k] + mu[argmax];
		}else{
			printError("G is zero and nu is not!");
			return false;
		}
		beta[argmax] = nu[k] / (1.0 - nu[k] * zeta[argmax]);
		if(beta[argmax] < EPSILON){
			beta[argmax] = EPSILON;
		}
		Vector s_nk = Vector(m_dataPoints), k_nk = Vector(m_dataPoints); // k_nk is not filled for k == 0!!!!
		Vector a_nk;
		if(k != 0){
			for(unsigned int i = 0; i < m_dataPoints; ++i){
				k_nk[i] = m_kernel.kernelFunc(i, argmax); // kernel from best point with all points
			}
			for(unsigned int i = 0; i < m_dataPoints; ++i){ // TODO for known active set only the relevant values have to been updated!
				double temp = 0.;
				for(unsigned int j = 0; j < k; ++j){
					temp += m_M(j, argmax) * m_M(j,i);
				}
				s_nk[i] = k_nk[i] - temp; // s_nk = k_nk - temp;
			}
			/*Vector colVec = m_M.col(argmax);
			s_nk = k_nk - (colVec.transpose() * m_M).transpose();*/
		}else{
			for(unsigned int i = 0; i < m_dataPoints; ++i){
				s_nk[i] = m_kernel.kernelFunc(i, argmax); // kernel from best point with all points
			}
		}
		if(verboseLevel == 2){
			std::cout << "Next: " << argmax << std::endl;
			std::cout << "zeta: " << zeta.transpose() << std::endl;
			std::cout << "mu: " << mu.transpose() << std::endl;
			//	std::cout << "k_nk: " << k_nk.transpose() << std::endl;
			std::cout << "s_nk: " << s_nk.transpose() << std::endl;
		}
		//zeta -= ((double) nu[k]) * s_nk.cwiseProduct(s_nk);
		//mu += ((double) g[k]) * s_nk; // <=> mu += g[k] * s_nk;
		for(unsigned int i = 0; i < m_dataPoints; ++i){ // TODO for known active set only the relevant values have to been updated!
			zeta[i] -= nu[k] * (s_nk[i] * s_nk[i]); // <=> zeta -= nu[k] * s_nk.cwiseProduct(s_nk); // <=> diag(A^new) = diag(A) - (u^2)_j
			mu[i] += g[k] * s_nk[i]; // <=> mu += g[k] * s_nk; // h += alpha_i * ( K_.,i - M_.,i^T * M_.,i) <=> alpha_i * (k_nk - s_nk)
		}
		/* IVM script:
		 * h += alpha_i * l / sqrt(p_i) * ->mu
		 * h += alpha_i * l / sqrt(p_i) * (1 / l * (sqrt(p_i) * K_.,i - sqrt(p_i) * M_.,i^T * M_.,i))
		 * h += alpha_i * l / sqrt(p_i) * (sqrt(p_i) / l * (K_.,i - M_.,i^T * M_.,i))
		 * h += alpha_i * (K_.,i - M_.,i^T * M_.,i)
		 * diag(A) -= (u_j^2)_j
		 * diag(A) -= l^-2 * (sqrt(p_i) / l * (K_.,i - M_.,i^T * M_.,i))^2
		 * diag(A) -= l^-2 * (p_i / l^2 * (K_.,i - M_.,i^T * M_.,i)^2)
		 * diag(A) -= p_i * (K_.,i - M_.,i^T * M_.,i)^2
		 * for: s_nk = (K_.,i - M_.,i^T * M_.,i)
		 * diag(A) -= p_i * s_nk.cwiseProduct(s_nk)
		 */
		if(nu[k] < 0.0){
			if(verboseLevel != 0){
				printError("The actual nu is below zero!");
				for(List<int>::const_iterator it = m_I.begin(); it != m_I.end(); ++it){
					std::cout << "(" << *it << ", " << (double) m_y[*it] << ")" << std::endl;
				}
			}
			return false;
		}
		const double sqrtNu = sqrt((double)nu[k]);
		// update K and L
		/*
		if(k == 0){
			m_K = Matrix(1,1);
			m_K(0,0) = m_kernel.calcDiagElement();
			m_L = Matrix(1,1);
			m_L(0,0) = 1.0 / sqrtNu;
		}else{
			Vector k_vec = Vector(m_I.size());
			unsigned int t = 0;
			for(List<int>::const_iterator itOfI = m_I.begin(); itOfI != m_I.end(); ++itOfI, ++t){
				k_vec[t] = m_kernel.kernelFunc(*itOfI, argmax);
			}
			Matrix D(m_K.rows() + 1, m_K.cols() + 1);
			D << m_K, k_vec,
			     k_vec.transpose(), m_kernel.calcDiagElement();
			m_K = D;
			// update L
			a_nk = m_M.col(argmax);
			Matrix D2(m_L.rows() + 1, m_L.cols() + 1);
			D2 << m_L, Vector::Zero(k),
					a_nk.transpose(), 1. / sqrtNu;
			m_L = D2;
		}*/
		if(k==0){
			if(m_doEPUpdate){
				m_K = Matrix(m_numberOfInducingPoints, m_numberOfInducingPoints); // init at beginning to avoid realloc
				m_K(0,0) = m_kernel.calcDiagElement();
			}
			m_L = Matrix::Zero(m_numberOfInducingPoints, m_numberOfInducingPoints);
			m_L(0,0) = 1.0 / sqrtNu;
			m_M = Matrix(m_numberOfInducingPoints, m_dataPoints);
		}else{
			if(m_doEPUpdate){
				unsigned int t = 0;
				const unsigned int lastRowAndCol = k;
				for(List<int>::const_iterator itOfI = m_I.begin(); itOfI != m_I.end() && t < lastRowAndCol; ++itOfI, ++t){
					// uses the kernel matrix from the actual element with all other elements in the active set
					const double temp = k_nk[*itOfI]; // <=> is the same: m_kernel.kernelFunc(*itOfI, argmax); saves recalc
					m_K(lastRowAndCol, t) = temp;
					m_K(t, lastRowAndCol) = temp;
				}
				m_K(lastRowAndCol, lastRowAndCol) = m_kernel.calcDiagElement();
			}
			// update L
			if(argmax < m_M.cols()){
				for(unsigned int i = 0; i < k; ++i){
					m_L(k,i) = m_M(i, argmax); // a_nk[i]; with a_nk = m_M.col(argmax);
				}
				m_L(k, k) = 1. / sqrtNu;
			}else{
				printError("The argmax value is bigger than the amount of columns in M!"); return false;
			}
		}
		updateMat.startTime();
		// update M
		/*if(k == 0){
			m_M = Matrix(m_numberOfInducingPoints, m_dataPoints);
			for(unsigned int i = 0; i < m_dataPoints; ++i){
				m_M(0,i) = sqrtNu * s_nk[i];
			}
		}else{
			Matrix D(m_M.rows() + 1, m_M.cols());
			D << m_M,
				(sqrtNu * s_nk).transpose();
			m_M = D;
		}*/
		for(unsigned int i = 0; i < m_dataPoints; ++i){
			m_M(k,i) = sqrtNu * s_nk[i];
		}
		updateMat.recordActTime();
		if(clearActiveSet){
			m_I.push_back(argmax);
		}
		m_J.remove(argmax);
		--amountOfPointsPerClass[m_y[argmax] == 1 ? 0 : 1];
	}
	std::cout << "Fraction in including points is: " << fraction * 100. << " %"<< std::endl;
	std::cout << "Upd M: " << updateMat.elapsedAvgAsPrettyTime() << std::endl;
	std::cout << "Find " << m_numberOfInducingPoints << " points: " << findPoints.elapsedAsPrettyTime() << std::endl;
	//DataWriterForVisu::writeSvg("deltas.svg", deltaValues, colors);
	//openFileInViewer("deltas.svg");
	if(m_I.size() != m_numberOfInducingPoints){
		if(verboseLevel != 0)
			printError("The active set has not the desired amount of points");
		return false;
	}
	unsigned int l = 0;
	Vector muSqueezed(m_numberOfInducingPoints);
	for(List<int>::const_iterator itOfI = m_I.begin(); itOfI != m_I.end(); ++itOfI, ++l){
		m_nuTilde[l] = m[*itOfI] * beta[*itOfI];
		m_tauTilde[l] = beta[*itOfI];
		muSqueezed[l] = mu[*itOfI];
	}
	const Matrix I = Matrix::Identity(m_numberOfInducingPoints, m_numberOfInducingPoints); // is needed after EP, too
	// calc m_L
	//std::cout << "m_L: \n" << m_L << std::endl;
	//std::cout << "m_K: \n" << m_K << std::endl;
	//std::cout << "m_M: \n" << m_M << std::endl;
	m_choleskyLLT.compute(m_L);
	if(m_doEPUpdate){ // EP update
		Matrix Sigma = m_K * (I - m_choleskyLLT.solve(m_K));
		//Matrix controlSigma = m_K * (I - m_choleskyLLT.solve(m_K));
		double deltaMax = 1.0;
		const unsigned int maxEpCounter = 100;
		double epThreshold = 1e-7;
		std::list<double> listToPrint;
		//double minDelta = DBL_MAX;
		StopWatch updateEP;
		StopWatch sigmaUp, sigmaUpNew;
		unsigned int counter = 0;
		for(; counter < maxEpCounter && deltaMax > epThreshold; ++counter){
			updateEP.startTime();
			Vector deltaTau(m_numberOfInducingPoints);
			unsigned int i = 0;
			for(List<int>::const_iterator itOfI = m_I.begin(); itOfI != m_I.end(); ++itOfI, ++i){
				const double tauMin = 1. / Sigma(i,i) - m_tauTilde[i];
				const double nuMin  = muSqueezed[i] / Sigma(i,i) - m_nuTilde[i];
				const unsigned int index = (*itOfI);
				const double label = m_y[index];

				const std::complex<double> tau_c(tauMin, 0);
				//double denom = std::max(abs(sqrt(tau_c * (tau_c / (lambda * lambda) + 1.))), EPSILON);
				double denom = std::max(std::abs((sqrt(tau_c * (tau_c / (m_lambda * m_lambda) + 1.0)))), EPSILON);
				const double c = label * tauMin / denom;
				double u;
				if(fabs(nuMin) < EPSILON){
					u = c * m_bias;
				}else{
					u = label * nuMin / denom + c * m_bias;
				}
				const double dlZ = c * exp(cumulativeDerivLog(u) - cumulativeLog(u));
				const double d2lZ  = dlZ * (dlZ + u * c);

				const double oldTauTilde = m_tauTilde[i];
				denom = 1.0 - d2lZ / tauMin;
				m_tauTilde[i] = std::max(d2lZ / denom, 0.);
				m_nuTilde[i]  = (dlZ + nuMin / tauMin * d2lZ) / denom;
				deltaTau[i]  = m_tauTilde[i] - oldTauTilde;

				// update approximate posterior
				/*
				sigmaUpNew.startTime();
				Vector si = Sigma.col(i);
				denom = 1.0 + deltaTau[i] * si[i];
				//if(fabs(denom) > EPSILON)
				Sigma -= (deltaTau[i] / denom) * (si * si.transpose());
				sigmaUpNew.recordActTime();
				 */
				sigmaUpNew.startTime();
				const Vector oldSigmaCol = Sigma.col(i);
				denom = 1.0 + deltaTau[i] * oldSigmaCol[i]; // <=> 1.0 + deltaTau[i] * si[i] for si = Sigma.col(i)
				const double fac = deltaTau[i] / denom;
				// is the same as Sigma -= (deltaTau[i] / denom) * (si * si.transpose()); but faster
				for(int p = 0; p < m_I.size(); ++p){
					Sigma(p,p) -= fac * oldSigmaCol[p] * oldSigmaCol[p];
					for(int q = p + 1; q < m_I.size(); ++q){
						const double sub = fac * oldSigmaCol[p] * oldSigmaCol[q];
						Sigma(p,q) -= sub;
						Sigma(q,p) -= sub;
					}
				}
				sigmaUpNew.recordActTime();

				/*for(int p = 0; p < m_I.size(); ++p){
			 	for(int q = 0; q < m_I.size(); ++q){
			 		if(fabs(controlSigma(p,q) - Sigma(p,q)) > fabs(Sigma(p,q)) * 1e-7){
			 			printError("Calc is wrong!");
			 		}
			 	}
			 }*/
				//else
				//Sigma -= delta_tau[i] / EPSILON * GP_Matrix::OutProd(si);
				muSqueezed = Sigma * m_nuTilde;
			}
			/*Vector _s_sqrt = Vector(n); // not used in the moment
		 const double sqrtEps = sqrt(EPSILON);
		 for(unsigned int i=0; i<m_numberOfInducingPoints; i++){
			 if(m_tauTilde[i] > EPSILON)
				 _s_sqrt[i] = sqrt(m_tauTilde[i]);
			 else
				 _s_sqrt[i] = sqrtEps;
		 }*/
			deltaMax = deltaTau.cwiseAbs().maxCoeff();
			listToPrint.push_back(deltaMax);
			//minDelta = std::min(minDelta, deltaMax);
			updateEP.recordActTime();
		}
		std::cout << "new sigma up time: " << sigmaUpNew.elapsedAvgAsPrettyTime() << std::endl;
		std::cout << "total new sigma up time: " << sigmaUpNew.elapsedAvgAsTimeFrame() * ((double) counter * m_I.size())<< std::endl;
		std::cout << "Ep time: " << updateEP.elapsedAvgAsPrettyTime() << std::endl;
		std::cout << "Total ep time: " << updateEP.elapsedAvgAsTimeFrame() * (double) counter << std::endl;
		//std::cout << "Min delta: " << minDelta << std::endl;

		//DataWriterForVisu::writeSvg("deltas.svg", listToPrint, true);
		//system("open deltas.svg");
	/*	Matrix temp = m_K;
		for(unsigned int i = 0; i < m_tauTilde.rows(); ++i){
			temp(i,i) += 1. / (double) m_tauTilde[i];
		}
		m_choleskyLLT.compute(temp);
		m_L = temp;*/
		m_L = m_K + DiagMatrixXd(m_tauTilde.cwiseInverse()).toDenseMatrix();
		m_choleskyLLT.compute(m_L);
		// compute log z
	}
	//std::cout << "m_L: \n" << m_L << std::endl;
	m_logZ = 0.0;
	const Matrix llt = m_choleskyLLT.matrixL().toDenseMatrix();
	//std::cout << "llt: \n" << llt << std::endl;
	for(unsigned int i = 0; i < m_numberOfInducingPoints; ++i){
		m_logZ -= log((double) llt(i,i));
	}
	const Vector muTilde = m_nuTilde.cwiseQuotient(m_tauTilde);
	Vector muL0 = Vector::Zero(m_numberOfInducingPoints);
	for(uint i=0; i<m_numberOfInducingPoints; ++i){
		double sum = muTilde[i];
		for(int k = (int)i-1; k >= 0; --k){
			sum -= (double)llt(i,k) * muL0[k];
		}
		muL0[i] = sum / (double) llt(i,i);
	}
	Vector muL1 = Vector::Zero(m_numberOfInducingPoints);
	for(int i= (int) m_numberOfInducingPoints - 1; i >= 0; --i){
		double sum = muL0[i];
		for(int k = i+1; k < m_numberOfInducingPoints; ++k){
			sum -= (double)llt(k,i) * muL1[k];
		}
		muL1[i] = sum / (double)llt(i,i);
	}
	m_logZ -= 0.5 * muTilde.dot(muL1);

	// compute derivatives
	Matrix Z2 = (muL1 * muL1.transpose()) - m_choleskyLLT.solve(I) * 0.5;
	Matrix CLen;
	m_kernel.calcCovarianceDerivativeForInducingPoints(CLen, m_I, Kernel::LENGTH);
	Matrix CFNoise;
	m_kernel.calcCovarianceDerivativeForInducingPoints(CFNoise, m_I, Kernel::FNOISE);
	m_derivLogZ[0] = m_derivLogZ[1] = 0;
	for(unsigned int i = 0; i < m_numberOfInducingPoints; ++i){
		for(unsigned int j = 0; j < m_numberOfInducingPoints; ++j){
			m_derivLogZ[0] += Z2(i,j) * CLen(i,j);
			m_derivLogZ[1] += Z2(i,j) * CFNoise(i,j);
		}
	}
	m_muTildePlusBias = m_nuTilde.cwiseQuotient(m_tauTilde) + (m_bias * Vector::Ones(m_numberOfInducingPoints));
	return true;
}

double IVM::calcInnerOfFindPointWhichDecreaseEntropyMost(const unsigned int j, const Vector& zeta, const Vector& mu, double& g_kn, double& nu_kn, const double fraction, const Eigen::Vector2i& amountOfPointsPerClassLeft, const int verboseLevel){
	const double label = m_y[j];
	if(amountOfPointsPerClassLeft[0] > 0 && amountOfPointsPerClassLeft[1] > 0){
		if((fraction < m_desiredFraction && label == -1) || (fraction > (1. - m_desiredFraction) && label == 1)){
			// => only less than 20 % of data is 1 choose 1
			return -DBL_MAX; // or only less than 20 % of data is -1 choose -1
		}
	}
	const double tau = 1.0 / zeta[j];
	const std::complex<double> tau_c(tau, 0);
	//double denom = std::max(abs(sqrt(tau_c * (tau_c / (lambda * lambda) + 1.))), EPSILON);
	const double denom = std::max(std::abs((sqrt(tau_c * (tau_c / (m_lambda * m_lambda) + 1.0)))), EPSILON);
	const double c = label * tau / denom;
	nu_kn = mu[j] / zeta[j];
	double u;
	if(fabs(nu_kn) < EPSILON){
		u = c * m_bias;
	}else{
		u = label * nu_kn / denom + c * m_bias;
	}
	g_kn = c * exp(cumulativeDerivLog(u) - cumulativeLog(u));
	nu_kn = g_kn * (g_kn + u * c);
	const double delta_kn = log(1.0 - nu_kn * (double) zeta[j]) / (2.0 * LOG2);
	//const double delta_kn = zeta[j] * nu_kn;
	// pointEntropies.append( (j, delta_ln));
	if(verboseLevel == 2){
		std::cout << (label == 1 ? RED : CYAN) << "j: " << j << ", is: " << label << ", with: "
				<< delta_kn << ", g: " << g_kn << ", nu: " << nu_kn << ", zeta: " << (double) zeta[j] << ", c: " << c << ", u: " << u<< RESET << std::endl; }
	/*if(delta_kn > delta[k]){ // nu_kn > EPSILON avoids that the ivm is not trained
		//if(k == j){
		//if(nu_kn < 0.)
		delta[k] = delta_kn;
		nu[k] = nu_kn;
		g[k] = g_kn;
		argmax = j;
	}*/
	return delta_kn;
}

double IVM::predict(const Vector& input) const{
	const unsigned int n = m_I.size();
	Vector k_star(n);
	unsigned int i = 0;
	for(List<int>::const_iterator itOfI = m_I.begin(); itOfI != m_I.end(); ++itOfI, ++i){
		k_star[i] = m_kernel.kernelFuncVec(input, m_dataMat.col(*itOfI));
	}
	const Vector v = m_choleskyLLT.solve(k_star);
	/*
	const Vector mu_tilde = m_nuTilde.cwiseQuotient(m_tauTilde);
	double mu_star = (mu_tilde + (m_bias * Vector::Ones(n))).dot(v);*/
	double mu_star = m_muTildePlusBias.dot(v);
	double sigma_star = (m_kernel.calcDiagElement() - k_star.dot(v));
	//std::cout << "mu_start: " << mu_star << std::endl;
	//std::cout << "sigma_star: " << sigma_star << std::endl;
	double contentOfSig = 0;
	if(1.0 / (m_lambda * m_lambda) + sigma_star < 0){
		contentOfSig = mu_star;
	}else{
		contentOfSig = (mu_star / sqrt(1.0 / (m_lambda * m_lambda) + sigma_star));
	}
	return boost::math::erfc(-contentOfSig / SQRT2) / 2.0;
}
