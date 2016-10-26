/*
 * IVM.h
 *
 *  Created on: 27.09.2016
 *      Author: Max
 */

#ifndef GAUSSIANPROCESS_IVM_H_
#define GAUSSIANPROCESS_IVM_H_

#include "../Data/Data.h"
#include "Kernel.h"
#include <boost/math/distributions/normal.hpp> // for normal_distribution
#include <list>

class IVM {
public:
	typedef typename Eigen::VectorXd Vector;
	typedef typename Eigen::MatrixXd Matrix;
	template <typename T>
	using List = std::list<T>;
	template <typename M, typename N>
	using Pair = std::list<M, N>;

	IVM();

	void init(const Matrix& dataMat, const Vector& y,
			const unsigned int numberOfInducingPoints, const bool doEPUpdate);

	void setNumberOfInducingPoints(unsigned int nr);

	bool train(bool clearActiveSet = true, const int verboseLevel = 0);

	double predict(const Vector& input) const;

	Kernel& getKernel(){ return m_kernel; };

	const List<int>& getSelectedInducingPoints(){ return m_I; };

	void setDerivAndLogZFlag(const bool doLogZ, const bool doDerivLogZ);

	virtual ~IVM();

	double m_logZ;
	Vector m_derivLogZ;

private:

	double calcInnerOfFindPointWhichDecreaseEntropyMost(const unsigned int j,
			const Vector& zeta, const Vector& mu,
			double& g_kn, double& nu_kn,
			const double fraction, const Eigen::Vector2i& amountOfPointsPerClassLeft, const int verboseLevel);

	double cumulativeLog(const double x);

	double cumulativeDerivLog(const double x);

	void calcDerivatives(const Vector& muL1);

	void calcLogZ();

	Matrix m_dataMat;
	Matrix m_M;
	Matrix m_K;
	Matrix m_L;
	Matrix m_eye;
	Vector m_y;
	Vector m_nuTilde;
	Vector m_tauTilde;
	Vector m_muTildePlusBias;
	unsigned int m_dataPoints;
	unsigned int m_numberOfInducingPoints;
	double m_bias;
	double m_lambda;
	bool m_doEPUpdate;
	double m_desiredFraction;
	bool m_calcLogZ;
	bool m_calcDerivLogZ;
	List<int> m_J, m_I;

	Eigen::LLT<Eigen::MatrixXd> m_choleskyLLT;

	Kernel m_kernel;
	boost::math::normal m_logisticNormal;


};

#endif /* GAUSSIANPROCESS_IVM_H_ */
