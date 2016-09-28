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
	typedef Eigen::VectorXd Vector;
	typedef Eigen::MatrixXd Matrix;
	template <typename T>
	using List = std::list<T>;
	template <typename M, typename N>
	using Pair = std::list<M, N>;

	IVM();

	void init(const Matrix& dataMat, const Vector& y, const unsigned int numberOfInducingPoints);

	void train();

	virtual ~IVM();

private:

	Matrix m_dataMat;
	Matrix m_M;
	Vector m_y;
	unsigned int m_dataPoints;
	unsigned int m_numberOfInducingPoints;
	List<int> m_J, m_I;

	Kernel m_kernel;
	boost::math::normal m_logisticNormal;
};

#endif /* GAUSSIANPROCESS_IVM_H_ */
