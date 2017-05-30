//
// Created by denn_ma on 5/24/17.
//

#ifndef RANDOMFORESTWITHGPS_TYPES_H
#define RANDOMFORESTWITHGPS_TYPES_H

#include <boost/random.hpp>
#include "BaseType.h"
#include <Eigen/Dense>
#include <vector>
#include <list>
#include <map>
#include <string>

using GeneratorType = boost::random::taus88;

#ifdef USE_DOUBLE

using VectorX = Eigen::VectorXd;

using Matrix = Eigen::MatrixXd;

using Vector2 = Eigen::Vector2d;

static const auto REAL_MAX = DBL_MAX;

static const auto NEG_REAL_MAX = -REAL_MAX;

#else

using VectorX = Eigen::VectorXf;

using Matrix = Eigen::MatrixXf;

using Vector2 = Eigen::Vector2f;

static const auto REAL_MAX = FLT_MAX;

static const auto NEG_REAL_MAX = -REAL_MAX;

#endif

using DiagMatrixXd = Eigen::DiagonalWrapper<const Matrix>;

using Vector2i = Eigen::Vector2i;

using Data = std::vector<VectorX*>;

using DataIterator = Data::iterator;

using DataConstIterator = Data::const_iterator;

using Labels = std::vector<unsigned int>;

template <typename T>
using List = std::list<T>;

#endif //RANDOMFORESTWITHGPS_TYPES_H
