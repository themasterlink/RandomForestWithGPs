//
// Created by denn_ma on 5/26/17.
//

#ifndef RANDOMFORESTWITHGPS_REALTYPE_H
#define RANDOMFORESTWITHGPS_REALTYPE_H

//#define USE_DOUBLE
//#define USE_UNIT_TYPE

#ifdef USE_DOUBLE

using Real = double;

#else

using Real = float;

#endif

// dimension type for DDT in BigDDT

#ifdef USE_UINT_TYPE

using dimTypeForDDT = unsigned int;

#else

using dimTypeForDDT = unsigned short;

#endif


#endif //RANDOMFORESTWITHGPS_REALTYPE_H
