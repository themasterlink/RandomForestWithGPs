//
// Created by denn_ma on 5/26/17.
//

#ifndef RANDOMFORESTWITHGPS_REALTYPE_H
#define RANDOMFORESTWITHGPS_REALTYPE_H

#include <memory>

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

using MemoryType = unsigned long;

template<typename T>
using SharedPtr = std::shared_ptr<T>;

template<typename T>
using UniquePtr = std::unique_ptr<T>;

#define BUILD_SYSTEM_LINUX @BUILD_SYSTEM_CMAKE@

#define SingeltonMacro(ClassName) \
    public: \
    static ClassName& instance(){ \
        static ClassName m_instance; \
        return m_instance; }; \
    ~ClassName() = default; \
    ClassName(const ClassName&) = delete; \
    void operator=(const ClassName&) = delete; \
    private: \
        ClassName(); \

#endif //RANDOMFORESTWITHGPS_REALTYPE_H
