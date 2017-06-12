//
// Created by denn_ma on 6/12/17.
//

#ifndef RANDOMFORESTWITHGPS_CLASSCOUNTER_H
#define RANDOMFORESTWITHGPS_CLASSCOUNTER_H

#include "../Utility/Util.h"

class ClassCounter {
public:

	unsigned int operator[](unsigned int classNr) const;

	void increment(unsigned int classNr);

	void decrement(unsigned int classNr);

	unsigned int argMax();

	// returns the new max class
	unsigned int incrementWithChange(unsigned int classNr, unsigned int& oldMaxClass);

private:
	std::map<unsigned int, unsigned int> m_classCounter;
};


#endif //RANDOMFORESTWITHGPS_CLASSCOUNTER_H
