/*
 * Observer.h
 *
 *  Created on: 13.10.2016
 *      Author: Max
 */

#ifndef BASE_OBSERVER_H_
#define BASE_OBSERVER_H_

#include <list>

class Subject;

class Observer {
	friend Subject;
public:
	Observer();

	virtual void update(Subject* caller, unsigned int event) = 0;

	virtual ~Observer();
};



class Subject {
	friend Observer;
public:
	Subject();

	void attach(Observer* obj);

	void deattach(Observer* obj);

	void deattachAll();

	void notify(const unsigned int event);

	unsigned int numberOfObservers() const;

	virtual ~Subject();

private:

	std::list<Observer*> m_observers;

};

#endif /* BASE_OBSERVER_H_ */
