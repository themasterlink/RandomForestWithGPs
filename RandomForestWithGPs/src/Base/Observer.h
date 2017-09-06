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

/**
 * \brief The Observer class is used, if a class A should do something if a certain event happens in a Subject class B
 * 		the class A observes.
 *
 * Deriving from this class is enough to ensure that the class can do actions, when something happens.
 * You have to implement the update(Subject* caller, unsigned int event) function.
 */
class Observer {
	friend class Subject;
public:
	/**
	 * \brief Constructor of the class.
	 *
	 * Std. Constructor
	 */
	Observer();

	/**
	 * \brief This function is called if an event happens in the Subject.
	 *
	 * The caller is given as a pointer of the type Subject and can then be casted accordingly.
	 * 		Each Subject class has an ClassTypeSubject classType(), which can be used to determine the class type
	 * 		of the object calling, to act accordingly
	 * \param caller which called this function
	 * \param event which started the update
	 */
	virtual void update(Subject* caller, unsigned int event) = 0;

	/**
	 * \brief Deconstructor of the class.
	 *
	 * Std. Deconstructor
	 */
	virtual ~Observer();
};

/**
 * \brief This type is used to find the type of a class in an update call.
 *
 * This helps to make a proper down-cast, all classes should be added.
 */
enum class ClassTypeSubject {
	ONLINESTORAGE,
	ONLINERANDOMFOREST,
	CLASSKNOWLEDGE,
	UNDEFINED
};

/**
 * \brief The Subject class is used, if a class A can perform actions and an other class B wants to get updates.
 *
 * Deriving from this class is enough to ensure that the class can inform other classes, when they do something.
 * You have to implement the ClassTypeSubject classType() function. Add a type to the ClassTypeSubject for an new
 * Subject class.
 */
class Subject {
	friend class Observer;
public:
	/**
	 * \brief The Constructor of this class.
	 */
	Subject();

	/**
	 * \brief attach a certain Observer* to the observer list.
	 *
	 * \param observer which is added to the observer list
	 */
	void attach(Observer* observer);

	/**
	 * \brief detach a certain Observer* from the observer list.
	 *
	 * \param observer which is delete from the observer list
	 */
	void detach(Observer* observer);

	/**
	 * \brief detach all observers from the list
	 *
	 */
	void detachAll();

	/**
	 * \brief Notify all appended observers, with the given event
	 *
	 * \param event event which is given to all observers
	 */
	void notify(const unsigned int event);

	/**
	 * \brief Returns the amount of observers
	 *
	 * \return the amount of observers
	 */
	unsigned int numberOfObservers() const;

	/**
	 * \brief Returns the class type of the class.
	 *
	 * This function has to be overwriten
	 *
	 * \return the class type of the derived class
	 */
	virtual ClassTypeSubject classType() const = 0;

	virtual ~Subject();

private:

	/**
	 * \brief list of observers, which contain all current observers
	 */
	std::list<Observer*> m_observers;

};

#endif /* BASE_OBSERVER_H_ */
