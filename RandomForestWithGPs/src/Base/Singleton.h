//
// Created by denn_ma on 7/6/17.
//

#ifndef RANDOMFORESTWITHGPS_SINGLETON_H
#define RANDOMFORESTWITHGPS_SINGLETON_H

#include <utility>
#include <functional>

// from http://ideone.com/Wh9cX9

template<typename T> // Singleton policy class
class Singleton {
protected:
	Singleton() = default;

	Singleton(const Singleton&) = delete;

	Singleton& operator=(const Singleton&) = delete;

	virtual ~Singleton() = default;

public:
	template<typename... Args>
	static T& instance(Args... args) // Singleton
	{
		//we pack our arguments in a T&() function...
		//the bind is there to avoid some gcc bug
		static auto onceFunction = std::bind(createInstanceInternal < Args... >, args... );
		//and we apply it once...
		return apply(onceFunction);
	}

private:

	//single instance so the static reference should be initialized only once
	//so the function passed in is called only the first time
	static T& apply(const std::function<T&()>& function){
		static T& instanceRef = function();
		return instanceRef;
	}

	template<typename... Args>
	static T& createInstanceInternal(Args... args){
		static T instance{std::forward<Args>(args)...};
		return instance;
	}

};

#endif //RANDOMFORESTWITHGPS_SINGLETON_H
