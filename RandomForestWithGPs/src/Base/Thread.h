//
// Created by denn_ma on 7/7/17.
//

#ifndef RANDOMFORESTWITHGPS_THREAD_H_H
#define RANDOMFORESTWITHGPS_THREAD_H_H


#include <thread>
#include <vector>
#include "BaseType.h"

using Thread = std::thread;

class ThreadGroup {
public:

	ThreadGroup() = default;
	ThreadGroup(unsigned int size){ m_threads.reserve(size); };
	~ThreadGroup() = default;

	void addThread(UniquePtr<Thread> thread){
		m_threads.emplace_back(std::move(thread));
	}

	void joinAll(){
		for(auto& thread : m_threads){
			if(thread->joinable()){
				thread->join();
			}
		}
	}

private:

	std::vector<UniquePtr<Thread> > m_threads;

};

template<typename... Args>
std::unique_ptr<Thread> makeThread(Args&&... args)
{
	return std::make_unique<Thread>(std::forward<Args>(args)...);
}

#endif //RANDOMFORESTWITHGPS_THREAD_H_H
