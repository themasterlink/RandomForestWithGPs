
#ifndef __INCLUDE_ONLINESTORAGE
#error "Don't include OnlineStorage_i.h directly. Include OnlineStorage.h instead."
#endif

#include "../Utility/Util.h"
#include "ClassKnowledge.h"

template<typename T>
PoolInfo<T>::PoolInfo(): m_desiredSizes(ClassKnowledge::instance().amountOfClasses(), 0),
						 m_performance(ClassKnowledge::instance().amountOfClasses(), (Real) 0.0){}

template<typename T>
void PoolInfo<T>::changeAmountOfClasses(const unsigned int amountOfClasses){
	const unsigned int old = (unsigned int) m_desiredSizes.size();
	m_desiredSizes.resize(amountOfClasses);
	m_performance.resize(amountOfClasses);
	for(unsigned int i = old; i < amountOfClasses; ++i){
		m_desiredSizes[i] = 0;
		m_performance[i] = (Real) 0.0;
	}
}

template<typename T>
OnlineStorage<T>::OnlineStorage(): m_lastUpdateIndex(0), m_storageMode(StorageMode::NORMAL){
}


template<typename T>
OnlineStorage<T>::OnlineStorage(OnlineStorage<T>& storage): m_lastUpdateIndex(storage.m_lastUpdateIndex){
	if(m_storageMode == StorageMode::NORMAL){
		m_internal.reserve(storage.size());
		m_internal.insert(m_internal.end(), storage.begin(), storage.end());
	}else if(m_storageMode == StorageMode::POOL){

	}else{
		printError("This storage type is not supported here!");
	}
}

template<typename T>
OnlineStorage<T>::~OnlineStorage(){
}

template<typename T>
void OnlineStorage<T>::append(const T& data){
	if(m_storageMode == StorageMode::NORMAL){
		m_lastUpdateIndex = size();
		m_internal.emplace_back(data);
		notify(static_cast<const unsigned int >(Event::APPEND));
	}else if(m_storageMode == StorageMode::POOL){
		printWarning("The pool mode should not be used with the single append function!");
		if(checkIfPointShouldBeAdded(data)){
			m_lastUpdateIndex = size();
			m_multiInternal[data->getLabel()].emplace_back(data);
			copyMultiInternalInInternal();
			notify(static_cast<const unsigned int >(Event::APPEND));
		}
	}else{
		printError("This storage type is not supported here!");
	}
}

template<typename T>
void OnlineStorage<T>::remove(const Iterator& it){
	if(m_storageMode == StorageMode::NORMAL){
		m_internal.erase(it);
		notify(static_cast<const unsigned int >(Event::ERASE));
	}else{
		printError("Remove not implemented for this type");
	}
}

template<typename T>
void OnlineStorage<T>::append(const OnlineStorage<T>& storage){
	if(m_storageMode == StorageMode::NORMAL && storage.m_storageMode == StorageMode::NORMAL){
		append(storage.m_internal); // just calls the internal append
	}else{
		printError("Not implemented yet!");
	}
}

template<typename T>
void OnlineStorage<T>::append(const std::vector<T>& storage){
	if(m_storageMode == StorageMode::NORMAL){
		m_lastUpdateIndex = size();
		m_internal.reserve(m_internal.size() + storage.size());
		m_internal.insert(m_internal.end(), storage.begin(), storage.end());
		notify(static_cast<const unsigned int>(Event::APPENDBLOCK));
	}else if(m_storageMode == StorageMode::POOL){
		for(auto& data : storage){
			if(checkIfPointShouldBeAdded(data)){
				m_lastUpdateIndex = size();
				m_multiInternal[data->getLabel()].emplace_back(data);
			}
		}
		copyMultiInternalInInternal();
	}else{
		printError("This type is not supported!");
	}
}

template<typename T>
void OnlineStorage<T>::appendUnique(const std::vector<T>& data){
	m_lastUpdateIndex = size();
	if(m_lastUpdateIndex == 0){ // first append
		append(data);
	}else{
		if(m_storageMode == StorageMode::NORMAL){
			m_internal.reserve(m_lastUpdateIndex + data.size());
			for(const auto& p : data){
				bool found = false;
				for(const auto& existingPoint : m_internal){
					if(existingPoint == p){
						found = true;
						break;
					}
				}
				if(!found){
					m_internal.emplace_back(p);
				}
			}
			notify(static_cast<const unsigned int>(Event::APPENDBLOCK));
		}else if(m_storageMode == StorageMode::POOL){
			std::vector<unsigned int> usedData;
			unsigned i = 0;
			for(const auto& p : data){
				bool found = false;
				for(const auto& existingPoint : m_internal){
					if(existingPoint == p){
						found = true;
						break;
					}
				}
				if(!found){
					usedData.emplace_back(i);
				}
				++i;
			}
			m_internal.reserve(m_lastUpdateIndex + data.size());
			for(auto& index : usedData){
				auto& point = data[index];
				if(checkIfPointShouldBeAdded(point)){
					m_lastUpdateIndex = size();
					m_multiInternal[point->getLabel()].emplace_back(point);
				}
			}
			copyMultiInternalInInternal();
			notify(static_cast<const unsigned int>(Event::APPENDBLOCK));
		}else{
			printError("This type is not supported!");
		}
	}
}

template<typename T>
unsigned int OnlineStorage<T>::dim() const{
	if(size() > 0){
		return m_internal.front()->rows();
	}else{
		return 0;
	}
}

template<typename T>
T& OnlineStorage<T>::operator[](int element){
	return m_internal[element];
}

template<typename T>
const T& OnlineStorage<T>::operator[](int element) const{
	return m_internal[element];
}

template<typename T>
typename OnlineStorage<T>::Iterator OnlineStorage<T>::begin(){
	return m_internal.begin();
}

template<typename T>
typename OnlineStorage<T>::Iterator OnlineStorage<T>::end(){
	return m_internal.end();
}

template<typename T>
typename OnlineStorage<T>::ConstIterator OnlineStorage<T>::begin() const{
	return m_internal.cbegin();
}

template<typename T>
typename OnlineStorage<T>::ConstIterator OnlineStorage<T>::end() const{
	return m_internal.cend();
}

template<typename T>
typename OnlineStorage<T>::ConstIterator OnlineStorage<T>::cbegin() const{
	return m_internal.cbegin();
}

template<typename T>
typename OnlineStorage<T>::ConstIterator OnlineStorage<T>::cend() const{
	return m_internal.cend();
}

template<typename T>
T& OnlineStorage<T>::first(){
	return m_internal.front();
}

template<typename T>
T& OnlineStorage<T>::last(){
	return m_internal.back();
}

template<typename T>
typename OnlineStorage<T>::InternalStorage& OnlineStorage<T>::storage(){
	return m_internal;
}

template<typename T>
const typename OnlineStorage<T>::InternalStorage& OnlineStorage<T>::storage() const{
	return m_internal;
}

template<typename T>
unsigned int OnlineStorage<T>::getLastUpdateIndex() const{
	return m_lastUpdateIndex;
}

template<typename T>
unsigned int OnlineStorage<T>::getAmountOfNew() const{
	return size() - m_lastUpdateIndex;
}

template<typename T>
ClassTypeSubject OnlineStorage<T>::classType() const{
	return ClassTypeSubject::ONLINESTORAGE;
}

template<typename T>
bool OnlineStorage<T>::checkIfPointShouldBeAdded(const T& data){
	if(m_storageMode == StorageMode::POOL){
	}
	return true;
}

template<typename T>
void OnlineStorage<T>::copyMultiInternalInInternal(){
	if(m_storageMode == StorageMode::POOL){
		m_internal.clear();
		typename OnlineStorage<T>::MultiClassInternalStorage::size_type amount;
		for(auto& storage : m_multiInternal){
			amount += storage.size();
		}
		m_internal.resize(amount);
		for(auto& storage : m_multiInternal){
			for(auto& point : storage){
				m_internal.emplace_back(point);
			}
		}
	}else{
		printError("This type is not defined here!");
	}
}

template<typename T>
void OnlineStorage<T>::update(Subject* caller, unsigned int event){
	if(caller == ClassKnowledge::instance().getCaller() && event == ClassKnowledge::Caller::NEW_CLASS){
		const auto newAmountOfClasses = ClassKnowledge::instance().amountOfClasses();
		const auto oldAmount = m_multiInternal.size();
		if(newAmountOfClasses < oldAmount){
			printError("The amount of classes was reduced, something went wrong!");
		}
		m_poolInfo.changeAmountOfClasses(newAmountOfClasses);
		m_multiInternal.resize(newAmountOfClasses);
	}else{
		printError("This update is not possible on an OnlineStorage!");
	}
}

template<typename T>
void OnlineStorage<T>::setStorageModeToPoolBase(){
	m_storageMode = StorageMode::POOL;
	attach(ClassKnowledge::instance().getCaller());
}
