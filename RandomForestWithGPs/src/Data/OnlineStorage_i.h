
#ifndef __INCLUDE_ONLINESTORAGE
#error "Don't include OnlineStorage_i.h directly. Include OnlineStorage.h instead."
#endif

#include "../Utility/Util.h"

template<typename T>
OnlineStorage<T>::OnlineStorage(): m_lastUpdateIndex(0){
}


template<typename T>
OnlineStorage<T>::OnlineStorage(OnlineStorage<T>& storage): m_lastUpdateIndex(storage.m_lastUpdateIndex){
	m_internal.reserve(storage.size());
	m_internal.insert(m_internal.end(), storage.begin(), storage.end());
}

template<typename T>
OnlineStorage<T>::~OnlineStorage(){
}

template<typename T>
void OnlineStorage<T>::append(const T& data){
	m_lastUpdateIndex = size();
	m_internal.push_back(data);
	notify(static_cast<const unsigned int >(Event::APPEND));
}

template<typename T>
void OnlineStorage<T>::remove(const Iterator& it){
	m_internal.erase(it);
	notify(static_cast<const unsigned int >(Event::ERASE));
}

template<typename T>
void OnlineStorage<T>::append(const OnlineStorage<T>& storage){
	append(storage.m_internal); // just calls the internal append
}

template<typename T>
void OnlineStorage<T>::append(const std::vector<T>& storage){
	m_lastUpdateIndex = size();
	m_internal.reserve(m_internal.size() + storage.size());
	m_internal.insert(m_internal.end(), storage.begin(), storage.end());
	notify(static_cast<const unsigned int >(Event::APPENDBLOCK));
}

template<typename T>
void OnlineStorage<T>::resize(const unsigned int size){
	m_internal.resize(size);
	if(this->size() < m_lastUpdateIndex){
		m_lastUpdateIndex = this->size();
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
	return m_internal.begin();
}

template<typename T>
typename OnlineStorage<T>::ConstIterator OnlineStorage<T>::end() const{
	return m_internal.end();
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
unsigned int OnlineStorage<T>::getLastUpdateIndex(){
	return m_lastUpdateIndex;
}

template<typename T>
ClassTypeSubject OnlineStorage<T>::classType() const{
	return ClassTypeSubject::ONLINESTORAGE;
}
