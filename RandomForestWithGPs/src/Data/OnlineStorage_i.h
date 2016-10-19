
#ifndef __INCLUDE_ONLINESTORAGE
#error "Don't include OnlineStorage_i.h directly. Include OnlineStorage.h instead."
#endif


template<typename T>
void OnlineStorage<T>::append(const T& data){
	m_internal.push_back(data);
	notify(APPEND);
}

template<typename T>
void OnlineStorage<T>::remove(const Iterator& it){
	m_internal.erase(it);
	notify(ERASE);
}

template<typename T>
void OnlineStorage<T>::append(const OnlineStorage<T>& storage){
	m_internal.push_back(storage.m_internal);
	notify(APPENDBLOCK);
}

template<typename T>
void OnlineStorage<T>::append(const std::vector<T>& storage){
	m_internal.push_back(storage.m_internal);
	notify(APPENDBLOCK);
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
