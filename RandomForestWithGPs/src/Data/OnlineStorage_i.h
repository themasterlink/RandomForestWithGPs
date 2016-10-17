
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
	for(ConstIterator it = storage.m_internal.begin(); it != storage.m_internal.end(); ++it){
		m_internal.push_back(*it);
	}
	notify(APPENDBLOCK);
}


