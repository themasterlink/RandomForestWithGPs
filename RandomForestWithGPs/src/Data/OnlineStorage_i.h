
#ifndef __INCLUDE_ONLINESTORAGE
#error "Don't include OnlineStorage_i.h directly. Include OnlineStorage.h instead."
#endif

#include "../Utility/Util.h"
#include "ClassKnowledge.h"
#include "../Base/Settings.h"
#include "DataConverter.h"

template<typename T>
PoolInfo<T>::PoolInfo(): m_desiredSizes(ClassKnowledge::instance().amountOfClasses(), 0),
						 m_currentSizes(ClassKnowledge::instance().amountOfClasses(), 0),
						 m_performance(ClassKnowledge::instance().amountOfClasses(), AvgNumber(NEG_REAL_MAX)),
						 m_totalAmountOfSavedPoints(0), m_amountOfPointsPerClass(0){}

template<typename T>
void PoolInfo<T>::changeAmountOfClasses(const unsigned int amountOfClasses){
	const unsigned int old = (unsigned int) m_desiredSizes.size();
	m_desiredSizes.resize(amountOfClasses);
	m_currentSizes.resize(amountOfClasses);
	m_performance.resize(amountOfClasses);
	for(unsigned int i = old; i < amountOfClasses; ++i){
		m_currentSizes[i] = 0;
		m_performance[i] = AvgNumber(NEG_REAL_MAX);
	}
	m_amountOfPointsPerClass = m_totalAmountOfSavedPoints / amountOfClasses;
	for(unsigned int i = 0; i < amountOfClasses; ++i){
		m_desiredSizes[i] = m_amountOfPointsPerClass;
	}
}

template<typename T>
void PoolInfo<T>::setMaxNumberOfSavedPoints(const unsigned int maxNr){
	m_totalAmountOfSavedPoints = maxNr;
	const unsigned int amountOfClasses = ClassKnowledge::instance().amountOfClasses();
	if(amountOfClasses > 0){
		m_amountOfPointsPerClass = m_totalAmountOfSavedPoints / amountOfClasses;
		for(unsigned int i = 0; i < amountOfClasses; ++i){
			m_desiredSizes[i] = m_amountOfPointsPerClass;
		}
	}else{
		printError("The amount of classes is zero!");
	}
}

template<typename T>
bool PoolInfo<T>::checkIfPointShouldBeAdded(const T& data){
	const auto label = data->getLabel();
	return m_desiredSizes[label] > m_currentSizes[label];
}

template<typename T>
unsigned int PoolInfo<T>::getDifferenceForClass(const unsigned int classNr){
	return m_currentSizes[classNr] - m_desiredSizes[classNr];
}

template<typename T>
unsigned int PoolInfo<T>::getClassWherePointShouldBeRemoved(){
	for(unsigned int i = 0, end = (unsigned int) m_desiredSizes.size(); i < end; ++i){
		if(m_desiredSizes[i] < m_currentSizes[i]){
			return i;
		}
	}
	return UNDEF_CLASS_LABEL;
}

template<typename T>
inline void PoolInfo<T>::addPointToClass(const unsigned int classNr){
	++m_currentSizes[classNr];
}

template<typename T>
inline void PoolInfo<T>::removePointsFromClass(unsigned int classNr, unsigned int amount){
	m_currentSizes[classNr] -= amount;
}

template<typename T>
void PoolInfo<T>::updateAccordingToPerformance(){
	Real min, max;
	const auto ignoreRealNegMax = true; // avoids the classes which have no size at the moment
	DataConverter::getMinMax(m_performance, min, max, &AvgNumber::mean, ignoreRealNegMax);
	if(min < max){
		const auto amountOfFixedPoints = (unsigned int) (0.8 * m_amountOfPointsPerClass);
		auto addedError = 0._r;
		for(unsigned int i = 0, end = (unsigned int) m_desiredSizes.size(); i < end; ++i){
			const auto val = m_performance[i].mean();
			if(val >= min){ // exclude NEG_REAL_MAX
				addedError += 1._r - val;
			}
		}
		const auto amountOfTotalSharedPoints = (m_amountOfPointsPerClass - amountOfFixedPoints) * m_performance.size();
		for(unsigned int i = 0, end = (unsigned int) m_desiredSizes.size(); i < end; ++i){
			const auto val = m_performance[i].mean();
			if(val >= min){ // exclude NEG_REAL_MAX
				const auto fac = (1._r - val) / addedError;
				m_desiredSizes[i] = (unsigned int) (amountOfFixedPoints + (Real) amountOfTotalSharedPoints * fac);
				printOnScreen("Class: " << i << ", performance: " << val
										<< ", new size: " << m_desiredSizes[i]
										<< ", current size: " << m_currentSizes[i]);
			}
		}
//		const auto fac = 1.0_r - (m_performance[i] - min) / (max - min);
//

		printOnScreen("Min: " << min << ", max: " << max);
	}else{
		printError("Min and max are equal!");
	}
}

template<typename T>
void PoolInfo<T>::update(Subject* caller, unsigned int event){
	if(caller){
		if(caller->classType() == ClassTypeSubject::ONLINERANDOMFOREST){
			if(event == OnlineStorage<T>::UPDATE_POOL_ACCORDING_TO_PERFORMANCE){
				updateAccordingToPerformance();
			}
			// other event types are not direct at the pool info
		}else{
			printError("This type is not supported here!");
		}
	}else{
		printError("The subject caller was zero!");
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
		printError("This storage type is not supported here!");
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
			const auto label = data->getLabel();
			m_multiInternal[label].emplace_back(data);
			m_poolInfo.addPointToClass(label);
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
	appendInternal(storage, false);
}

template<typename T>
void OnlineStorage<T>::appendUnique(const std::vector<T>& data){
	appendInternal(data, m_lastUpdateIndex != 0);
}

template<typename T>
void OnlineStorage<T>::appendInternal(const std::vector<T>& data, const bool shouldBeAddedUnique){
	m_lastUpdateIndex = size();
	if(m_storageMode == StorageMode::NORMAL){
		m_internal.reserve(m_lastUpdateIndex + data.size());
		if(shouldBeAddedUnique){
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
		}else{
			m_internal.insert(m_internal.end(), data.begin(), data.end());
		}
		notify(static_cast<const unsigned int>(Event::APPENDBLOCK));
	}else if(m_storageMode == StorageMode::POOL){
		for(const auto& p : data){
			bool found = false;
			if(shouldBeAddedUnique){
				for(const auto& existingPoint : m_internal){
					if(existingPoint == p){
						found = true;
						break;
					}
				}
			}
			if(!found){
				m_multiInternal[p->getLabel()].emplace_back(p);
				m_poolInfo.addPointToClass(p->getLabel());
			}
		}
		unsigned int next = m_poolInfo.getClassWherePointShouldBeRemoved();
		unsigned int amountOfRemovedPoints = 0;
		while(next != UNDEF_CLASS_LABEL){
			const auto diff = m_poolInfo.getDifferenceForClass(next);
			if(diff > 0){
				auto& actInternal = m_multiInternal[next];
				for(unsigned int k = 0, endNext = actInternal.size(); k + diff < endNext; ++k){
					actInternal[k] = actInternal[k + diff];
				}
				actInternal.resize(std::max(0u, (unsigned int) actInternal.size() - diff));
				m_poolInfo.removePointsFromClass(next, diff);
				amountOfRemovedPoints += diff;
			}else{
				printError("The diff is negativ!");
			}
			next = m_poolInfo.getClassWherePointShouldBeRemoved();
		}
		std::stringstream str2;
		str2 << "Added " << data.size() << " points to the online storage";
		if(amountOfRemovedPoints > 0){
			str2 << " and removed " << amountOfRemovedPoints << " old points";
		}
		printOnScreen(str2.str());
		copyMultiInternalInInternal();
		notify(static_cast<const unsigned int>(Event::APPENDBLOCK));
	}else{
		printError("This type is not supported!");
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
	return m_poolInfo.checkIfPointShouldBeAdded(data);
}

template<typename T>
void OnlineStorage<T>::copyMultiInternalInInternal(){
	if(m_storageMode == StorageMode::POOL){
		m_internal.clear();
		typename OnlineStorage<T>::MultiClassInternalStorage::size_type amount = 0;
		for(auto& storage : m_multiInternal){
			amount += storage.size();
		}
		m_internal.reserve(amount);
		for(const auto& storage : m_multiInternal){
			for(const auto& point : storage){
				m_internal.emplace_back(point);
			}
		}
		printOnScreen("Current size of the pool: " << m_internal.size());
	}else{
		printError("This type is not defined here!");
	}
}

template<typename T>
void OnlineStorage<T>::update(Subject* caller, unsigned int event){
	if(event == ClassKnowledge::Caller::NEW_CLASS){
		if(m_storageMode == StorageMode::POOL){
			const auto newAmountOfClasses = ClassKnowledge::instance().amountOfClasses();
			const auto oldAmount = m_multiInternal.size();
			if(newAmountOfClasses < oldAmount){
				printError("The amount of classes was reduced, something went wrong!");
			}
			m_poolInfo.changeAmountOfClasses(newAmountOfClasses);
			m_multiInternal.resize(newAmountOfClasses);
			const auto nr = Settings::instance().getDirectValue<unsigned int>(
					"OnlineRandomForest.maxAmountOfPointsSavedInPool");
			m_poolInfo.setMaxNumberOfSavedPoints(nr);
		}
	}else{
		printError("This update is not possible on an OnlineStorage!");
	}
}

template<typename T>
void OnlineStorage<T>::setStorageModeToPoolBase(){
	m_storageMode = StorageMode::POOL;
	ClassKnowledge::instance().attach(this);
}
