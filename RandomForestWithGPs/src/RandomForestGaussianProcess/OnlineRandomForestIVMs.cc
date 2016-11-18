/*
 * OnlineRandomForestIVMs.cc
 *
 *  Created on: 17.11.2016
 *      Author: Max
 */

#include "OnlineRandomForestIVMs.h"

OnlineRandomForestIVMs::OnlineRandomForestIVMs() {
	// TODO Auto-generated constructor stub
	// removes orf to avoid that the update is called directly to the orf
//	m_orf.getStorageRef().deattach(&m_orf);
//	// instead call the update on the
//	m_orf.getStorageRef().attach(this);

}

OnlineRandomForestIVMs::~OnlineRandomForestIVMs() {
	// TODO Auto-generated destructor stub
}

