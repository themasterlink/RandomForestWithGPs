/*
 * ColorConverter.h
 *
 *  Created on: 28.09.2016
 *      Author: Max
 */

#ifndef UTILITY_COLORCONVERTER_H_
#define UTILITY_COLORCONVERTER_H_

#include "Util.h"

namespace ColorConverter{

void HSV2RGB(const double h, const double s, const double v, double& r, double &g, double& b){
	const double c = v * s;
	const double x = c * (1. - fabs(fmod(h / 60., 2.) - 1.));
	const double m = v - c;
	r = g = b = 0;
	if(0. <= h && h < 60.){
		r = c + m;
		g = x + m;
	}else if(60. <= h && h < 120.){
		r = x + m;
		g = c + m;
	}else if(120. <= h && h < 180.){
		g = c + m;
		b = x + m;
	}else if(180. <= h && h < 240.){
		g = x + m;
		b = c + m;
	}else if(240. <= h && h < 300.){
		r = x + m;
		b = c + m;
	}else if(300. <= h && h <= 360.){
		r = c + m;
		b = x + m;
	}else{
		printError("The h value have to be between 0 and 360!");
	}
}

}



#endif /* UTILITY_COLORCONVERTER_H_ */
