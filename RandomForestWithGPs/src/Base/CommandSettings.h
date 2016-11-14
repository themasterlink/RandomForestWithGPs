/*
 * CommandSettings.h
 *
 *  Created on: 27.10.2016
 *      Author: Max
 */

#ifndef BASE_COMMANDSETTINGS_H_
#define BASE_COMMANDSETTINGS_H_

#include <boost/program_options.hpp>
#include <list>

#define MEMBER_PARAM(param) m_##param

/** adds a parameter as member with description, default value*/
#define ADD_PARAM(Type, param, defVal, descr) \
	static Type MEMBER_PARAM(param);  \
	public: ADD_PARAM_INFO(Type, param, defVal, descr)

/** adds a parameter as member with description, default value.*/
#define ADD_PARAM_INFO(Type, param, defVal, descr) \
	static Type defaultvalue_##param(){return defVal;} \
	static Type get_##param(){return MEMBER_PARAM(param);} const \
	static std::string description_##param(){return std::string( descr );} \
	static std::string basename_##param(){return #param;} const \
	static std::string typename_##param(){return #Type;}

#define DEFINE_PARAM(Type, param) \
	Type CommandSettings::MEMBER_PARAM(param)(CommandSettings::defaultvalue_##param())

#define INIT_PARAM(Type, param) \
	m_params.push_back(Param(#param, &typename_##param, (void*) &MEMBER_PARAM(param)))

struct Param {
	Param(std::string name, const std::string (*type)(), void* ref);

	std::string name;
	const std::string (*type)();
	void* ref;
};

/* Always ADD, DEFINE and INIT, for adding new params
 */

class CommandSettings {
public:

	ADD_PARAM(bool, useFakeData, false, "Uses fake data for the test");
	ADD_PARAM(int, visuRes, 0, "If possible visualize the data, zero means no visualization");
	ADD_PARAM(int, visuResSimple, 0, "If possible visualize the data in a simple manner, zero means no visualization");
	ADD_PARAM(bool, onlyDataView, false, "Only visualizes the data, without any training");
	ADD_PARAM(double, samplingAndTraining, false, "The training and sampling is performed, if the value in seconds is bigger than 0.");
	ADD_PARAM(bool, plotHistos, false, "Should some histogramms be plotted");

	static void init();

	static void setValues(boost::program_options::variables_map& vm);

private:
	static std::list<Param> m_params;

	CommandSettings(){};
	virtual ~CommandSettings(){};
};


#endif /* BASE_COMMANDSETTINGS_H_ */
