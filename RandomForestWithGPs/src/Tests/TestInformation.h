//
// Created by denn_ma on 5/31/17.
//

#ifndef RANDOMFORESTWITHGPS_TESTINFORMATION_H
#define RANDOMFORESTWITHGPS_TESTINFORMATION_H

#include <string>
#include "../Utility/Util.h"
#include "../Data/OnlineStorage.h"
#include "../RandomForests/OnlineRandomForest.h"

enum class TestMode {
	LOAD = 0,
	DEFINE,
	REMOVE,
	TRAIN,
	UPDATE,
	TEST,
	COMBINE,
	UNDEFINED
};

//using Words = std::vector<std::string>;

class TestInformation {
public:

	static const std::string trainSettingName;
	static const std::string testSettingName;

	struct Instruction {
		using ExitMode = OnlineRandomForest::TrainingsConfig::TrainingsMode;

		Instruction(const TestMode mode, const std::string& varName):
				m_mode(mode), m_varName(varName), m_exitMode(ExitMode::UNDEFINED),
				m_seconds((Real) 0.0), m_amountOfTrees(0), m_memory(0){};

		void processLine(const std::vector<std::string>& line);

		void addType(const std::string& preposition, const std::string& nrType);

		TestMode m_mode;
		std::string m_varName;
		ExitMode m_exitMode;
		Real m_seconds;
		unsigned int m_amountOfTrees;
		unsigned int m_memory;
	};

	struct TestDefineName {
		std::string m_varName;
		bool m_withClasses; // is true if classes are used, false if they are not used
		std::string m_firstFromVariable;
		std::string m_secondFromVariable;
		std::vector<unsigned int> m_classes;

		bool isTrainOrTestSetting();

		void useAllClasses();
	};

	TestInformation();

	void addDefinitionOrInstruction(const TestMode mode, const std::string& def);

	TestDefineName getDefinition(const std::string& name);

	Instruction getInstruction(const unsigned int i);

	bool decipherClasses(const std::vector<std::string>& words, const unsigned int start,
						 const unsigned int end, std::vector<unsigned int>& usedClasses);

private:

	std::map<std::string, TestDefineName> m_definitions;

	std::vector<Instruction> m_instructions;

};

inline
std::ostream& operator<<(std::ostream& stream, const TestInformation::TestDefineName& testDef){
	stream << testDef.m_varName << ", " << testDef.m_withClasses << ": (" << testDef.m_firstFromVariable << ", " << testDef.m_secondFromVariable << ")";
	return stream;
}

#endif //RANDOMFORESTWITHGPS_TESTINFORMATION_H
