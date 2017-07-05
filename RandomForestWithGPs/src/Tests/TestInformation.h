//
// Created by denn_ma on 5/31/17.
//

#ifndef RANDOMFORESTWITHGPS_TESTINFORMATION_H
#define RANDOMFORESTWITHGPS_TESTINFORMATION_H

#include <string>
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
	SPLIT,
	FOR,
	END_FOR,
	UNDEFINED
};

//using Words = std::vector<std::string>;

class TestInformation {
public:

	static const std::string trainSettingName;
	static const std::string testSettingName;

	class Instruction {
	public:
		using ExitMode = OnlineRandomForest::TrainingsConfig::TrainingsMode;

		Instruction(const TestMode mode, const std::string& varName, Instruction* scope):
				m_mode(mode), m_varName(varName), m_exitMode(ExitMode::UNDEFINED),
				m_seconds((Real) 0.0), m_amountOfTrees(0), m_memory(0), m_scope(scope), m_currentValue(0){};

		void processLine(const std::vector<std::string>& line);

		void addType(const std::string& preposition, const std::string& nrType);

		TestMode m_mode;
		std::string m_varName;
		ExitMode m_exitMode;
		Real m_seconds;
		unsigned int m_amountOfTrees;
		MemoryType m_memory;
		Instruction* m_scope;
		std::vector<unsigned int> m_range;
		unsigned int m_currentValue;

		bool operator==(const Instruction& instruction) const;

		void replaceScopeVariables(std::string& definition);
	};

	class TestDefineName {
	public:
		TestDefineName(): m_includeClasses(true), m_splitAmount(-1){};

		std::string getVarName() const;

		bool isTrainOrTestSetting() const;

		void useAllClasses();

		void setVarName(std::string name);

		bool m_includeClasses; // is true if classes are included, false if they are excluded
		std::string m_firstFromVariable;
		std::string m_secondFromVariable;
		int m_splitAmount;
		std::vector<unsigned int> m_classes;

	private: // avoid direct access to var name
		std::string m_varName;
	};

	TestInformation();

	void addDefinitionOrInstruction(const TestMode mode, const std::string& def);

	TestDefineName getDefinition(const std::string& name, Instruction* currentScope);

	Instruction getInstruction(const unsigned int i);

	unsigned int getInstructionNr(const Instruction& instruction);

	int getSplitNumber(const std::string& name);

private:

	std::map<std::string, TestDefineName> m_definitions;

	std::vector<Instruction> m_instructions;

	Instruction* m_currentScope;
};

inline
std::ostream& operator<<(std::ostream& stream, const TestInformation::TestDefineName& testDef){
	stream << testDef.getVarName() << ", " << testDef.m_includeClasses << ": (" << testDef.m_firstFromVariable << ", " << testDef.m_secondFromVariable << ")";
	return stream;
}

#endif //RANDOMFORESTWITHGPS_TESTINFORMATION_H
