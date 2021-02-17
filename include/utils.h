#ifndef CPP_HELPERS_UTILS_H
#define CPP_HELPERS_UTILS_H

#include <iostream>
#include <string>
#include <vector>

enum TextType {INFO, SUCCESS, WARNING, DANGER, INFO_B, SUCCESS_B, WARNING_B, DANGER_B};

namespace utils {
	std::string colorText(int state, std::string text);
	
	void findFiles(const std::string path, std::string filter, std::vector<std::string> &outputs);
	
	std::string getStrId(int id, int N, char prefix = '0');
	
	bool isValidPath(std::string path);
	
	char nonBlockingKeyboardEvent();
};

#endif

