#ifndef COMMON_H
#define COMMON_H

#include <opencv2/opencv.hpp>
#include <iostream>

struct MyBox {
	int id = -1;
	int cx = -1;
	int cy = -1;
	cv::Rect box;
	
	bool isOk() {
		return cx >= 0 && cy >= 0 && id >= 0;
	}
};

struct MyImageInfo {
	cv::Mat image;
	std::string name = "";
	std::string path = "";
	std::vector<MyBox> labels;
	std::vector<MyBox> detections;
};

#endif
