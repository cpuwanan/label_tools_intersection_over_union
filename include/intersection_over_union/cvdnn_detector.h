#ifndef CVDNN_DETECTOR_H
#define CVDNN_DETECTOR_H

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <map>
#include <opencv4/opencv2/opencv.hpp>
#include <opencv4/opencv2/dnn.hpp>
#include "common.h"

namespace cvdnn_detector {
	std::vector<std::string> getNetModelOutputsNames(const cv::dnn::Net &net);
};

class Detector {
public:
	Detector();
	~Detector();
	void init(
		int net_width, 
		int net_height, 
		std::string weights_file, 
		std::string cfg_file, 
		std::map<int, std::string> classnames,
		double confidence_threshold,
		double nms_threshold
	);
	char detect(MyImageInfo &item, cv::Mat &dst);
private:
	cv::dnn::Net net_;
	std::map<int, std::string> classnames_;
	std::map<int, cv::Scalar> colors_;
	cv::Scalar mean_;
	double scale_;
	cv::Size net_size_;
	double conf_thr_, nms_thr_;
};

#endif
