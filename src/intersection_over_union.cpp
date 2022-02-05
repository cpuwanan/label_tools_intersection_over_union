#include <iostream>
#include <yaml-cpp/yaml.h>
#include <opencv2/opencv.hpp>

#include "utils.h"
#include "intersection_over_union/common.h"
#include "intersection_over_union/cvdnn_detector.h"

namespace my_utils {
	MyBox getValue(std::string text, std::string key, cv::Size image_size) {
		std::size_t found = text.find(key);
		std::vector<double> values;
		while (found != std::string::npos) {
			double value;
			std::stringstream ss;
			ss << text.substr(0, int(found));
			ss >> value;
			values.push_back(value);
			int length = text.size() - int(found) - int(key.size());
			text = text.substr(int(found) + key.size(), length);
			found = text.find(key);
		}
		
		if (text != "") {
			double value;
			std::stringstream ss;
			ss << text.substr(0, int(found));
			ss >> value;
			values.push_back(value);
		}

		MyBox box;
		if (values.size() == 5) {
			enum {ID, X, Y, W, H};
			int cx = int(values[X] * image_size.width);
			int cy = int(values[Y] * image_size.height);
			int width = int(values[W] * image_size.width);
			int height = int(values[H] * image_size.height);
			box.id = int(values[ID]);
			box.cx = cx;
			box.cy = cy;
			box.box = cv::Rect(
				cx - width / 2,
				cy - height / 2,
				width,
				height
			);
		}
		return box;
	}
	
	cv::Rect overlappingRect(cv::Rect rect1, cv::Rect rect2) {
		int x1 = std::max(rect1.x, rect2.x);
		int y1 = std::max(rect1.y, rect2.y);
		int x2 = std::min(rect1.x + rect1.width , rect2.x + rect2.width);
		int y2 = std::min(rect1.y + rect1.height, rect2.y + rect2.height);
		return cv::Rect(x1, y1, x2 - x1, y2 - y1);
	}
	
	double unionRectArea(cv::Rect rect1, cv::Rect rect2, cv::Rect overlap) {
		return overlap.area() + (rect1.area() - overlap.area()) + (rect2.area() - overlap.area());
	}
}

class MyTools {
public:
	MyTools(std::string config_file) {
		is_ok_ = this->loadConfig(config_file);
	}
	
	bool loadConfig(std::string file) {
		if (!utils::isValidPath(file)) {
			std::cout << utils::colorText(TextType::DANGER_B, "Invalid file: " + file) << std::endl;
			return false;
		} else {
			std::cout << "Reading file: " << file << std::endl;
		}
		
		// ### Reading header
		YAML::Node node = YAML::LoadFile(file);
		std::string header = "iou";
		std::string header2 = "yolo";
		auto data = node[header];
		auto model = node[header2];
		
		if (!data) {
			std::cout << " " << utils::colorText(TextType::DANGER_B, cv::format("Cannot find param '%s'", header.c_str())) << std::endl;
			return false;
		}

		// ### Reading subfix
		std::string subfix = "image_root";
		if (!data[subfix]) {
			std::cout << " " << utils::colorText(TextType::DANGER_B, cv::format("Failed loading param '%s/%s'", header.c_str(), subfix.c_str())) << std::endl;
			return false;
		} else {
			image_root_ = data[subfix].as<std::string>();
			if (!utils::isValidPath(image_root_)) {
				std::cout << " " << utils::colorText(TextType::DANGER_B, cv::format("Invalid path: %s", image_root_.c_str())) << std::endl;
				return false;
			}
			std::cout << " Successfully read params: " << utils::colorText(TextType::SUCCESS_B, cv::format("%s/%s", header.c_str(), subfix.c_str())) << std::endl;
		}
		
		std::string image_filetype(".png");
		if (!data["image_filetype"]) {
			std::cout << " " << utils::colorText(TextType::DANGER_B, "Image filetype was not defined") << std::endl;
			return false;
		}
		image_filetype = data["image_filetype"].as<std::string>();
		std::cout << " Successfully set filetype: " << utils::colorText(TextType::SUCCESS_B, image_filetype) << std::endl;

		// ### Reading subfix
		subfix = "meta_data_file";
		if (!this->loadTestImageFilenames(data[subfix], image_filetype)) {
			std::cout << " " << utils::colorText(TextType::DANGER_B, cv::format("Failed loading param '%s/%s'", header.c_str(), subfix.c_str())) << std::endl;
			return false;
		} else {
			std::cout << " Successfully read params: " << utils::colorText(TextType::SUCCESS_B, cv::format("%s/%s", header.c_str(), subfix.c_str())) << std::endl;
		}
		
		// ### Reading subfix
		if (!this->loadAnnotations(data[subfix])) {
			std::cout << " " << utils::colorText(TextType::DANGER_B, cv::format("Failed loading param '%s/%s'", header.c_str(), subfix.c_str())) << std::endl;
			return false;
		} else {
			std::cout << " Successfully read params: " << utils::colorText(TextType::SUCCESS_B, cv::format("%s/%s", header.c_str(), subfix.c_str())) << std::endl;
		}
		
		if (!this->loadModel(model)) {
			std::cout << " " << utils::colorText(TextType::DANGER_B, cv::format("Failed loading param '%s/%s'", header.c_str(), subfix.c_str())) << std::endl;
			return false;
		} else {
			std::cout << " Successfully read params: " << utils::colorText(TextType::SUCCESS_B, cv::format("%s/%s", header.c_str(), subfix.c_str())) << std::endl;
		}
		
		return true;
	}
	
	void run() {
		if (!is_ok_) {
			std::cout << utils::colorText(TextType::DANGER_B, "Failed setting config") << std::endl;
			return;
		}
		
		int index = 0;
		int N = int(test_images_.size());
		bool is_quit = false;
		std::map<std::string, double> acc_list;
		int delay = 0;
		while(!is_quit) {
			MyImageInfo item = test_images_[index];			
			std::cout << " [" << index << "] " << item.name << std::endl;
			cv::Mat dst;
			detector_.detect(item, dst);
			double acc = this->computeIOU(item, dst);
			
			double scale = dst.cols /1000.0;
			if (scale > 0) {
				cv::resize(dst, dst, cv::Size(int(dst.cols / scale), int(dst.rows / scale)));
			}
			
			cv::imshow("IOU", dst);
			char key = cv::waitKey(delay);
			
			if (key == '1') {
				index = (index + 1) % N;
			} else if (key == '0') {
				index = (index - 1 + N) % N;
			} else if (key == 'q') {
				is_quit = true;
			} else if (key == 'r') {
				delay = (delay == 0) ? 10 : 0;
			}
			
			if (delay > 0) {
				if (index + 1 == N) {
					is_quit = true;
				} else {
					index = (index + 1) % N;
				}
			}
			
			std::map<std::string, double>::iterator it = acc_list.find(item.name);
			if (it == acc_list.end()) {
				acc_list.insert(std::pair<std::string, double>(item.name, acc));
			} else {
				it->second = acc;
			}
		}
		
		double total_accuracy = 0.0;
		std::map<std::string, double>::iterator it;
		for (it = acc_list.begin(); it != acc_list.end(); it++) {
			total_accuracy += it->second;
		}
		total_accuracy = total_accuracy / double(acc_list.size());
		std::cout << "\n----------------------------" << std::endl;
		std::cout << "Total accuracy of this model: " << total_accuracy << std::endl;
	}
	
private:
	
	double computeIOU(MyImageInfo item, cv::Mat &image) {
		int fontface = cv::FONT_HERSHEY_SIMPLEX;
    double fontscale = 0.5;
    int thickness = 1;
		
		if (item.detections.size() == 0 || item.labels.size() == 0) {
			std::cout << " -- Invalid box size. Labels: " << int(item.labels.size()) << ", Detected: " << int(item.detections.size()) << std::endl;
			cv::putText(image, "Invalid detections", cv::Point(10, image.rows - 10), fontface, fontscale, cv::Scalar(0, 0, 255), thickness);
			return false;
		}
		
		double total_accuracy = 0.0;
		int total_num = 0;
		
		for (int i=0; i<(int)item.labels.size(); i++) {
			if (!item.labels[i].isOk()) { continue; }
			int cx = item.labels[i].cx;
			int cy = item.labels[i].cy;
			int detected_id = -1;
			cv::Rect detected_box;
			for (int k=0; k<(int)item.detections.size() && detected_id == -1; k++) {
				auto detected = item.detections[k].box;
				if (cx >= detected.x && cx < detected.x + detected.width 
						&& cy >= detected.y && cy < detected.y + detected.height
				) {
					detected_id = item.detections[k].id;
					detected_box = item.detections[k].box;
				}
			}
			
			if (detected_id >= 0) {
				auto defined_box = item.labels[i].box;
				
				cv::Rect overlap_rect = my_utils::overlappingRect(defined_box, detected_box);
				// cv::rectangle(image, overlap_rect, cv::Scalar(0, 255, 255), -1);
				
				double union_area = my_utils::unionRectArea(defined_box, detected_box, overlap_rect);
				double accuracy = double(overlap_rect.area()) / union_area;
				accuracy = (detected_id == item.labels[i].id) ? accuracy : 0.0;
				
				total_accuracy += accuracy;
				total_num += 1;
				
				cv::rectangle(image, defined_box, cv::Scalar(0, 0, 255), 1);
				cv::putText(image, cv::format("[%d] Acc: %.2lf", item.labels[i].id, accuracy), cv::Point(defined_box.x, defined_box.y + defined_box.height - 5), fontface, fontscale, cv::Scalar(0, 0, 255), thickness);
				
				/*
				std::cout << "Rects: ";
				std::cout << " -- (" << detected_box.x << ", " << detected_box.y << ", " << detected_box.width << ", " << detected_box.height 
				<< "), (" << defined_box.x << ", " << defined_box.y << ", " << defined_box.width << ", " << defined_box.height << ")"
				<< std::endl;
				*/
			}
		}
		
		if (total_num > 0) {
			total_accuracy = total_accuracy / (double)total_num;
			cv::putText(image, cv::format( "Prediction accuracy: %.2lf", total_accuracy), cv::Point(10, image.rows - 10), fontface, 0.8, cv::Scalar(0, 0, 255), 2);
		}
		std::cout << "    Accuracy: " << utils::colorText(TextType::SUCCESS_B, cv::format("%.3lf", total_accuracy)) << std::endl;
		
		return total_accuracy;
	}
	
	bool loadTestImageFilenames(YAML::Node node, std::string image_filetype) {
				
		if (image_root_ == "") {
			std::cout << " -- " << utils::colorText(TextType::DANGER_B, "image_root_ cannot be empty") << std::endl;
			return false;
		}
		
		std::string path = node.as<std::string>();
		if (!utils::isValidPath(path)) {
			std::cout << " |-- Invalid path: " << utils::colorText(TextType::DANGER_B, path) << std::endl;
			return false;
		}
		
		std::ifstream reader;
		reader.open(path);
		std::cout << "Reading a file " << utils::colorText(TextType::SUCCESS_B, path) << std::endl;
		std::string filename = "";
		if (reader.is_open()) {
			std::string line;
			std::string key1 = "= ";
			std::string key2 = "test.txt";
			while (std::getline(reader, line) && filename == "") {
				std::size_t found1 = line.find(key1);
				std::size_t found2 = line.find(key2);
				if (found1 != std::string::npos && found2 != std::string::npos) {
					int length = line.size() - int(found1) - key1.size();
					filename = line.substr(int(found1) + key1.size(), length);
					int length2 = length - key2.size();
					test_file_prefix_ = line.substr(int(found1) + key1.size(), length2);
				}
			}
			reader.close();
		} else {
			return false;
		}

		if (filename == "") {
			std::cout << " -- " << utils::colorText(TextType::DANGER_B, "Path for 'test.txt' cannot be empty") << std::endl;
			return false;
		} else {
			std::cout << "Test image filename: " << filename << std::endl;
			if (!utils::isValidPath(filename)) {
				std::cout << " |-- " << utils::colorText(TextType::DANGER_B, "Invalid path: " + filename) << std::endl;
				return false;
			} else {
				std::cout << " |-- test_file: " << utils::colorText(TextType::SUCCESS_B, filename) << std::endl; 
				std::cout << " |-- test_file_prefix: " << utils::colorText(TextType::SUCCESS_B, test_file_prefix_) << std::endl; 
			}
		}
		
		reader.open(filename);
		if (reader.is_open()) {
			std::string line;
			while (std::getline(reader, line)) {
				if (line.size() > 0) {
					MyImageInfo item;
					bool done = false;
					for (int i=line.size() - 1; i>=0 && !done; i--) {
						if (line[i] == '/') {
							done = true;
							item.name = line.substr(i + 1, line.size() - i - 1);
						}
					}
					item.path = image_root_ + "/" + line;
					item.image = cv::imread(item.path, cv::IMREAD_COLOR);
					std::cout << "Name: " << item.name << "\t(Size: " << item.image.cols << "x" << item.image.rows << ")" << std::endl;
					if (!item.image.empty()) {
						item.labels = this->getBoundingBox(item.path, image_filetype, item.image.size());
						test_images_.push_back(item);
					}
				}
			}
			reader.close();
		} else {
			return false;
		}
		
		if (test_images_.size() == 0) {
			std::cout << " |-- " << utils::colorText(TextType::DANGER_B, "Cannot find any images from " + filename) << std::endl;
			return false;
		}
		
		std::cout << " |-- test_images: " << utils::colorText(TextType::SUCCESS_B, cv::format("%d images", int(test_images_.size()))) << std::endl; 
		
		return true;
	}
	
	std::vector<MyBox> getBoundingBox(std::string image_filename, std::string filetype, cv::Size image_size) {
		std::vector<MyBox> boxes;
		std::size_t found = image_filename.find(filetype);
		if (found != std::string::npos) {
			std::string label_filename = image_filename.substr(0, int(found)) + ".txt";
			if (utils::isValidPath(label_filename)) {
				std::ifstream reader;
				reader.open(label_filename);
				if (reader.is_open()) {
					std::string line;
					while (std::getline(reader, line)) {
						if (line != "") {
							MyBox item = my_utils::getValue(line, " ", image_size);
							if (item.id >= 0 && item.box.width > 0 && item.box.height > 0) {
								boxes.push_back(item);
							}
						}
					}
				}
			}
		}
		return boxes;
	}

	bool loadAnnotations(YAML::Node node) {
		std::string path = node.as<std::string>();
		if (!utils::isValidPath(path)) {
			std::cout << " |-- " << utils::colorText(TextType::DANGER_B, "Invalid path: " + path) << std::endl;
			return false;
		}
		
		std::ifstream reader;
		reader.open(path);
		std::string filename = "";
		if (reader.is_open()) {
			std::string line;
			std::string key1 = "= ";
			std::string key2 = ".names";
			while (std::getline(reader, line) && filename == "") {
				std::size_t found1 = line.find(key1);
				std::size_t found2 = line.find(key2);
				if (found1 != std::string::npos && found2 != std::string::npos) {
					int length = line.size() - int(found1) - key1.size();
					filename = line.substr(int(found1) + key1.size(), length);
				}
			}
			reader.close();
		} else {
			return false;
		}
		
		if (filename == "") {
			std::cout << " -- " << utils::colorText(TextType::DANGER_B, "Path for '.names' cannot be empty") << std::endl;
			return false;
		} else {
			if (!utils::isValidPath(filename)) {
				std::cout << " |-- " << utils::colorText(TextType::DANGER_B, "Invalid path: " + filename) << std::endl;
				return false;
			} else {
				std::cout << " |-- annotations_file: " << utils::colorText(TextType::SUCCESS_B, filename) << std::endl; 
			}
		}
		
		reader.open(filename);
		if (reader.is_open()) {
			std::string line;
			int index = 0;
			while (std::getline(reader, line)) {
				if (line != "") {
					classnames_.insert(std::pair<int, std::string>(index, line));
					index++;
				}
			}
			reader.close();
		} else {
			return false;
		}
		
		if (classnames_.size() == 0) {
			std::cout << " |-- " << utils::colorText(TextType::DANGER_B, "Cannot find any classname from " + filename) << std::endl;
			return false;
		}
		
		std::stringstream ss;
		std::map<int, std::string>::iterator it;
		for (it = classnames_.begin(); it != classnames_.end(); it++) {
			ss << "[" << it->second << "] ";
		}
		
		std::cout << " |-- classnames: " << utils::colorText(TextType::SUCCESS_B, ss.str()) << std::endl; 
		
		return true;
	}
	
	bool loadModel(YAML::Node node) {
		std::string weights_file = node["weights_file"] ? node["weights_file"].as<std::string>() : "";
		std::string cfg_file = node["cfg_file"] ? node["cfg_file"].as<std::string>() : "";
		
		if (weights_file == "" || cfg_file == "") {
			std::cout << " |-- " << utils::colorText(TextType::DANGER_B, "Invalid config for weights_file or cfg_file") << std::endl;
			return false;
		} else {
			if (!utils::isValidPath(weights_file) || !utils::isValidPath(cfg_file)) {
				std::cout << " |-- " << utils::colorText(TextType::DANGER_B, "Invalid path for weights_file or cfg_file") << std::endl;
				return false;
			}
		}
		
		int width, height;
		if (!node["net_width"] || !node["net_height"]) {
			std::cout << " |-- " << utils::colorText(TextType::DANGER_B, "Invalid config for net_width or net_height") << std::endl;
			return false;
		} else {
			width = node["net_width"].as<int>();
			height = node["net_height"].as<int>();
		}
		
		double conf, nms;
		if (!node["confidence_thr"] || !node["nms_thr"]) {
			std::cout << " |-- " << utils::colorText(TextType::DANGER_B, "Invalid config for confidence_thr or nms_thr") << std::endl;
			return false;
		} else {
			conf = node["confidence_thr"].as<double>();
			nms = node["nms_thr"].as<double>();
		}
		
		std::cout << " Loading YOLO model" << std::endl;
		std::cout << " |-- weights: " << utils::colorText(TextType::SUCCESS_B, weights_file) << std::endl; 
		std::cout << " |-- cfg: " << utils::colorText(TextType::SUCCESS_B, cfg_file) << std::endl; 
		std::cout << " |-- net size: " << utils::colorText(TextType::SUCCESS_B, cv::format("%d x %d", width, height)) << std::endl; 
		std::cout << " |-- confidence threshold: " << utils::colorText(TextType::SUCCESS_B, std::to_string(conf)) << std::endl; 
		std::cout << " |-- nms threshold: " << utils::colorText(TextType::SUCCESS_B, std::to_string(nms)) << std::endl; 
		
		detector_.init(
			width, height,
			weights_file, cfg_file, 
			classnames_,
			conf, nms
		);
		
		return true;
	}
	
	bool is_ok_;
	std::string image_root_;
	std::string test_file_prefix_;
	std::vector<MyImageInfo> test_images_;
	std::map<int, std::string> classnames_;
	Detector detector_;
};

void checkInput(int argc, char **argv, int &i, std::string key, std::string &value) {
	if (i+1 < argc) {
		value = argv[i+1];
		i = i+1;
	} else {
		std::cerr << utils::colorText(TextType::DANGER_B, "'" + key + "' option requires one argument") << std::endl;
	}
}

static void showUsage(std::string name) {
	std::stringstream ss;
	ss << "\nUsage: " << name << " <options> <value>"
		<< "\nOptions:"
		<< "\n  -h, --help\tShow this help message"
		<< "\n  -c, --config\tConfig about the training"
		<< std::endl;
	std::cout << utils::colorText(TextType::INFO, ss.str()) << std::endl;
}

void testStringPattern() {
	std::string text = "2 0.9083 0.7075 0.0667 0.0781\n";
	std::string key = " ";
	
	MyBox box = my_utils::getValue(text, key, cv::Size(200, 200));
	std::cout << "Box: " << box.id << ", Rect: " << box.box << std::endl;	
}

void testIOUComputation() {
	std::vector<cv::Rect> pair;
	pair.push_back(cv::Rect(460, 218, 26, 87));
	pair.push_back(cv::Rect(460, 218, 26, 80));
	pair.push_back(cv::Rect(291, 368, 163, 171));
	pair.push_back(cv::Rect(316, 363, 151, 138));
	pair.push_back(cv::Rect(291, 368, 163, 171));
	pair.push_back(cv::Rect(278, 432, 117, 115));	
	pair.push_back(cv::Rect(401, 479, 186, 92));
	pair.push_back(cv::Rect(377, 471, 187, 99));	
	
	cv::Mat image = cv::Mat::zeros(600, 800, CV_8UC3);
	int N = int(pair.size()) / 2;
	for (int i=0; i<N; i++) {
		cv::Scalar color(rand()%255, rand()%255, rand()%255);
		cv::Rect rect1 = pair[2 * i];
		cv::Rect rect2 = pair[2 * i + 1];
		cv::rectangle(image, rect1, color, 1);
		cv::rectangle(image, rect2, color, 1);
		
		cv::Rect overlap = my_utils::overlappingRect(rect1, rect2);
		cv::rectangle(image, overlap, color, -1);
		double union_area = my_utils::unionRectArea(rect1, rect2, overlap);
		std::cout << " -- Overlap: " << overlap.area() 
				<< ", Union: " << union_area
				<< "\tAccuracy: " << double(overlap.area()) / double(union_area)
				<< std::endl;
		cv::imshow("overlap union", image);
		cv::waitKey(0);
	}
}

int main(int argc, char **argv) {
	
	//testIOUComputation();
	//return 0;
	
	// testStringPattern();
	// return 0;

	std::string config_file("");
	
	for (int i=1; i<argc; i++) {
		std::string arg = argv[i];
		if (arg == "-h" || arg == "--help") {
			return 1;
		} else if (arg == "-c" || arg == "--config") {
			checkInput(argc, argv, i, "--config", config_file);
		}
	}
	
	if (config_file == "") {
		std::cout << utils::colorText(TextType::DANGER_B, "Invalid config file") << std::endl;
		showUsage(argv[0]);
		return -1;
	} else {
		if (!utils::isValidPath(config_file)) {
			std::cout << utils::colorText(TextType::DANGER_B, "Invalid config file: " + config_file) << std::endl;
			showUsage(argv[0]);
			return -1;	
		}
	}
	
	MyTools mytools(config_file);
	mytools.run();
	
	return 0;
}
