#include "cvdnn_detector.h"
#include <chrono>
#include <sys/time.h>
#include <ctime>
#include "utils.h"

namespace cvdnn_detector {
	std::vector<std::string> getNetModelOutputsNames(const cv::dnn::Net &net) {
		static std::vector<std::string> names;
		if (names.empty()) {
			// Get the indices of the output layers
			std::vector<int> out_layers = net.getUnconnectedOutLayers();
			// get names of all layers in network
			std::vector<cv::String> layernames = net.getLayerNames();
			names.resize(int(out_layers.size()));
			for (size_t i=0; i<out_layers.size(); i++) {
				names[i] = layernames[out_layers[i] - 1];
			}
		}
		return names;
	}
}

Detector::Detector()
{
	classnames_.clear();
	colors_.clear();
}

Detector::~Detector()
{
}

void Detector::init(
	int net_width, 
	int net_height, 
	std::string weights_file, 
	std::string cfg_file, 
	std::map<int, std::string> classnames, 
	double confidence_threshold, 
	double nms_threshold
) {
	scale_ = 1.0 / 255.0;
	mean_ = cv::Scalar(0, 0, 0);
	classnames_ = classnames;
	net_size_ = cv::Size(net_width, net_height);
	conf_thr_ = confidence_threshold;
	nms_thr_ = nms_threshold;
	
	std::map<int, std::string>::iterator it;
	for (it = classnames_.begin(); it != classnames_.end(); it++) {
		colors_.insert(std::pair<int, cv::Scalar>(it->first, cv::Scalar(rand()%100, rand()%100, rand()%100)));
	}
	
	net_ = cv::dnn::readNet(weights_file, cfg_file);
	net_.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
	net_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
}

char Detector::detect(MyImageInfo &item, cv::Mat &dst)
{
		auto t_start = std::chrono::high_resolution_clock::now();
		
		dst = item.image.clone();
		cv::Mat blob = cv::dnn::blobFromImage(dst, scale_, net_size_, mean_, true, false);
		net_.setInput(blob);
		
		std::vector<cv::Mat> outs;
		net_.forward(outs, cvdnn_detector::getNetModelOutputsNames(net_));
		
		std::vector<int> class_ids;
		std::vector<float> confidences;
		std::vector<cv::Rect> boxes;
		
		for (int i=0; i<(int)outs.size(); i++) {
        cv::Mat out = outs[i];
        float *data = (float*)out.data;
        //std::cout << " Mat: " << out.cols << " x " << out.rows << std::endl;
        for (int j=0; j<out.rows; j++, data += out.cols) {
            cv::Mat scores = out.row(j).colRange(5, out.cols);
            cv::Point class_id_point;
            double conf;
            cv::minMaxLoc(scores, 0, &conf, 0, &class_id_point);
            if (conf > conf_thr_) {
                // std::cout << " -- conf: " << conf << std::endl;
                int cx = int(data[0] * dst.cols);
                int cy = int(data[1] * dst.rows);
                int w = int(data[2] * dst.cols);
                int h = int(data[3] * dst.rows);
                int minx = cx - w / 2;
                int miny = cy - h / 2;
                class_ids.push_back(class_id_point.x);
                confidences.push_back(conf);
                boxes.push_back(cv::Rect(minx, miny, w, h));
            }
        }
    }
    
    int fontface = cv::FONT_HERSHEY_SIMPLEX;
    double fontscale = 0.6;
    int thickness = 1;
    
		item.detections.clear();
		
    std::vector<int> indices;
		int text_height = 20;
    cv::dnn::NMSBoxes(boxes, confidences, conf_thr_, nms_thr_, indices);
    for (size_t i=0; i<indices.size(); i++) {
        int idx = indices[i];
        cv::Rect box = boxes[idx];
        int class_id = class_ids[idx];
				
				std::map<int, std::string>::iterator it = classnames_.find(class_id);
				std::map<int, cv::Scalar>::iterator it2 = colors_.find(class_id);

				if (it == classnames_.end()) { continue; }
				
				MyBox result;
				result.id = class_id;
				result.box = box;
				item.detections.push_back(result);
				
        std::string text = cv::format("[%d] %s, %.2f", class_id, it->second.c_str(), confidences[idx]);
        //std::cout << " >> Detected: " << text << ", " << confidences[idx] << std::endl;
        int baseline = 0;
        cv::Size tsize = cv::getTextSize(text, fontface, fontscale, thickness, &baseline);
				text_height = tsize.height;
        cv::Rect trect(box.x, box.y - tsize.height - baseline, tsize.width, tsize.height + 2 * baseline);
        cv::rectangle(dst, trect, it2->second, -1);
        cv::rectangle(dst, box, it2->second, 2);
        cv::putText(dst, text, cv::Point(box.x, box.y), fontface, fontscale, cv::Scalar::all(255), thickness);
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double, std::milli>(t_end-t_start).count();
		std::cout << "    Detection elapsed: " << utils::colorText(TextType::SUCCESS_B, cv::format("%.3lf ms", elapsed)) << std::endl;
    
    // Put efficiency information.
    std::vector<double> layersTimes;
    double freq = cv::getTickFrequency() / 1000;
    double t = net_.getPerfProfile(layersTimes) / freq;
        
		cv::putText(dst, cv::format("%s", item.name.c_str()), cv::Point(10, 20), fontface, fontscale, cv::Scalar(0, 255, 0), thickness);
		cv::putText(dst, cv::format("Inference time: %.3lf ms", t), cv::Point(10, int(20 + text_height * 1.5)), fontface, fontscale, cv::Scalar(0, 255, 0), thickness);
		cv::putText(dst, cv::format("Total elapsed: %.3lf ms", elapsed), cv::Point(10, int(20 + text_height * 3.0)), fontface, fontscale, cv::Scalar(0, 255, 0), thickness);
	
		blob.release();
		return ' ';
}

