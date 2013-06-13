#ifndef DETECTIONRESULT_H
#define DETECTIONRESULT_H

#include <string>
#include <opencv2/opencv.hpp>

class detectionResult
{
public:
    detectionResult(){}
    virtual ~detectionResult(){}
    std::string className;
    cv::Rect bbox;

    double error;
    int found;
    float score;
};

#endif // DETECTIONRESULT_H
