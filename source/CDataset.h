#ifndef CDATASET_H
#define CDATASET_H

//#include "util.h"
#include "HoG.h"
#include <iostream>
#include <sstream>

//#include "util.h"

class CParamset{
public:
    CParamset(){}
    ~CParamset(){}

    int setCenterPoint(cv::Point cp){centerPoint = cp;return 0;}
    cv::Point getCenterPoint()const{return centerPoint;}

    int setClassName(std::string name){className = name;return 0;}
    std::string getClassName()const{return className;}

    int setAngle(double an){angle = an;return 0;}
    double getAngle(){return angle;}

    int showParam();

    std::string outputParam();
    //void readParam(std::stringstream *in);

private:
    // parameters should be estimated
    cv::Point centerPoint;
    std::string className;
    double angle;

    cv::Rect boundingbox;
};

// !!caution!!
// this is abstract class you should inherit this class!
class CDataset {
public:
    CDataset();
    virtual ~CDataset();

    int loadImage(double mindist, double maxdist);
    int releaseImage();

    int extractFeatures();
    int releaseFeatures();

    void setRgbImagePath(std::string rgb_path){rgb = rgb_path;}
    void setDepthImagePath(std::string depth_path){depth = depth_path;}

    std::string getRgbImagePath(){return rgb;}

    // loaded images and features
    std::vector<cv::Mat*> img, feature;

private:
    // flag for image or features loaded on memory
    bool imgFlag, featureFlag;

    // image file path
    std::string rgb, depth, mask;

    // min and max filter
    void minFilter(cv::Mat* src, cv::Mat* des, int fWind);
    void maxFilter(cv::Mat* src, cv::Mat* des, int fWind);

    //  cv::Rect bBox;
    //  std::vector<std::string> className;
    //  std::vector<cv::Point> centerPoint;
    //  std::vector<double> angles;

    //void showDataset();

    HoG hog;
};

class CPosDataset : public CDataset{
public:
    CPosDataset(){}
    virtual ~CPosDataset(){}

    void setClassName(std::string name){param.setClassName(name);}
    void setAngle(double an){param.setAngle(an);}
    void setCenterPoint(cv::Point cp){param.setCenterPoint(cp);}

    CParamset* getParam(){return &param;}

private:
    CParamset param;
};

class CNegDataset : public CDataset{
public:
    CNegDataset(){}
    virtual ~CNegDataset(){}

};

class CTestDataset : public CDataset{
public:
    CTestDataset(){}
    virtual ~CTestDataset(){}

    std::vector<CParamset> param;
};

#endif // CDATASET_H
