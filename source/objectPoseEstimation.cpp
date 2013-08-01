#include "CRForest.h"
#include "ctlkinect.h"
#include <opencv2/opencv.hpp>
#include <boost/timer.hpp>

#include "util.h"
#include "CDataset.h"

using namespace std;

void loadTestFileMultiObject(CConfig conf, std::vector<CTestDataset> &testSet){
    std::string testfilepath = conf.testPath + PATH_SEP +  conf.testData;
    int n_folders;
    int n_files;
    std::vector<std::string> testimagefolder;
    CDataset temp;
    std::vector<CDataset> tempDataSet;
    std::string testDataListPath;
    int dataSetNum;

    cv::Point tempPoint;

    std::ifstream in(testfilepath.c_str());
    if(!in.is_open()){
        std::cout << "test data floder list is not found!" << std::endl;
        exit(1);
    }
    in >> n_folders;

    testimagefolder.resize(n_folders);
    for(int i = 0;i < n_folders; ++i)
        in >> testimagefolder.at(i);
    in.close();

    //read train file name and grand truth from file
    tempDataSet.resize(0);
    for(int i = 0;i < n_folders; ++i){
        CTestDataset testTemp;
        std::string nameTemp;

        testDataListPath
                = conf.testPath + PATH_SEP + testimagefolder.at(i)
                + PATH_SEP + conf.testdatalist;
        std::string imageFilePath
                = conf.testPath + PATH_SEP + testimagefolder.at(i) + PATH_SEP;
        //std::cout << trainDataListPath << std::endl;
        std::ifstream testDataList(testDataListPath.c_str());
        if(testDataList.is_open()){
            testDataList >> n_files;
            //std::cout << "number of file: " << n_files << std::endl;
            for(int j = 0;j < n_files; ++j){
                //read file names
                testDataList >> nameTemp;
                testTemp.setRgbImagePath(imageFilePath + nameTemp);

                testDataList >> nameTemp;
                testTemp.setDepthImagePath(imageFilePath + nameTemp);

                testDataList >> nameTemp;// dummy

                //temp.centerPoint.resize(0);
                testTemp.param.clear();

                //read center point
                std::string tempClassName;
                cv::Point tempPoint;
                double tempAngle[3];
                do{
                    CParamset tempParam;
                    //read class name
                    testDataList >> tempClassName;

                    if(tempClassName != "EOL"){
                        tempParam.setClassName(tempClassName);
                        testDataList >> tempPoint.x;
                        testDataList >> tempPoint.y;
                        //temp.centerPoint.push_back(tempPoint);
                        tempParam.setCenterPoint(tempPoint);
                        testDataList >> tempAngle[2];
                        //temp.angles.push_back(tempAngle);
                        tempParam.setAngle(tempAngle);

                        testTemp.param.push_back(tempParam);
                        //tempParam.showParam();
                    }
                }while(tempClassName != "EOL");

                testSet.push_back(testTemp);
                testTemp.param.clear();
            }
            testDataList.close();
        }
    }
}

void detect(const CRForest &forest, CConfig conf){
    //std::vector<CTestDataset> dataSet;
    CtlKinect kinect;

    //std::fstream result("detectionResult.txt", std::ios::out);

    CDetectionResult detectR;

    //set dataset
    //dataSet.clear();
    //loadTestFileMultiObject(conf,dataSet);
//    while(cv::waitKey(1) == -1){
//    cv::namedWindow("depthImage");
//    cv::Mat showDepth = cv::imread("./bottle_1_1_1_depthcrop.png");//new cv::Mat(480,640, CV_8U);
//    cv::imshow("depthImage",showDepth);
//    std::cout << "hitamuki" << std::endl;
//    }
    //cv::waitKey(0);
    cv::namedWindow("depth");
    cv::namedWindow("detectResult");
    cv::namedWindow("voteImage");

    while(cv::waitKey(1) == -1){
//        dataSet.at(i).loadImage(conf.mindist, conf.maxdist);
//        detectR = forest.detection(dataSet.at(i));
//        result << dataSet.at(i).param.at(0).getClassName() << " " << detectR.className << " " << detectR.found << " " << detectR.score << " " << detectR.error << std::endl;



        cv::Mat *rgb = new cv::Mat(480,640,CV_8UC3);
        cv::Mat *depth = new cv::Mat(480,640,CV_16UC1);

        kinect.getRGBDData(rgb, depth);

        cropImageAndDepth(rgb, depth, conf.mindist, conf.maxdist);

        CTestDataset seqImg;
        seqImg.img.push_back(rgb);
        seqImg.img.push_back(depth);

        cv::Mat showDepth = cv::Mat(depth->rows, depth->cols, CV_8U);
        depth->convertTo(showDepth, CV_8U, 255.0 / 1000.0);

        cv::imshow("detectResult", *seqImg.img.at(0));
        cv::imshow("depth", showDepth);


        //seqImg.img.at(1)->convertTo(*showDepth, CV_8U, 255.0 / 1000.0);

        //cv::waitKey(0);

        //detectR = forest.detection(seqImg);

    }
    //delete showDepth;
    cv::destroyAllWindows();

    //result.close();
}



int main(int argc, char* argv[]){

    CConfig		conf;	 // setting
    std::vector<CDataset> dataSet; // training data name list and grand truth

    //read argument
    //check argument
    if(argc < 2) {
        cout << "Usage: ./learning [config.xml]"<< endl;
        conf.loadConfig("hfConfig.xml");
    } else
        conf.loadConfig(argv[1]);

    if(argc < 3)
        conf.off_tree = 0;
    else
        conf.off_tree = atoi(argv[2]);

    // create random forest class
    CRForest forest(conf);

    forest.loadForest();

    //create tree directory
    string opath(conf.outpath);
    //std::cout << "kokomade kitayo" << std::endl;
    //std::cout << conf.outpath << std::endl;
    opath.erase(opath.find_last_of(PATH_SEP));
    string execstr = "mkdir ";
    execstr += opath;
    system( execstr.c_str() );

    // learning
    //forest.learning();
    detect(forest, conf);

    return 0;
}

