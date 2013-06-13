#ifndef __UTIL__
#define __UTIL__

#define PATH_SEP "/"

#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <ctime>

// opencv include file
#include <opencv2/opencv.hpp>

// boost include file
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/xml_parser.hpp>
#include <boost/foreach.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/random.hpp>

#include "CDataset.h"
#include "CPatch.h"

class CConfig
{
public:
    CConfig(){

    };
    ~CConfig(){};

    double widthScale,heightScale;

    int loadConfig(const char* filename);

    // Path to trees
    std::string	treepath;
    // Number of trees
    int		ntrees;
    // Patch width
    int		p_width;
    // Patch height
    int		p_height;
    // Path to images
    std::string	impath;
    // File with names of images
    std::string	imfiles;
    // Extract features
    bool		xtrFeature;
    // Scales
    std::vector<float>	scales;
    // Ratio
    std::vector<float>	ratios;
    // Output path
    std::string	outpath;
    // scale factor for output image (default: 128)
    int		out_scale;
    // Path to training image folder 12/12/14 MasahikoToyoshi
    std::string	trainpath;
    // training data file name
    std::string	traindatafile;
    // Path to positive examples
    std::string	trainpospath;
    // File with postive examples
    std::string	trainposfiles;
    // Subset of positive images -1: all images
    int		subsamples_pos;
    // Sample patches from pos. examples
    unsigned int	samples_pos;
    // Path to positive examples
    std::string	trainnegpath;
    // File with postive examples
    std::string	trainnegfiles;
    // Subset of neg images -1: all images
    int		subsamples_neg;
    // Samples from pos. examples
    unsigned int	samples_neg;
    // offset for saving tree number
    int	off_tree;
    // train image per tree
    int	imagePerTree;
    // sample number per leaf
    int	min_sample;
    // max depth of tree
    int	max_depth;
    // feature channel number
    int	featureChannel;
    // extract patch ratio
    double	patchRatio;
    // stride of getting patch
    int stride;
    // test data path
    std::string testPath;
    // test data file
    std::string testData;
    // test mode
    int testMode;
    double detectThreshold;

    std::string negDataPath,negDataList;

    int showGT;

    std::string classDatabaseName;

    int learningMode;

    std::string traindatalist;
    std::string testdatalist;
    int negMode;

    double pnRatio;

    double acPatchRatio;
    double mindist;
    double maxdist;

    int nOfTrials;

    int crossVMode;

    std::string negFolderList;

private:
    boost::property_tree::ptree pt;

};

class databaseNode{
public:
    databaseNode(std::string className,cv::Size size, uchar depth)
        : name(className), instances(1), classSize(size),classDepth(depth){}

    std::string name;
    cv::Size classSize;
    uchar classDepth;

    int instances;
};

class CClassDatabase{
public:
    CClassDatabase(){
        vNode.clear();
    }

    void add(std::string str, cv::Size size, uchar depth);
    void write(const char* str);
    void read(const char* str);

    int search(std::string str) const;

    void show() const;

    void clear(){
        vNode.clear();
    };

    std::vector<databaseNode> vNode;
};

class nCk{
public:
    std::set<int> generate(int n, int k){
        std::set<int> cmb;
        int j;
        //srand(time(NULL));
        boost::uniform_smallint<> dst(0,n);
        boost::variate_generator<
                boost::lagged_fibonacci1279&, boost::uniform_smallint<>
                > rand( gen, dst );

        for(int i = n - k; i < n; i++){
            j = rand()%i;
            if(cmb.find(j) == cmb.end())
                cmb.insert(j);
            else
                cmb.insert(i);
        }
        return cmb;
    }
private:
    static boost::lagged_fibonacci1279 gen;
};

void loadTrainPosFile(CConfig conf, std::vector<CPosDataset> &posSet);//, boost::mt19937 &gen);
void loadTrainNegFile(CConfig conf, std::vector<CNegDataset> &negSet);
//void loadTestFile(CConfig conf, std::vector<CTestDataset> &testSet);

void extractPosPatches(std::vector<CPosDataset> &posSet,
                       std::vector<CPosPatch> &posPatch,
                       CConfig conf,
                       const int treeNum,
                       const CClassDatabase &classDatabase);
void extractNegPatches(std::vector<CNegDataset> &negSet,std::vector<CNegPatch> &negPatch,
                       CConfig conf);
void extractTestPatches(CTestDataset &testSet,std::vector<CTestPatch> &testPatch, CConfig conf);

void pBar(int p,int maxNum, int width);


#endif
