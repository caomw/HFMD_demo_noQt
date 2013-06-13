#include <boost/timer.hpp>
#include "CRForest.h"

void CRForest::learning(){
    // grow each tree
    // if you want to fix this program multi thread
    // you should change below
    for(int i = 0;i < conf.ntrees; ++i){
        growATree(i);
    } // end tree loop
}

void CRForest::growATree(const int treeNum){
    // positive, negative dataset
    std::vector<CPosDataset> posSet(0);
    std::vector<CNegDataset> negSet(0);

    // positive, negative patch
    std::vector<CPosPatch> posPatch(0);
    std::vector<CNegPatch> negPatch(0);

    char buffer[256];

    std::cout << "tree number " << treeNum << std::endl;

    // initialize random seed
    boost::mt19937    gen( treeNum * static_cast<unsigned long>(time(NULL)) );
    boost::timer t;

    loadTrainPosFile(conf, posSet);//, gen);
    loadTrainNegFile(conf, negSet);

    std::cout << "dataset loaded" << std::endl;

    // initialize class database
    classDatabase.clear();

    // extract pos features and register classDatabase
    for(int i = 0; i < posSet.size(); ++i){
        //std::cout << i << std::endl;

        posSet.at(i).loadImage(conf.mindist, conf.maxdist);
        posSet.at(i).extractFeatures();


        //std::cout << posSet.size() << std::endl;

        classDatabase.add(posSet.at(i).getParam()->getClassName(),posSet.at(i).img.at(0)->size(),0);
    }

    classDatabase.show();

    // extract neg features
    for(int i = 0; i < negSet.size(); ++i){
        negSet.at(i).loadImage(conf.mindist,conf.maxdist);
        negSet.at(i).extractFeatures();
    }

    CRTree *tree = new CRTree(conf.min_sample, conf.max_depth, classDatabase.vNode.size(),this->classDatabase);
    std::cout << "tree created" << std::endl;

    extractPosPatches(posSet,posPatch,conf,treeNum,this->classDatabase);
    extractNegPatches(negSet,negPatch,conf);

    std::cout << "extracted pathes" << std::endl;
    std::vector<int> patchClassNum(classDatabase.vNode.size(), 0);

    for(int j = 0; j < posPatch.size(); ++j)
        patchClassNum.at(classDatabase.search(posPatch.at(j).getClassName()))++;
    
    // grow tree
    //vTrees.at(treeNum)->growTree(vPatches, 0,0, (float)(vPatches.at(0).size()) / ((float)(vPatches.at(0).size()) + (float)(vPatches.at(1).size())), conf, gen, patchClassNum);
    tree->growTree(posPatch,negPatch, 0,0, ((float)posPatch.size() / (float)(posPatch.size() + negPatch.size())), conf, patchClassNum);

    // save tree
    sprintf(buffer, "%s%03d.txt",
            conf.treepath.c_str(), treeNum + conf.off_tree);
    std::cout << "tree file name is " << buffer << std::endl;
    tree->saveTree(buffer);

    // save class database
    sprintf(buffer, "%s%s%03d.txt",
            conf.treepath.c_str(),
            conf.classDatabaseName.c_str(), treeNum + conf.off_tree);
    std::cout << "write tree data" << std::endl;
    classDatabase.write(buffer);

    double time = t.elapsed();

    std::cout << "tree " << treeNum << " calicuration time is " << time << std::endl;

    sprintf(buffer, "%s%03d_timeResult.txt",conf.treepath.c_str(), treeNum + conf.off_tree);
    std::fstream lerningResult(buffer, std::ios::out);
    if(lerningResult.fail()){
        std::cout << "can't write result" << std::endl;
    }

    lerningResult << time << std::endl;

    lerningResult.close();

    delete tree;

    posPatch.clear();
    negPatch.clear();

    posSet.clear();
    negSet.clear();
}

void CRForest::loadForest(){
    char buffer[256];
    char buffer2[256];
    for(int i = 0; i < vTrees.size(); ++i){
        sprintf(buffer, "%s%03d.txt",conf.treepath.c_str(),i);
        sprintf(buffer2, "%s%s%03d.txt", conf.treepath.c_str(), conf.classDatabaseName.c_str(), i);
        vTrees[i] = new CRTree(buffer, buffer2, conf);

        //std::cout << buffer2 << std::endl;
        classDatabase.read(buffer2);
    }
}

// name   : detect function
// input  : image and dataset
// output : classification result and detect picture
void CRForest::detection(CTestDataset &testSet) const{



    cv::namedWindow("outputImage");
    cv::namedWindow("depthImage");
    cv::namedWindow("voteImage");


    int classNum = classDatabase.vNode.size();//contain class number
    std::vector<CTestPatch> testPatch;
    std::vector<const LeafNode*> result;

    std::vector<std::vector<const LeafNode*> > storedLN(0);

    cv::vector<cv::Mat> outputImage(classNum);
    cv::vector<cv::Mat> outputImageColorOnly(classNum);
    std::vector<int> totalVote(classNum,0);
    boost::timer t;

    testSet.loadImage(conf.mindist, conf.maxdist);


    cv::Mat showDepth = cv::Mat(480,640, CV_8U);
    testSet.img.at(1)->convertTo(showDepth, CV_8U, 255.0 / 1000.0);
    cv::imshow("depthImage",showDepth);



    testSet.extractFeatures();
    //testSet.releaseImage();

    for(int i = 0; i < classNum; ++i){
        outputImage.at(i) = testSet.img.at(0)->clone();
        outputImageColorOnly.at(i) = cv::Mat::zeros(testSet.img.at(0)->rows,testSet.img.at(0)->cols,CV_32FC1);
    }

    // extract feature from test image
    //features.clear();
    //extractFeatureChannels(image.at(0), features);

    // add depth image to features
    //features.push_back(image.at(1));

    // extract patches from features
    extractTestPatches(testSet,testPatch,this->conf);

    std::cout << "patch num: " << testPatch.size() << std::endl;

    t.restart();

    // regression and vote for every patch
    for(int j = 0; j < testPatch.size(); ++j){
        result.clear();
        this->regression(result, testPatch.at(j));

        storedLN.push_back(result);

        // vote for all trees (leafs)
        for(std::vector<const LeafNode*>::const_iterator itL = result.begin();itL!=result.end(); ++itL) {

            for(int c = 0; c < classNum; c++){
                //if(!(*itL)->param.at(c).empty()){
                //if((*itL)->pfg.at(c) > 0.9  ){
                for(int l = 0; l < (*itL)->param.at(c).size(); ++l){
                    cv::Point patchSize(conf.p_height/2,conf.p_width/2);
                    cv::Point pos(testPatch.at(j).getRoi().x + patchSize.x +  (*itL)->param.at(c).at(l).getCenterPoint().x, testPatch.at(j).getRoi().y + patchSize.y +  (*itL)->param.at(c).at(l).getCenterPoint().y);
                    if(pos.x > 0 && pos.y > 0 && pos.x < outputImageColorOnly.at(c).cols && pos.y < outputImageColorOnly.at(c).rows){
                        outputImageColorOnly.at(c).at<float>(pos.y,pos.x) += (*itL)->pfg.at(c) / 100.0;//((*itL)->pfg.at(c) - 0.9);// * 100;//weight * 500;
                        //std::cout << classDatabase.vNode.at(c).name << " " << (*itL)->pfg.at(c) << std::endl;

                        totalVote.at(c) += 1;
                    }
                }
                //}
                //}
            }
        } // for every leaf
    } // for every patch

    // vote end


    // find balance by mean shift
    for(int i = 0; i < classNum; ++i){
        //        cv::Mat hsv,hue,rgb;
        //        int bins = 256;

        //        double min,max;
        //        cv::Point minLoc,maxLoc;
        //        cv::minMaxLoc(outputImageColorOnly.at(i),&min,&max,&minLoc,&maxLoc);

        cv::GaussianBlur(outputImageColorOnly.at(i),outputImageColorOnly.at(i), cv::Size(21,21),0);

        //        //cv::cvtColor(outputImageColorOnly.at(i), rgb, CV_GRAY2BGR);
        //        //cv::cvtColor(rgb, hsv , CV_BGR2HSV);

        //        hue.create( outputImageColorOnly.at(i).size(), outputImageColorOnly.at(i).depth() );
        //        int ch[] = { 0, 0 };
        //        mixChannels( &outputImageColorOnly.at(i), 1, &hue, 1, ch, 1 );


        //        const int ch_width = 400;
        //        cv::Mat hist;
        //        cv::Mat hist_img(cv::Size(ch_width, 200), CV_8UC3, cv::Scalar::all(255));;
        //        int histSize = MAX( bins, 2 );
        //        float hue_range[] = { 0, 1 };
        //        const float* ranges = { hue_range };
        //        const int hist_size = 256;
        //        double max_val = .0;
        //        double second_val = .0;

        //        /// Get the Histogram and normalize it
        //        cv::calcHist( &outputImageColorOnly.at(i) , 1, 0, cv::Mat(), hist, 1, &histSize, &ranges, true, false );
        //        cv::normalize( hist, hist, 0., 256., cv::NORM_MINMAX, -1, cv::Mat() );

        //        cv::minMaxLoc(hist, 0, &max_val);
        //        hist.at<float>(0) = 0;
        //        cv::minMaxLoc(hist, 0, &second_val);

        //        hist.at<float>(0) = max_val;

        //        // (4)scale and draw the histogram(s)
        //        cv::Scalar color = cv::Scalar::all(100);
        //        //for(int i=0; i<sch; i++) {
        //        //  if(sch==3)
        //        //    color = Scalar((0xaa<<i*8)&0x0000ff,(0xaa<<i*8)&0x00ff00,(0xaa<<i*8)&0xff0000, 0);
        //        hist.convertTo(hist, hist.type(), 200 * 1.0/second_val,0);//?1./max_val:0.,0);
        //        for(int j=0; j<hist_size; ++j) {
        //            int bin_w = cv::saturate_cast<int>((double)ch_width/hist_size);
        //            //std::cout << "draw rect " << bin_w << " " << i << " " << hist.at<float>(j) << " " << max_val << std::endl;
        //            cv::rectangle(hist_img,
        //                          cv::Point( j*bin_w, hist_img.rows),
        //                          cv::Point((j+1)*bin_w, hist_img.rows-cv::saturate_cast<int>(hist.at<float>(j))),
        //                          color, -1);
        //        }


        //        //show and write histgram
        ////        cv::imwrite("test.png",hist_img);

        ////        cv::namedWindow("test");
        ////        cv::imshow("test",hist_img);
        ////        cv::waitKey(0);
        ////        cv::destroyWindow("test");

        //        /// Get Backprojection
        //        cv::Mat backproj;
        //        calcBackProject( &hue, 1, 0, hist, backproj, &ranges, 1, true );

        //        cv::Rect tempRect = cv::Rect(maxLoc.x,maxLoc.y,classDatabase.vNode.at(i).classSize.width,classDatabase.vNode.at(i).classSize.height);//classDatabase.vNode.at(i).classSize.width,classDatabase.vNode.at(i).classSize.height);//outputImageColorOnly.at(i).cols,outputImageColorOnly.at(i).rows);
        //        cv::TermCriteria terminator;
        //        terminator.maxCount = 1000;
        //        terminator.epsilon  = 10;
        //        terminator.type = cv::TermCriteria::MAX_ITER + cv::TermCriteria::EPS;
        //        cv::meanShift(backproj,tempRect,terminator);

        //        //cv::Size tempSize = classDatabase.vNode.at(c).classSize;
        //        //cv::Rect_<int> outRect(tempRect.x,maxLoc.y - tempSize.height / 2 , tempSize.width,tempSize.height);
        //        cv::rectangle(outputImage.at(i),tempRect,cv::Scalar(0,200,0),3);
        //        cv::putText(outputImage.at(i),classDatabase.vNode.at(i).name,cv::Point(tempRect.x,tempRect.y),cv::FONT_HERSHEY_SIMPLEX,1.2, cv::Scalar(0,0,0), 2, CV_AA);

    }

    double time = t.elapsed();
    std::cout << time << "sec" << std::endl;
    std::cout << 1 / (time / classNum) << "Hz" << std::endl;

    //create result directory
//    std::string opath(testSet.getRgbImagePath());
//    opath.erase(opath.find_last_of(PATH_SEP));
//    std::string imageFilename = testSet.getRgbImagePath();
//    imageFilename.erase(imageFilename.find_last_of("."));
//    //imageFilename.erase(imageFilename.begin(),imageFilename.find_last_of(PATH_SEP));
//    imageFilename = imageFilename.substr(imageFilename.rfind(PATH_SEP),imageFilename.length());

    //opath += PATH_SEP;
//    opath += imageFilename;
//    std::string execstr = "mkdir ";
//    execstr += opath;
//    system( execstr.c_str() );

//    for(int c = 0; c < classNum; ++c){
//        std::stringstream cToString;
//        cToString << c;
//        std::string outputName = "output" + cToString.str() + ".png";
//        std::string outputName2 = opath + PATH_SEP + "vote_" + classDatabase.vNode.at(c).name + ".png";
//        //cv::imwrite(outputName.c_str(),outputImage.at(c));
//        //cv::cvtColor(outputImageColorOnly)

//        cv::Mat writeImage;
//        //hist.convertTo(hist, hist.type(), 200 * 1.0/second_val,0);
//        outputImageColorOnly.at(c).convertTo(writeImage, CV_8UC1, 254);
//        cv::imwrite(outputName2.c_str(),writeImage);
//    }

    cv::imshow("voteImage",outputImageColorOnly.at(0));

    std::cout << "detection result outputed" << std::endl;


    //    cv::cvtColor(outputImageColorOnly.at(0),outputImageC1,CV_8UC1);

    //    cv::namedWindow("test");
    //    cv::imshow("test",outputImageC1);
    //    cv::waitKey(0);
    //    cv::destroyWindow("test");

    //std::vector<detectionResult*> dResult(0);
    detectionResult detectResult;

    std::cout << "show grand truth" << std::endl;
    //    std::cout << dataSet.className.size() << std::endl;
    //    std::cout << dataSet.centerPoint.size() << std::endl;
    for(int i = 0; i < testSet.param.size(); ++i){
        testSet.param.at(i).showParam();
    }

    for(int c = 0; c < classNum; ++c){

        double min,max;
        cv::Point minLoc,maxLoc;
        cv::minMaxLoc(outputImageColorOnly.at(c),&min,&max,&minLoc,&maxLoc);
        //double score  = (double)(outputImageColorOnly.at(c).at<int>(maxLoc.y,maxLoc.x) / (double)( conf.stride * conf.stride * outputImage.at(c).cols * outputImage.at(c).rows)) * 1000000;

        //cv::circle(outputImage.at(c),maxLoc,20,cv::Scalar(200,0,0),3);

        //if(score > conf.detectThreshold){
        //        cv::namedWindow("test");
        //        cv::imshow("test",outputImage.at(c));
        //        cv::waitKey(0);
        //        cv::destroyWindow("test");
        cv::Size tempSize = classDatabase.vNode.at(c).classSize;
        cv::Rect_<int> outRect(maxLoc.x - tempSize.width / 2,maxLoc.y - tempSize.height / 2 , tempSize.width,tempSize.height);
        cv::rectangle(outputImage.at(0),outRect,cv::Scalar(0,0,200),3);
        cv::putText(outputImage.at(0),classDatabase.vNode.at(c).name,cv::Point(outRect.x,outRect.y),cv::FONT_HERSHEY_SIMPLEX,1.2, cv::Scalar(0,0,200), 2, CV_AA);
        //}

        // display grand truth
        if(conf.showGT){
            for(int i = 0; i < testSet.param.size(); ++i){
                int tempClassNum = classDatabase.search(testSet.param.at(i).getClassName());
                if(tempClassNum != -1){
                    cv::Size tempSize = classDatabase.vNode.at(tempClassNum).classSize;
                    cv::Rect_<int> outRect(testSet.param.at(i).getCenterPoint().x - tempSize.width / 2,testSet.param.at(i).getCenterPoint().y - tempSize.height / 2 , tempSize.width,tempSize.height);
                    cv::rectangle(outputImage.at(0),outRect,cv::Scalar(200,0,0),3);
                    cv::putText(outputImage.at(0),classDatabase.vNode.at(c).name,cv::Point(testSet.param.at(i).getCenterPoint().x, testSet.param.at(i).getCenterPoint().y),cv::FONT_HERSHEY_SIMPLEX,1.2, cv::Scalar(200,0,0), 2, CV_AA);
                }
            }
        }

        cv::imshow("outputImage", outputImage.at(0));

        std::cout << c << "\tName : " << classDatabase.vNode.at(c).name << "\tvote : " << totalVote.at(c) << "\tScore : " << outputImageColorOnly.at(c).at<float>(maxLoc.y, maxLoc.x) << "\tCenterPoint : " << maxLoc << std::endl;//"\tscore : " << score << std::endl;

//        std::string outputName = opath + PATH_SEP + "detectionResult" + "_" + classDatabase.vNode.at(c).name + ".png";

//        cv::imwrite(outputName.c_str(),outputImage.at(c));
        detectResult.className = classDatabase.vNode.at(c).name;

        //double zizyoHeikin = (double)((maxLoc.x - testSet.param.at(0).getCenterPoint().x) ^ 2)
        //                               +  (double)((maxLoc.x - testSet.param.at(0).getCenterPoint().x) ^ 2);

//        if(zizyoHeikin != 0)
//            detectResult.error = std::sqrt(zizyoHeikin);
//        else
//            detectResult.error = 0;

//        double saidaikyori = (double)((testSet.param.at(0).getCenterPoint().x)^2) + (double)((testSet.param.at(0).getCenterPoint().y)^2);
//        detectResult.score = outputImageColorOnly.at(c).at<float>(maxLoc.y, maxLoc.x);
//        if(detectResult.error >= saidaikyori != 0 ? std::sqrt( saidaikyori) : 0)
//            detectResult.found = 0;
//        else
//            detectResult.found = 1;
    }

//    return detectResult;

    //testSet.releaseImage();
    //testSet.releaseFeatures();

    //cv::waitKey(0);
}

// Regression 
void CRForest::regression(std::vector<const LeafNode*>& result, CTestPatch &patch) const{
    result.resize( vTrees.size() );
    //std::cout << "enter regression" << std::endl;
    for(int i=0; i < vTrees.size(); ++i) {
        //std::cout << "regressioning " << i << std::endl;
        result[i] = vTrees[i]->regression(patch);
    }
}

//void CRForest::loadImages(cv::vector<cv::vector<cv::Mat *> > &img, std::vector<CDataset> &dataSet){
//    img.resize(0);

//    cv::Mat* rgb,*depth;//, *mask;
//    cv::vector<cv::Mat*> planes;
//    cv::vector<cv::Mat*> allImages;
//    //cv::vector<cv::Mat> rgbSplited;

//    for(int i = 0;i < dataSet.size(); ++i){
//        rgb = new cv::Mat();
//        depth = new cv::Mat();
//        //mask = new cv::Mat();

//        // load Mask image

//        //*mask = cv::imread(dataSet.at(i).imageFilePath
//        //                   + dataSet.at(i).maskImageName,3).clone();

//        // load RGB image
//        *rgb = cv::imread(dataSet.at(i).imageFilePath
//                          + dataSet.at(i).rgbImageName,3).clone();

//        //std::cout << dataSet.at(i).rgbImageName << " " << rgb->channels() << std::endl;
//        // load Depth image
//        *depth = cv::imread(dataSet.at(i).imageFilePath
//                            + dataSet.at(i).depthImageName,
//                            CV_LOAD_IMAGE_ANYDEPTH).clone();
//        cv::Point tempPoint;
//        tempPoint.x = (*rgb).cols / 2;
//        tempPoint.y = (*rgb).rows / 2;

//        dataSet.at(i).centerPoint.push_back(tempPoint);

//        //        cv::namedWindow("test");
//        //        cv::imshow("test",*rgb);
//        //        cv::waitKey(0);
//        //        cv::destroyWindow("test");



//        //std::cout << depth << std::endl;



//        for(int k = 0;k < rgb->cols; ++k)
//            for(int l = 0;l < rgb->rows; ++l){
//                //std::cout << depth.at<ushort>(l, k) << " " << std::endl;
//                //if(!(bool)mask->at<char>(l, k))
//                //depth->at<ushort>(l, k) = 0;
//                // for(int j = 0;j < 3; ++j)
//                //   if(!(bool)mask.at<char>(l, k))
//                //     rgb.at<cv::Vec3b>(l, k)[j] = 0;
//            }
//        //rgbSplited.resize(rgb.channels());

//        //cv::split(rgb, rgbSplited);

//        allImages.clear();
//        allImages.push_back(rgb);
//        allImages.push_back(depth);
//        img.push_back(allImages);

//        //delete mask;
//    }


//}

