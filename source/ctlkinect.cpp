#include "ctlkinect.h"

// setting file place
#define CONFIG_PATH "./xtionConfig.xml"

using namespace xn;

CtlKinect::CtlKinect()
{
    // init kinect
    rc = XN_STATUS_OK;
    //open XML file
    rc = g_context.InitFromXmlFile(CONFIG_PATH);
    //rc = g_context.InitFromXmlFile(CONFIG_PATH, g_scriptNode, &errors);
    //if(rc != XN_STATUS_OK)
        //emit errorOccurred(0);

    //generate camear
    rc = g_context.FindExistingNode(XN_NODE_TYPE_IMAGE, g_image);
//    if(rc != XN_STATUS_OK)
//        emit errorOccurred(1);

    //generate depth
    rc = g_context.FindExistingNode(XN_NODE_TYPE_DEPTH, g_depth);
//    if (rc != XN_STATUS_OK)
//        emit errorOccurred(2);

    XnMapOutputMode mapMode;

    rc = g_depth.GetMapOutputMode(mapMode);
//    if (rc != XN_STATUS_OK)
//        emit errorOccurred(2);

//    qDebug() << tr("mapMode.nYRes is %1").arg(mapMode.nYRes);

//    rgb = new cv::Mat(mapMode.nYRes,mapMode.nXRes,CV_8UC3);
//    depth = new cv::Mat(mapMode.nYRes,mapMode.nXRes,CV_16UC1);
//    qDebug() << tr("%1 %1").arg(rgb->step,rgb->rows);

}

CtlKinect::~CtlKinect(){
//    delete rgb;
//    delete depth;
}

void CtlKinect::getRGBDData(cv::Mat* rgb, cv::Mat* depth){
    //qDebug() << tr("start getting data");


    //cv::namedWindow("imagewindow");

    //while(1){
        rc = g_context.WaitAnyUpdateAll();
//        if(rc != XN_STATUS_OK){
//          emit errorOccurred(3);
//          return;
//        }

        //各種generatorからメタデータを取得しています
        g_depth.GetMetaData(g_depthMD);
        g_depth.GetAlternativeViewPointCap().SetViewPoint(g_image);
        g_image.GetMetaData(g_imageMD);


        //それぞれOpenCVの画像形式に変換しています
        memcpy(rgb->data,g_imageMD.Data(),rgb->step*rgb->rows);
        memcpy(depth->data,g_depthMD.Data(),depth->step*depth->rows);

        //そのままでは正しい色情報を持っていないので，OpenCVを使って変換してやります
        cv::cvtColor(*rgb,*rgb,CV_RGB2BGR);

        //cv::imshow("imagewindow", *rgb);

//        emit getData(rgb, depth);
    //}

}







