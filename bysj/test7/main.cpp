#include <iostream>
#include <cv.h>
#include <highgui.h>
#include<ml.h>
#include<fstream>
#include "MySvm.h"
using namespace std;
using namespace cv;
void getBinMask( const Mat& comMask, Mat& binMask )
{
    binMask.create( comMask.size(), CV_8UC1 );
    binMask = comMask & 1;
}
void Train()//提取样本Hog特征训练生成训练器文件
{
    char classifierPath[256] = "resource\\classfifier.txt";
    string posPath = "resource\\pos3\\";
    string negPath = "resource\\neg4\\";

    int posCount =20;
    int negCount =40;
    int totalCount = posCount + negCount;

    CvMat *sampleFeaturesMat = cvCreateMat(totalCount , 1764, CV_32FC1); //64*64的训练样本，4 * 7 * 7 * 9=1764,该矩阵将是totalSample*1764
    CvMat *sampleLabelMat = cvCreateMat(totalCount, 1, CV_32FC1);//样本标识
    cvSetZero(sampleFeaturesMat);
    cvSetZero(sampleLabelMat);

    cout<<"************************************************************"<<endl;
    cout<<"start to training positive samples..."<<endl;
    char posImgName[256];
    string imgPath;
    for(int i=0; i<posCount; i++)
    {
        memset(posImgName, '\0', 256*sizeof(char));
        sprintf(posImgName, "%d.jpg", i);
        int len = strlen(posImgName);
        string tempStr = posImgName;
        imgPath = posPath + tempStr;
        cv::Mat img = cv::imread(imgPath);
        if( img.data == NULL )
        {
            cout<<"positive image sample load error: "<<i<<" "<<imgPath<<endl;
            system("pause");
            continue;
        }

        cv::HOGDescriptor hog(cv::Size(64,64), cv::Size(16,16), cv::Size(8,8), cv::Size(8,8), 9);
        vector<float> featureVec;//特征数组

        hog.compute(img, featureVec, cv::Size(8,8));
        int featureVecSize = featureVec.size();
        for (int j=0; j<featureVecSize; j++)
        {
            CV_MAT_ELEM( *sampleFeaturesMat, float, i, j ) = featureVec[j];
        }
        sampleLabelMat->data.fl[i] = 1;
    }
    cout<<"end of training for positive samples..."<<endl;

    cout<<"*********************************************************"<<endl;
    cout<<"start to train negative samples..."<<endl;
    char negImgName[256];
    for (int i=0; i<negCount; i++)
    {
        memset(negImgName, '\0', 256*sizeof(char));
        sprintf(negImgName, "%d.jpg", i);
        imgPath = negPath + negImgName;
        cv::Mat img = cv::imread(imgPath);
        if(img.data == NULL)
        {
            cout<<"negative image sample load error: "<< imgPath<<endl;
            continue;
        }

        cv::HOGDescriptor hog(cv::Size(64,64), cv::Size(16,16), cv::Size(8,8), cv::Size(8,8), 9);
        vector<float> featureVec;

        hog.compute(img,featureVec,cv::Size(8,8));//计算HOG特征
        int featureVecSize = featureVec.size();

        for( int j=0; j<featureVecSize; j++)
        {
             CV_MAT_ELEM( *sampleFeaturesMat, float, i + posCount, j ) = featureVec[ j ];
        }

        sampleLabelMat->data.fl[ i + posCount ] = -1;
    }
    cout<<"end of training for negative samples..."<<endl;
    cout<<"********************************************************"<<endl;

    MySvm svm;
    CvSVMParams params;
    params.svm_type = CvSVM::C_SVC;
    params.kernel_type = CvSVM::RBF;
    params.term_crit = cvTermCriteria(CV_TERMCRIT_ITER, 2000, FLT_EPSILON);
    params.C = 1;
    svm.train( sampleFeaturesMat, sampleLabelMat, NULL, NULL, params );
    svm.save(classifierPath);
    cvReleaseMat(&sampleFeaturesMat);
    cvReleaseMat(&sampleLabelMat);

}

void Classfifier()
{
    MySvm svm;
    svm.load("resource\\classfifier.txt");
    int DescriptorDim = svm.get_var_count();//特征向量的维数，即HOG描述子的维数
    int supportVectorNum = svm.get_support_vector_count();//支持向量的个数
    CvMat *alphaMat= cvCreateMat(1, supportVectorNum, CV_32FC1);
    CvMat *supportVectorMat= cvCreateMat(supportVectorNum,DescriptorDim,CV_32FC1);
    CvMat *resultMat= cvCreateMat(1, DescriptorDim,CV_32FC1);
    cvSetZero(alphaMat);
    cvSetZero(supportVectorMat);
    cvSetZero(resultMat);
    //将alpha向量的数据复制到alphaMat中
    double * pAlphaData = svm.get_alpha();//得到SVM的决策函数中的alpha向量
    for(int i=0; i<supportVectorNum; i++)
    {
       alphaMat->data.fl[i] = pAlphaData[i];
    }

    //将支持向量的数据复制到supportVectorMat矩阵中
    for(int i=0; i<supportVectorNum; i++)
    {
       const float * pSVData = svm.get_support_vector(i);//返回第i个支持向量的数据指针
       for(int j=0; j<DescriptorDim; j++)
       {
          CV_MAT_ELEM( *supportVectorMat, float, i, j ) =pSVData[j];
       }
    }
    cvMatMul(alphaMat, supportVectorMat, resultMat);
    for (int i=0; i<1764; i++)
    {
        resultMat->data.fl[i] *= -1;
    }


    vector<float> myDetector;
     //将resultMat中的数据复制到数组myDetector中
     for(int i=0; i<DescriptorDim; i++)
     {
       myDetector.push_back(resultMat->data.fl[i]);
     }
     //最后添加偏移量rho，得到检测子
     myDetector.push_back(svm.get_rho());
     cout<<"the dim is"<<myDetector.size()<<endl;

     ofstream fout("resource\\hogclassfifier1.txt");
     for(int i=0; i<myDetector.size(); i++)
     {
        fout<<myDetector[i]<<endl;
     }

}
     //检测
void Check()
{
    vector<float> myDetector;
    ifstream fin("resource\\hogclassfifier1.txt");
    float d;
    while(fin>>d)
    {
      myDetector.push_back(d);
    }
     vector<cv::Rect>  found;
     cv::HOGDescriptor hog1(cv::Size(64,64), cv::Size(16,16), cv::Size(8,8), cv::Size(8,8), 9);
     hog1.setSVMDetector(myDetector);
     Mat img3=imread("d:\\bspic\\46.jpg");
     Mat showImg = img3.clone();
     Mat bg;
     Mat fg;
     Mat mask,res;
     Mat binMask;
     hog1.detectMultiScale(img3, found, 0, cv::Size(8,8), cv::Size(32,32), 1.05, 2);
     cout<<"the found size is"<<found.size()<<endl;
     if (found.size() > 0)
       {
         for (int i=0; i<found.size(); i++)
         {
           CvRect tempRect = cvRect(found[i].x, found[i].y, found[i].width, found[i].height);
           Rect rectcut = Rect(cvPoint(tempRect.x-5,tempRect.y-5), cvPoint(tempRect.x+tempRect.width+5,tempRect.y+tempRect.height+5));
           rectangle(showImg, cvPoint(tempRect.x-5,tempRect.y-5),
                     cvPoint(tempRect.x+tempRect.width+5,tempRect.y+tempRect.height+5),CV_RGB(255,0,0), 2);
           mask.create( img3.size(), CV_8UC1);
           grabCut(img3, mask, rectcut, bg, fg, 3, 0 );
           getBinMask( mask, binMask );
           img3.copyTo( res, binMask );

         }
         imshow("leaf",showImg);
         imshow("1",res);
       }

    // imwrite("D:\\A.jpg",res);

     cvWaitKey(0);


}
void Predict()
{
    MySvm svm;
    svm.load("resource\\classfifier.txt");

    //predict
    CvMat *testFeaturesMat = cvCreateMat(1, 1764, CV_32FC1);
    cv::Mat img2 = cv::imread("d:\\svm\\pos1\\21.jpg");
    cv::HOGDescriptor hog(cv::Size(64,64), cv::Size(16,16), cv::Size(8,8), cv::Size(8,8), 9);
    vector<float> testVec;
    hog.compute(img2,testVec,cv::Size(8,8));//计算HOG特征
    int testVecSize = testVec.size();
    for (int j=0; j<testVecSize; j++)
    {
        CV_MAT_ELEM( *testFeaturesMat, float, 0, j ) = testVec[j];
    }
    float ret = svm.predict(testFeaturesMat);
    cout<<"the ret is"<<ret<<endl;
}


int main()
{
    Train();
    Classfifier();
    Check();
   // Predict();
    cvWaitKey(0);
    return 0;
}
