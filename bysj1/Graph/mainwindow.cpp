#include "mainwindow.h"
#include "ui_mainwindow.h"
#include<fstream>
#include"MySvm.h"
#include<cv.h>
#include<highgui.h>
#include<Qgraphicsview>
#include<QFileDialog>
#include<QMessageBox>
#include<QPushButton>
using namespace std;
using namespace cv;
QString filename,savename;
Rect select;
bool select_flag=false;
Mat img,img3,showImg,resImg;
QImage qimg;
MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    QPalette win=palette();
    win.setBrush(QPalette::Window,QBrush(QColor(245,230,245)));
    setPalette(win);
    ui->actionOpen->setIcon(QIcon("resource\\open.jpg"));
    ui->actionSave->setIcon(QIcon("resource\\save.jpg"));
    imagelabel=new QLabel(this);
    imagelabel->setGeometry(10,40,40,40);
    connect(ui->actionOpen,SIGNAL(triggered()),this,SLOT(OpenPicture()));
    connect(ui->actionAutomatic,SIGNAL(triggered()),this,SLOT(Auto()));
    connect(ui->actionSave,SIGNAL(triggered()),this,SLOT(SavePicture()));
   // connect(ui->actionClose, SIGNAL(clicked()), this, SLOT(quit()) );
}
MainWindow::~MainWindow()
{
    delete imagelabel;
    delete ui;

}
void MainWindow::getBinMask( const Mat& comMask, Mat& binMask )
{
    binMask.create( comMask.size(), CV_8UC1 );
    binMask = comMask & 1;
}

void MainWindow::OpenPicture()
{
    filename = QFileDialog::getOpenFileName(0,
                                            "Open Image",
                                            "testpicture",
                                            "Image Files (*.png *.jpg *.bmp)");

    img3=imread((const char *)filename.toLocal8Bit());
    displayMat(img3);
    //imshow("img",img3);
}
void MainWindow:: Check()
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
     Mat showImg = img3.clone();
     Mat bg;
     Mat fg;
     Mat mask,res;
     Mat binMask;
     hog1.detectMultiScale(img3, found, 0, cv::Size(4,4), cv::Size(4,4), 1.05, 2);
   //  cout<<"the found size is"<<found.size()<<endl;
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
         //imshow("leaf",showImg);
         //imshow("result",res);
         displayMat(res);
         resImg=res.clone();
       }
     else
     {
            MyMessageBox();
     }

     //cvWaitKey(0);
    // delete dialog;


}
void MainWindow::Auto()
{
    Check();
}
void MainWindow::displayMat(Mat image)
{
    Mat rgb;
    if(image.channels()==3)
    {
        //cvt Mat BGR 2 QImage RGB
        cvtColor(image,rgb,CV_BGR2RGB);
        qimg =QImage((const unsigned char*)(rgb.data),
                    rgb.cols,rgb.rows,
                    rgb.cols*rgb.channels(),
                    QImage::Format_RGB888);
    }
    else
    {
        qimg =QImage((const unsigned char*)(image.data),
                    image.cols,image.rows,
                    image.cols*image.channels(),
                    QImage::Format_RGB888);
    }
    imagelabel->setPixmap(QPixmap::fromImage(qimg));
    imagelabel->resize(imagelabel->pixmap()->size());
    //imagelabel->

}
void MainWindow:: MyMessageBox()
{
    QPixmap iconImg("resource\\sorry.jpg");
    QIcon icon(iconImg);
    QMessageBox msgBox;
    msgBox.setWindowIcon(icon);
    msgBox.setWindowTitle("提示信息");
    msgBox.setIconPixmap(iconImg);
    msgBox.exec();
}
void MainWindow::SavePicture()
{
   savename = QFileDialog::getSaveFileName(0,
                                           "Open Image",
                                            "savepicture",
                                            "Image Files (*.png *.jpg *.bmp)");
   imwrite((const char *)savename.toLocal8Bit(),resImg);
}
