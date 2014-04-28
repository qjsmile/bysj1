#include <iostream>
#include <cv.h>
#include <highgui.h>
#include<ml.h>
#include<fstream>
using namespace std;
using namespace cv;
const string b_imgpath="resource\\pos\\";//存放需处理的图片的文件
const string a_imgpath="resource\\a_pos\\";//存放处理后的图片的文件
string imgname="";
string imgpath="";
Mat src; //源图像
Mat dst; //目标图像
void ImgPrePro(string imgname);
int main()
{
    ifstream imgfile("resource\\pos.txt");
    if(imgfile)
    {
        while (getline (imgfile, imgname))
        {
            ImgPrePro(imgname);
        }
    }
    cout<<"ok"<<endl;
    return 0;
}
void ImgPrePro(string imgname)
{
    imgpath=b_imgpath+imgname;
    src =imread(imgpath);
    if( src.data == NULL )
    {
        cout<<" image sample load error: "<<imgpath<<endl;
    }

    //尺寸处理为64*64
    Size dsize = Size(64,64);
    dst = Mat(dsize,CV_32S);
    resize(src,dst,dsize);
    //灰度化处理
    Mat gray;
    cvtColor(dst,gray,CV_BGR2GRAY);
    imgpath=a_imgpath+imgname;
    imwrite(imgpath,gray);
}

