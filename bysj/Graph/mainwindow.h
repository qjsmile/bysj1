#ifndef MAINWINDOW_H
#define MAINWINDOW_H
#include <QMainWindow>
#include<QLabel>
#include<cv.h>
#include<highgui.h>
using namespace cv;
namespace Ui {
class MainWindow;
}
class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = 0);

    ~MainWindow();
private slots:
    void OpenPicture();
    void SavePicture();
    void Auto();
    void displayMat(Mat image);

private:
    Ui::MainWindow *ui;
    QLabel *imagelabel;
    void getBinMask( const Mat& comMask, Mat& binMask );
    void Train();
    void Check();
    void MyMessageBox();

};

#endif // MAINWINDOW_H
