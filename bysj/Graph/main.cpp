#include "mainwindow.h"
#include<iostream>
#include <QApplication>
#include<QLabel>
#include<QPushButton>
#include<QHBoxLayout>
#include<QFileDialog>
#include<QSplashScreen>
#include<QDateTime>
using namespace std;
int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    MainWindow w;
    w.setWindowTitle("LeafCut");
    w.show();
    return a.exec();
}


/*QSplashScreen *splash = new QSplashScreen;
splash->setPixmap(QPixmap("d:\\bspic\\sorry.jpg"));
splash->show();

QDateTime n=QDateTime::currentDateTime();
QDateTime now;
do{
now=QDateTime::currentDateTime();
} while (n.secsTo(now)<=3);//6为需要延时的秒数
splash->finish(&w);
delete splash;
*/
