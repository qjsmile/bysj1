#-------------------------------------------------
#
# Project created by QtCreator 2014-03-09T19:48:39
#
#-------------------------------------------------

QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = Graph
TEMPLATE = app


SOURCES += main.cpp\
        mainwindow.cpp

HEADERS  += mainwindow.h

FORMS    += mainwindow.ui
INCLUDEPATH+=D:\opencv\include\opencv\
             D:\opencv\include\opencv2\
             D:\opencv\include
LIBS+=-LD:\opencv\lib -lopencv_core248 -lopencv_highgui248 -lopencv_imgproc248 -lopencv_objdetect248  -lopencv_ml248
