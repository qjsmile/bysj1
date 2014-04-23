TEMPLATE = app
CONFIG += console
CONFIG -= app_bundle
CONFIG -= qt

SOURCES += main.cpp

INCLUDEPATH+=D:\opencv\include\opencv\
             D:\opencv\include\opencv2\
             D:\opencv\include
LIBS+=-LD:\opencv\lib -lopencv_core248 -lopencv_highgui248 -lopencv_imgproc248 -lopencv_objdetect248  -lopencv_ml248

HEADERS += \
    MySvm.h
