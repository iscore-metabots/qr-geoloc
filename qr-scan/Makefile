CC=g++-5

INCLUDE_FLAGS= \
	-I/usr/local/include \
	-I/usr/local/include/OssiaAPI \
	-I/usr/include
LIB_FLAGS= \
	-L/usr/local/lib \
	-L/usr/lib \
	-L/usr/local/jamoma/lib \
	-L/usr/lib/jamoma \
	-L/mnt/data/PERSO/Boulot/ENSC3A/ISCORE/build-ossia/Implementations/Jamoma \
	-lopencv_core \
	-lopencv_highgui \
	-lopencv_imgproc \
	-lopencv_features2d \
	-lopencv_gpu \
	-lzbar \
	-lJamomaFoundation \
	-lJamomaModular \
	-lAPIJamoma

SOURCES = network.cpp qr-scan.cpp
EXECUTABLE = qr-scan.xc
$(EXECUTABLE): $(SOURCES)
	$(CC) -std=c++11 -o $(EXECUTABLE) $(SOURCES) $(INCLUDE_FLAGS) $(LIB_FLAGS) -LD_PATH_LIBRARY=/mnt/data/PERSO/Boulot/ENSC3A/ISCORE/build-ossia/Implementations/Jamoma/libAPIJamoma.so