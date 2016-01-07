CC=g++

INCLUDE_FLAGS= \
	-I/usr/local/include
LIB_FLAGS= \
	-L/usr/local/lib \
	-lopencv_core \
	-lopencv_highgui \
	-lopencv_imgproc \
	-lopencv_calib3d \
	-lopencv_features2d

SOURCES = chess-calib.cpp
EXECUTABLE = chess-calib.xc
$(EXECUTABLE): $(SOURCES)
	$(CC) -o $(EXECUTABLE) $(SOURCES) $(INCLUDE_FLAGS) $(LIB_FLAGS)