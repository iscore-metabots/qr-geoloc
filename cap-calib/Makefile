CC=g++

INCLUDE_FLAGS=-I/usr/local/include
LIB_FLAGS=  -L/usr/local/lib/opencv \
			-lopencv_core \
			-lopencv_highgui \

SOURCES = cap-calib.cpp
EXECUTABLE = cap-calib.xc
$(EXECUTABLE): $(SOURCES)
	$(CC) -o $(EXECUTABLE) $(SOURCES) $(INCLUDE_FLAGS) $(LIB_FLAGS)