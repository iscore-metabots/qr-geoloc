CC=g++

INCLUDE_FLAGS=  -I/usr/local/include \
			    -I/usr/include
LIB_FLAGS=  -L/usr/local/lib/opencv \
			-L/usr/lib \
			-lopencv_core \
			-lopencv_highgui \
			-lopencv_imgproc \
			-lopencv_features2d \
			-lopencv_gpu \
			-lzbar

SOURCES = qr-track.cpp
EXECUTABLE = qr-track.xc
$(EXECUTABLE): $(SOURCES)
	$(CC) -o $(EXECUTABLE) $(SOURCES) $(INCLUDE_FLAGS) $(LIB_FLAGS)