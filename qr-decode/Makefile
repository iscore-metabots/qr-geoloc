CC=g++

INCLUDE_FLAGS= \
	-I/usr/local/include \
	-I/usr/include
LIB_FLAGS= \
	-L/usr/local/lib \
	-L/usr/lib \
	-lopencv_core \
	-lopencv_highgui \
	-lopencv_imgproc \
	-lopencv_features2d \
	-lzbar

SOURCES = qr-decode.cpp
EXECUTABLE = qr-decode.xc
$(EXECUTABLE): $(SOURCES)
	$(CC) -o $(EXECUTABLE) $(SOURCES) $(INCLUDE_FLAGS) $(LIB_FLAGS)