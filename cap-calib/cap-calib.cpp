#include <iostream>
#include <stdlib.h>
using namespace std;

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/features2d/features2d.hpp"
using namespace cv;


/*
  openCam
  Function attempting to connect to a camera up to index 9
    videocap: output
      VideoCapture object corresponding to the first found camera
    index: input output
      As input: first camera index to try
      As output: last camera index tried, index of the first found camera if applicable
*/
static bool openCam(VideoCapture& videocap, int& index)
{
  bool opened = false;
    int maxindex = index + 10;

  // Try to open a camera among the ten first found
    while (!opened && (index < maxindex)) {
    videocap = VideoCapture(index);
    opened = videocap.isOpened();
    index++;
  }

  index--; // Get the last index actually used

  return opened;
}



/*
  process
    camindex: input
      Index of the camera to connect to
    savename: input
      Full path and name to the file where to save the captured image
*/
int process(int camindex, const char* savename)
{
  VideoCapture videocap;
  bool cap_opened = cap_opened = openCam(videocap, camindex);
  cout << ( cap_opened ? "Camera connection successfully opened at index " : "Failed to connect to camera! Final index: ") << camindex << endl;

  if (! cap_opened) {
    cerr << "Aborting capture..." << endl;
    exit(EXIT_FAILURE);
  }
  else {
    Mat frame;
    videocap >> frame;
    imwrite(savename, frame);
    cout << "Calibration image successfully saved at: " << savename << endl;

    return EXIT_SUCCESS;
  }
}



#define param 2
#define bound "# -----------------------------------"

int main(int args, char* argv[])
{
  if (args != param + 1) {
    if (args < param + 1) {
      cout << "Too few arguments!";
    }
    else {
      cout << "Too many arguments!";
    }
    cerr << " Number given: " << args - 1 << endl << "Usage: cap-calib <cam-index> <im-name>" << endl;
    return EXIT_FAILURE;
  }
  else {
    cout << bound << endl << "Image calibration capture" << endl << bound << endl << endl;
    return process(atoi(argv[1]), argv[2]);
  }
}
