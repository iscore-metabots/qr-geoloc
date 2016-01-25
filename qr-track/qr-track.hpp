using namespace std;
using namespace cv;
using namespace zbar;



/*
  detectGPU
  Function looking for compatible GPUs up to index 9
    dInfo: output
      VideoCapture object corresponding to the first found camera
    index: input output
      As input: first camera index to try
      As output: last camera index tried, index of the first found camera if applicable
    Returns if the detection was successful
*/
bool detectGPU(int& dIndex);



/*
  readProj
  Function importing reprojection data as a transformation matrix from a YML file
    filename: input
      Full path and name to the YML file to read
    M: output
      OpenCV matrix to return transformation matrix
    Returns if the data import was successful
*/
bool readProj( const char* filename, Mat& M);



/*
  readScene
  Function importing scene reference data from a YML file
    filename: input
      Full path and name to the YML file to read
    scnsize: output
      Dimensions of the scene
    Returns if the data import was successful
*/
bool readScene( const char* filename, Size& scnsize);



/*
  openCam
  Function attempting to connect to a camera at up to ten different indices
    videocap: output
      VideoCapture object corresponding to the first found camera
    index: input output
      As input: first camera index to try
      As output: last camera index tried, index of the first found camera if applicable
    Returns if the program could connect to a camera
*/
bool openCam(VideoCapture& videocap, int& index);



/*
  openAVI
  Function attempting to open an AVI video file
    videocap: output
      VideoCapture object corresponding to the opened file
    path: input
      Full path and name to the AVI file to open
    Returns if the program could open the video file
*/
bool openAVI(VideoCapture& videocap, char* path);



/*
  loadData
  Function loading and checking all required data
    projname: input
      Full path and name to the YML file from which to get reprojection data
      About required YML structure, refer to example file
    scnname: input
      Full path and name to the YML file from which to get scene reference data
      About required YML structure, refer to example file
    source: input
      String indicating which source will be used : AVI file or camera
      source should be a full path to an AVI file
      or an integer corresponding to the index of the first camera to try to connect to
    M: output
      Loaded transformation matrix
    scnsize: output
      Loaded dimensions of the scene
    videocap: output
      VideoCapture object corresponding to the loaded video source
    Returns if the data loading was successful
*/
bool loadData(const char* projname, const char* scnname, char* source, Mat& M, Size& scnsize, VideoCapture& videocap);



/*
  Ctrl-C interruption handling
*/
void interrupt_loop(int sig);



/*
  scan
  Function scanning an image taken from a calibrated camera to identify QR or bar codes
    M: input
      Transformation matrix to reproject the images from the video stream
    scnsize: input
      Dimensions of the scene, bounding the reprojected images
    videocap: input
      VideoCapture object corresponding to the video source
*/
int scan(Mat M, Size scnsize, VideoCapture& videocap);



/*
  scanGPU
  Function scanning an image taken from a calibrated camera to identify QR or bar codes
    Mostly same usage as 'scan'
    Uses GPU-accelerated computing to process the video stream faster
  dIndex: input
    Index of the GPU device to enable
*/
int scanGPU(Mat M, Size scnsize, VideoCapture& videocap, const int dIndex);