
#include <iostream> // Console outputs
#include <vector>
#include <string>
#include <signal.h> // Keyboard interruption
#include <math.h>   // atan2
#define PI 3.1415927 
using namespace std;

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/gpu/gpu.hpp"
using namespace cv;

#include <zbar.h>
using namespace zbar;

/*#include <memory>
#include <functional>
#include "Network/Address.h"
#include "Network/Device.h"
#include "Network/Protocol/Local.h"
#include "Network/Protocol/Minuit.h"
using namespace OSSIA; */



/*
  ItemData
  Structure aggregating geometric and ID data about each detected item in the image
    X: horizontal coordinate of the item
    Y: vertical coordinate of the item
    theta: orientation of the item
    ID: number identifying the item, as a positive integer
*/
struct ItemData {
  float X;
  float Y;
  float theta;
  char ID;
} ;



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
bool detectGPU(int& dIndex)
{
  bool detected = false;
  gpu::DeviceInfo dInfo;

  // Try to get GPU info among the ten first found
  while (!detected && (dIndex < 10)) {
    dInfo = gpu::DeviceInfo(dIndex);
    detected = dInfo.isCompatible();
    dIndex++;
  }

  dIndex--; // Get the last index actually used

  if (detected)
    cout << "Detected GPU " << dInfo.name() << " at index " << dIndex << endl;

  return detected;
}



/*
  readProj
  Function importing reprojection data as a transformation matrix from a YML file
    filename: input
      Full path and name to the YML file to read
    M: output
      OpenCV matrix to return transformation matrix
    Returns if the data import was successful
*/
bool readProj( const char* filename, Mat& M)
{
  FileStorage fs(filename, FileStorage::READ);
  if( !fs.isOpened() )
    return false;
  
  FileNode Mn = fs["transform_mat"];
  if ( Mn.empty() )
    return false;

  // Get the transformation matrix
  Mn >> M;
  return true;
}



/*
  readScene
  Function importing scene reference data from a YML file
    filename: input
      Full path and name to the YML file to read
    scnsize: output
      Dimensions of the scene
    Returns if the data import was successful
*/
bool readScene( const char* filename, Size& scnsize)
{
  FileStorage fs(filename, FileStorage::READ);
  if ( !fs.isOpened() )
    return false;
  
  FileNode sizen = fs["Size"];
  if ( sizen.empty() )
    return false;

  // Get the scene's dimensions
  sizen >> scnsize;

  fs.release();
  return true;
}



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
bool openCam(VideoCapture& videocap, int& index)
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
  openAVI
  Function attempting to open an AVI video file
    videocap: output
      VideoCapture object corresponding to the opened file
    path: input
      Full path and name to the AVI file to open
    Returns if the program could open the video file
*/
bool openAVI(VideoCapture& videocap, char* path)
{
  videocap = VideoCapture(path);
  return videocap.isOpened();
}



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
bool loadData(const char* projname, const char* scnname, char* source, Mat& M, Size& scnsize, VideoCapture& videocap)
{
  // Load transformation matrix and scene data from reference files
  bool proj_loaded = readProj(projname, M);
  cout << ( proj_loaded ? "Reprojection data successfully loaded from: " : "Failed to load reprojection data from: ") << projname << endl;

  bool scn_loaded = readScene(scnname, scnsize);
  cout << ( scn_loaded ? "Scene data successfully loaded from: " : "Failed to load scene data from: ") << scnname << endl;

  // Open the video source
  bool cap_opened = false;
  string src(source);

  if(src.substr(src.find_last_of(".") + 1) == "avi") {
    cout << "Source detected: AVI video file." << endl;
    cap_opened = openAVI(videocap, source);
    cout << ( cap_opened ? "Video successfully opened at: " : "Failed to open video file at: ") << src << endl;
  }
  else {
    cout << "Source detected: camera." << endl;
    int camindex = atoi(source);
    if (camindex < 0) {
      cerr << "Camera index given is invalid: " << source << ". Positive integer expected." << endl;
      exit(EXIT_FAILURE);
    }
    cap_opened = openCam(videocap, camindex);
    cout << ( cap_opened ? "Camera connection successfully opened at index " : "Failed to connect to camera! Final index: ") << camindex << endl;
  }

  return (cap_opened && proj_loaded && scn_loaded);
}



/*
  initNetwork
  Function initializing the network protocol for publishing geolocation data
  Standard name of the device is "qr-geoloc"
  Returns if the device creation was successful
*/
/*Device localDevice // Device corresponding to the program
vector< shared_ptr< Node > > nodes // Vector containing all top-level "Metabot.#" nodes
bool initNetwork()
{
  // Declare
  Local localProtocol = Local::create();
  localDevice = Device::create(localProtocol, "qr-geoloc");

  return true;
}



/*
  createTree
  Function creating a data tree describing the localization of a Metabot in the scene plane
  Standard architecture is
  /Metabot.#/
            /X    float, horizontal position coordinate
            /Y    float, vertical position coordinate
            /th   float, orientation angle
    ID: input
      Character corresponding to the ID of the Metabot
    Returns the top-level "Metabot.#" node
*
Node createTree(char ID)
{
  // Top-level node, root of the data tree
  Node metabotNode = *(localDevice->emplace(localDevice->children().cend(), "Metabot." + ID));

  // Localization of the item
  Node XNode = *(metabotNode->emplace(metabotNode->children().cend(), "X"));
  Address XAddress = XNode->createAddress(Value::Type::FLOAT);
  Node YNode = *(metabotNode->emplace(metabotNode->children().cend(), "Y"));
  Address YAddress = YNode->createAddress(Value::Type::FLOAT);
  Node thNode = *(metabotNode->emplace(metabotNode->children().cend(), "th"));
  Address thAddress = thNode->createAddress(Value::Type::FLOAT);

  shared_ptr< Node > sptr(metabotNode);
  nodes.push_back(sptr);

  return metabotNode
}



/*
  getNode
  Function accessing a node with the given ID
  or creating it if it doesn't exist yet
    ID: input
      Character corresponding to the ID of the Metabot
    Returns the top-level "Metabot.#" node with the given ID
*
Node getNode(char ID)
{
  Node resNode;
  bool found = false;
  // Check if a tree already exists for the given ID
  if (!found)
    resNode = createTree(ID);

  return resNode;
}



/*
  publishTree
  Function updating the data tree of the metabot with the given ID
    ID: input
      Character corresponding to the ID of the Metabot
    center: input
      Position of the center of the Metabot
    angle: input
      Orientation angle of the Metabot within the scene plane
    Returns if the publication was successful
*
bool publishTree(char ID, Point2f center, float angle)
{
  // Get nodes of the tree values
  Node metabotNode = getNode(ID);
  Container< Node > children = metabotNode.children();
  
  // Update tree values
  Float X(center.x);
  children[0].getAddress()->pushValue(&X);
  Float Y(center.y);
  children[1].getAddress()->pushValue(&Y);
  Float th(angle);
  children[2].getAddress()->pushValue(&th);

  return true;
}*/



/*
  Ctrl-C interruption handling
*/
bool loop_exit = false;
void interrupt_loop(int sig) // Whenever the user exits with Ctrl-C
{
  cout << endl << "Keyboard interruption catched. Terminating program..." << endl;
  loop_exit = true; // The programs exits the loop cleanly
}



/*
  process
  Function scanning an image taken from a calibrated camera to identify QR or bar codes
    M: input
      Transformation matrix to reproject the images from the video stream
    scnsize: input
      Dimensions of the scene, bounding the reprojected images
    videocap: input
      VideoCapture object corresponding to the video source
*/
int process(Mat M, Size scnsize, VideoCapture& videocap)
{
  Mat frame, gray; // Images that will be read and scanned
  bool frame_OK = false;

  int width = scnsize.width, height = scnsize.height; // Dimensions of the scene

  ImageScanner scanner; // Code scanner
  scanner.set_config(ZBAR_QRCODE, ZBAR_CFG_ENABLE, 1);

  /* # SHOW # Display current frame in a window
  namedWindow("Reprojected frame", 1);
  // # SHOW # */

  //* # HIGHLIGHT # Delimit detected symbols in the reprojected image
  Scalar color(0, 0, 255); // BGR pure red to highlight detected symbols
  namedWindow("Found symbols", 1);
  // # HIGHLIGHT # */

  // Main loop going through the video stream
  signal(SIGINT, interrupt_loop); // Register interruption signal

  while(! loop_exit) {
    frame_OK = videocap.read(frame);
    waitKey(1); // Allows the buffer to refresh
    if (! (frame.data && frame_OK)) {
      cerr << "Failed to load image from source!" << endl;
      exit(EXIT_FAILURE);
    }

    warpPerspective(frame, frame, M, scnsize); // Apply this transformation on the whole image
    cvtColor(frame, gray, CV_BGR2GRAY); // Get grayscale image for scanning phase

    /* # SHOW #
    imshow("Reprojected frame", frame);
    // # SHOW # */
    
    // Convert image from cv::Mat to zbar::Image
    uchar *raw = (uchar*) gray.data; // Raw image data
    Image image(width, height, "Y800", raw, width * height);
    // Using another syntax to call the same constructor seems to cause a systematic crash...
    
    // Scan for codes in the image
    int nsyms = scanner.scan(image);

    // Extract results

    //* # DATA # Write symbols' data in the console
    cout << nsyms << " symbol(s) found in the given image" << endl;
    // # DATA # */

    for(Image::SymbolIterator symbol = image.symbol_begin(); symbol != image.symbol_end(); ++symbol) {
      char ID = symbol->get_data()[0];
      
      int n = symbol->get_location_size();
      Point2f center, pNorth;
      for(int i = 0; i < n; i++) {
        Point2f p = Point2f(symbol->get_location_x(i),symbol->get_location_y(i));
        center += p;
        if ((i == 0) || (i == 3))
          pNorth += p;

        //* # HIGHLIGHT #
        circle(frame, p, 6, color, 2);
        // # HIGHLIGHT # */
      }

      center = 0.25 * center; // Center of the QRcode
      pNorth = 0.5 * pNorth; // Middle of the north west and north east points of the QR code
      float angle = atan2(pNorth.y - center.y, pNorth.x - center.x) * 180. / PI; // Angle of the QR code

      //* # HIGHLIGHT #
      arrowedLine(frame, center, pNorth, color, 2);
      // # HIGHLIGHT # */
      
      //* # DATA #
      cout << "Data: \"" << ID << "\" - Angle: " << angle << " - Center: " << center << endl;
      //publishTree(ID, center, angle);
      // # DATA # */
    }

    //* # HIGHLIGHT #
    imshow("Found symbols", frame);
    // # HIGHLIGHT # */
  }
  
  return EXIT_SUCCESS;
}



/*
  processGPU
  Function scanning an image taken from a calibrated camera to identify QR or bar codes
    Mostly same usage as process
    Uses GPU-accelerated computing to process the video stream faster
  dIndex: input
    Index of the GPU device to enable
*/
int processGPU(Mat M, Size scnsize, VideoCapture& videocap, const int dIndex)
{
  // Set detected GPU as used device
  gpu::setDevice(dIndex);

  // Images that will be read and scanned
  Mat frame, gray;
  gpu::GpuMat gframe, ggray;
  bool frame_OK = false;

  int width = scnsize.width, height = scnsize.height; // Dimensions of the scene

  ImageScanner scanner; // Code scanner
  scanner.set_config(ZBAR_NONE, ZBAR_CFG_ENABLE, 1);

  /* # SHOW # Display current frame in a window
  namedWindow("Reprojected frame", 1);
  // # SHOW # */

  // Main loop going through the video stream
  signal(SIGINT, interrupt_loop); // Register interruption signal

  while(! loop_exit) {
    frame_OK = videocap.read(frame);
    waitKey(1); // Allows the buffer to refresh
    if (! (frame.data && frame_OK)) {
      cerr << "Failed to load image from source!" << endl;
      exit(EXIT_FAILURE);
    }

    gframe.upload(frame);
    gpu::warpPerspective(gframe, gframe, M, scnsize); // Apply this transformation on the whole image
    gpu::cvtColor(gframe, ggray, CV_BGR2GRAY); // Get grayscale image for scanning phase
    ggray.download(gray);

    /* # SHOW #
    imshow("Reprojected frame", frame);
    // # SHOW # */
    
    uchar *raw = (uchar*) gray.data; // Raw image data
    Image image(width, height, "Y800", raw, width * height);
    
    // Scan for codes in the image
    int nsyms = scanner.scan(image);

    // Extract results

    //* # DATA # Write symbols' data in the console
    cout << nsyms << " symbol(s) found in the given image" << endl;
    // # DATA # */

    for(Image::SymbolIterator symbol = image.symbol_begin(); symbol != image.symbol_end(); ++symbol) {
      char ID = symbol->get_data()[0];

      int n = symbol->get_location_size();
      Point2f center, pNorth;
      for(int i = 0; i < n; i++) {
        Point2f p = Point2f(symbol->get_location_x(i),symbol->get_location_y(i));
        center += p;
        if (i < 2)
          pNorth += p;
      }

      center = 0.25 * center; // Center of the QRcode
      pNorth = 0.5 * pNorth; // Middle of the north west and north east points of the QR code
      float angle = atan2(pNorth.y - center.y, pNorth.x - center.x) * 180. / PI; // Angle of the QR code
      
      //* # DATA #
      cout << "Data: \"" << ID << "\" - Angle: " << angle << " - Center: " << center << endl;
      //publishTree(ID, center, angle);
      // # DATA # */
    }
  }

  return EXIT_SUCCESS;
}



#define param 3
#define bound "# -----------------------------------"
#define tryGPU false

int main(int args, char* argv[])
{
  if (args != param + 1) {
    if (args < param + 1) {
      cout << "Too few arguments!";
    }
    else {
      cout << "Too many arguments!";
    }
    cerr << " Number given: " << args - 1 << endl << "Usage: qr-track <calib-data.yml> <scn-data.yml> <video-source>" << endl;
    exit(EXIT_FAILURE);
  }
  else {
    cout << bound << endl << "QR tracker based on reprojection data" << endl << endl;
    Mat M;
    Size scnsize;
    VideoCapture videocap;

    if ( loadData(argv[1], argv[2], argv[3], M, scnsize, videocap) ) {
      bool useCPU = true;
      int dIndex = 0; // Try to detect a GPU on the computer
      if (tryGPU) {
        try {
          useCPU = !detectGPU(dIndex);

          if ( useCPU )
            cout << "No compatible GPU detected. Processing with CPU only..." << endl << bound << endl << endl;
          else
            cout << "Processing with GPU..." << endl << bound << endl << endl;
        }
        catch (cv::Exception& e) {
          cerr << e.what() << endl;
          cout << "ERROR: Could not search for compatible GPUs! This error can occur if OpenCV was not build with CUDA support, or if the user doesn't have the rights to access to the GPUs of the system." << endl;
          cout << "Processing with CPU only..." << endl << bound << endl << endl;
        }
      }
      else
      {
        cout << "\"tryGPU\" option disabled. Processing with CPU..." << endl << bound << endl << endl;
      }

      if( useCPU )
        return process(M, scnsize, videocap);
      else
        return processGPU(M, scnsize, videocap, dIndex);
    }
    else {
      cerr << endl << bound << endl << "Aborting scanning..." << endl;
      exit(EXIT_FAILURE); // In case of error during initialization phase, the program exits
    }
  }
}
