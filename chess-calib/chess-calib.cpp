#include <iostream> // Console outputs
#include <vector>
using namespace std;

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/imgproc/imgproc.hpp"
using namespace cv;

#include "chess-calib.hpp"



bool readRef( const char* filename, vector< Point2f >& corners, Size& boardsize)
{
  corners.resize(0); // Empty the output vector

  FileStorage fs(filename, FileStorage::READ);
  if( !fs.isOpened() )
    return false;
  
  FileNode sizen = fs["Size"], orign = fs["Origin"], stepn = fs["Step"];
  if ( sizen.empty() || orign.empty() || stepn.empty() )
    return false;

  // Get the size of the inner chessboard
  Size fullsize;
  sizen >> fullsize;
  boardsize = Size(fullsize.width - 1, fullsize.height - 1);

  Point2f orig;  // Get the board origin, i.e. top-left corner
  orign >> orig;
  double step; // Get the board step, i.e. square side length
  stepn >> step;
  
  double x0 = orig.x + step, y0 = orig.y + step;       // Fill the output array with the corners' positions
  for(int j = 0; j < boardsize.height; j++)  // Row by row
    for(int i = 0; i < boardsize.width; i++) // Then column by column
      corners.push_back(Point( x0 + i * step, y0 + j * step ) );

  fs.release();
  return true;
}



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



bool getCap( const char* source, Mat& ims)
{
  bool src_opened = false;
  string src(source), ext = src.substr(src.find_last_of(".") + 1);

  if ( (ext == "png") || (ext == "jpg") || (ext == "jpeg") || (ext == "PNG") || (ext == "JPG") || (ext == "JPEG") ) {
    // Supports only PNG and JPG files, but could be extended to other image file types that your OpenCV version can handle
    cout << "Source detected: image file." << endl;

    ims = imread(source, CV_LOAD_IMAGE_COLOR);
    src_opened = (! ims.empty() );
    cout << ( src_opened ? "Image successfully loaded from: " : "Failed to load image file from: ") << src << endl;
  }
  else {
    cout << "Source detected: camera." << endl;

    int camindex = atoi(source);
    VideoCapture videocap;
    if (camindex < 0) {
      cerr << "Camera index given is invalid: " << source << ". Positive integer expected." << endl;
      exit(EXIT_FAILURE);
    }
    src_opened = openCam(videocap, camindex);
    cout << ( src_opened ? "Camera connection successfully opened at index " : "Failed to connect to camera! Final index: ") << camindex << endl;
    
    if (src_opened) {
      videocap >> ims;
      src_opened = (! ims.empty() );

      if (src_opened) {
        // Show the user the captured image and ask if it should be saved
        imshow("Captured image", ims);
        cout << "Image successfully retrieved from camera." << endl << "Do you want to save it? Y/N" << endl;
        bool answered = false;
        char key;

        while(! answered) {
          key = (char) waitKey(30);
          if ( (key == 'y') || (key == 'Y') ) {
            string imd = "calib-cap.png";
            bool saved = imwrite(imd, ims);
            cout << (saved ? "Image successfully saved at: " : "Failed to save image at: ") << imd << endl;
            answered = true;
          }
          if ( (key == 'n') || (key == 'N') )
            answered = true;
        }
      }
      else
        cout << "Failed to retrieve image from camera!" << endl;
    }
    // VideoCapture object should be automatically destroyed after the next brace
  }

  return src_opened;
}



bool loadData(const char* refname, const char* scnname, const char* source, vector< Point2f >& refcorners, Size& boardsize, Size& scnsize, Mat& ims)
{
  bool ref_loaded = readRef(refname, refcorners, boardsize); // Load corners' positions in the destination plane
  cout << ( ref_loaded ? "Chessboard data successfully loaded from: " : "Failed to load chessboard data from: ") << refname << endl;

  bool scn_loaded = readScene(scnname, scnsize); // Load scene's dimensions
  cout << ( scn_loaded ? "Scene data successfully loaded from: " : "Failed to load scene data from: ") << scnname << endl;

  bool src_loaded = getCap(source, ims); // Load calibration image

  return (ref_loaded && scn_loaded && src_loaded);
}



bool saveCalibData(Mat M, char* filename)
{
  FileStorage fs(filename, FileStorage::WRITE);
  if( !fs.isOpened() )
    return false;

  time_t t; // Create a timestamp
  time( &t );
  struct tm *t2 = localtime( &t );
  char buffer[1024];
  strftime( buffer, sizeof(buffer) - 1, "%a %B %d %G - %X", t2 );

  fs << "calib_time" << buffer; // Write the timestamp on the file
  fs << "transform_mat" << M; // Write the transformation matrix on the file

  fs.release();
  return true;
}



int calibrateChess(vector< Point2f >& refcorners, Size boardsize, Size scnsize, Mat ims, char* savename, bool extra_acc)
{
  vector< Point2f > imcorners;
  bool found = findChessboardCorners(ims, boardsize, imcorners); // Get corners' positions in the image plane
  cout << ( found ? "Chessboard corners found." : "Failed to find chessboard corners.") << endl;

  if (found) {
    if (extra_acc) { // Optionnaly try to improve accuracy
      Mat imgray;
      cvtColor(ims, imgray, CV_BGR2GRAY);
      cornerSubPix(imgray, imcorners, Size(11,11), Size(-1,-1), TermCriteria( CV_TERMCRIT_EPS+CV_TERMCRIT_ITER, 30, 0.1 ));
    }

    //* # CORNER CHECK # Display the detected corners on the calibration image
    Mat imchess = ims.clone();
    drawChessboardCorners(imchess, boardsize, Mat(imcorners), found ); // Draw detected corners on the calibration image
    imshow("Found corners", imchess);
    cout << "Press any key to continue." << endl;
    waitKey();
    destroyWindow("Found corners");
    // # CORNER CHECK # */
    
    Mat M = findHomography(imcorners, refcorners), improj; // Get the transformation matrix projecting the corners from the image to the destination plane
    
    //* # WARP # Display the reprojected image
    warpPerspective(ims, improj, M, scnsize);  // Apply this transformation on the whole image
    drawChessboardCorners(improj, boardsize, Mat(refcorners), found ); // Draw expected corners on the projected image
    imshow("Reprojected image", improj);
    // # WARP #*/

    // Ask if the user is satisfied with the result
    bool correct = false, answered = false;
    char key;

    cout << endl << "Is the result correct? Y/N" << endl;
    while(! answered) {
      key = (char) waitKey(30);
      if ( (key == 'y') || (key == 'Y') ) {
        correct = true;
        answered = true;
      }
      if ( (key == 'n') || (key == 'N') )
        answered = true;
    }

    if (!correct) { // If the user is not satisfied
      cout << "Performing reprojection with corners in the reverse order..." << endl;
      reverse(imcorners.begin(), imcorners.end()); // Try with the points in the reversed order
      M = findHomography(imcorners, refcorners); // Get the new transformation matrix
    
      //* # WARP # Display the new reprojected image
      warpPerspective(ims, improj, M, scnsize);  // Apply the new transformation on the whole image
      drawChessboardCorners(improj, boardsize, Mat(refcorners), found ); // Draw expected corners on the projected image
      imshow("Reprojected image", improj);
      // # WARP #*/

      // Ask if the user is satisfied with the new result
      correct = false;
      answered = false;
      
      cout << endl << "Is the result correct? Y/N" << endl;
      while(! answered) {
        key = (char) waitKey(30);
        if ( (key == 'y') || (key == 'Y') ) {
          correct = true;
          answered = true;
        }
        if ( (key == 'n') || (key == 'N') )
          answered = true;
      }
    }
    
    if (correct) {
      bool saved = saveCalibData(M, savename);
      cout << (saved ? "Transformation matrix successfully saved at: " : "Failed to save transformation matrix at: ") << savename << endl;
      return EXIT_SUCCESS;
    }
    else {
      cout << "Could not calibrate successfully. Try improving the image resolution or placing the chessboard elsewhere." << endl;
      return EXIT_FAILURE;
    }  
  }
}



#define param 4
#define bound "# -----------------------------------"
#define acc true // This should become an optional parameter

int main(int args, char* argv[])
{
  if (args != param + 1) {
    if (args < param + 1)
      cout << "Too few arguments!";
    else
      cout << "Too many arguments!";
    cerr << " Number given: " << args - 1 << endl << "Usage: chess-calib <chess-data.yml> <scn-data.yml> <source> <calib-data.yml>" << endl;
    exit(EXIT_FAILURE);
  }
  
  else {
    cout << bound << endl << "Camera calibration with chessboard" << endl << endl;
    Mat ims;
    vector< Point2f > refcorners;
    Size boardsize, scnsize;

    if ( loadData(argv[1], argv[2], argv[3], refcorners, boardsize, scnsize, ims) ) {
      cout << bound << endl << endl;
      return calibrateChess(refcorners, boardsize, scnsize, ims, argv[4], acc);
    }
    else {
      cerr << bound << endl << "Aborting scanning..." << endl;
      exit(EXIT_FAILURE); // In case of error during initialization phase, the program exits
    }
  }
}