#include <iostream>
#include <stdlib.h>
using namespace std;

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/imgproc/imgproc.hpp"
using namespace cv;


/*
  readRef
  Function importing chessboard parameters as geometric data from a YML file
    filename: input
      Full path and name to the YML file to read
    corners: output
      Array of points that will be filled with the corners' positions
      Row by row, then column by column, from top to bottom, from left to right
    boardsize: output
      Size of the "inner" chessboard that will be used as reference
*/
static bool readRef( const char* filename, vector< Point2f >& corners, Size& boardsize)
{
  corners.resize(0); // Empty the output vector

  FileStorage fs(filename, FileStorage::READ);
  if( !fs.isOpened() )
    return false;
  
  FileNode sizen = fs["Size"], orign = fs["Origin"], stepn = fs["Step"];
  if ( sizen.empty() || orign.empty() || stepn.empty() )
    return false;

  sizen >> boardsize; // Get the size of the inner chessboard
  Point2f orig;  // Get the board origin, i.e. top-left corner
  orign >> orig;
  double step; // Get the board step, i.e. square side length
  stepn >> step;
  
  double x0 = orig.x, y0 = orig.y;       // Fill the output array with the corners' positions
  for(int j = 0; j < boardsize.height; j++)  // Row by row
    for(int i = 0; i < boardsize.width; i++) // Then column by column
      corners.push_back(Point( x0 + i * step, y0 + j * step ) );

  return true;
}



/*
  readRefXML
  UNUSED FUNCTION
  Function importing chessboard parameters as geometric data from a XML file
  Substitute to readRef, reading an XML rather than a YML file
    filename: input
      Full path and name to the XML file to read
    corners: output
      Array of points that will be filled with the corners' positions
      Row by row, then column by column, from top to bottom, from left to right
    boardsize: output
      Size of the "inner" chessboard that will be used as reference
*/
static bool readRefXML( const string& filename, vector< Point2f >& corners, Size& boardsize)
{
  corners.resize(0); // Empty the output vector

  FileStorage fs(filename, FileStorage::READ);
  if( !fs.isOpened() )
    return false;
  
  FileNode sizen = fs["Size"]; // Get the size of the inner chessboard
  boardsize = Size( (int) sizen["Rows"], (int) sizen["Columns"] );

  FileNode orign = fs["Origin"]; // Get the board origin, i.e. top-left corner
  Point2f orig( (double) orign["X"], (double) orign["Y"] );

  double step = (double) fs["Step"]; // Get the board step, i.e. square side length
  
  double x0 = orig.x, y0 = orig.y;       // Fill the output array with the corners' positions
  for(int j = 0; j < boardsize.height; j++)  // Row by row
    for(int i = 0; i < boardsize.width; i++) // Then column by column
      corners.push_back(Point( x0 + i * step, y0 + j * step ) );

  return true;
}



/*
	saveCalibData
  Function writing an YML file to save the transformation matrix needed to warp the image
  taken from the camera into a plane map
    M: input
      Transformation matrix to export
    filename: input
      Full path and name to the YML file to write
*/
static bool saveCalibData(Mat M, char* filename)
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



/*
  process
  Function processing an image containing a chessboard into a reprojected image
    imsname: input
      Full path and name to the calibration image to read
      The calibration image should be taken with a fixed camera
    refname: input
      Full path and name to the YML file from which to get chessboard parameters
      About required YML structure, refer to example file
    savename: input
      Full path and name to the YML file onto which to export calibration data
    extra_acc: input
      Corner detection accuracy improvement option
*/
int process(const char* imsname, const char* refname, char* savename, bool extra_acc)
{
  Mat ims = imread(imsname, CV_LOAD_IMAGE_COLOR); // Load calibration image
  cout << ( (ims.data) ? "Image successfully loaded from: " : "Failed to load image from: ") << imsname << endl;
  
  vector< Point2f > refcorners;
  Size boardsize;
  bool loaded = readRef(refname, refcorners, boardsize); // Load corners' positions in the destination plane
  cout << (loaded ? "Chessboard parameters successfully loaded from: " : "Failed to load chessboard parameters from: ") << refname << endl;

  if (! (loaded && ims.data) ) {
    cerr << "Aborting calibration..." << endl;
    exit(EXIT_FAILURE);
  }
  else {
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
      waitKey();
      destroyWindow("Found corners");
      // # CORNER CHECK # */

      Mat M = findHomography(imcorners, refcorners), improj; // Get the transformation matrix projecting the corners from the image to the destination plane
      
      //* # WARP # Display the reprojected image
      warpPerspective(ims, improj, M, Size(1200, 1200));  // Apply this transformation on the whole image
      drawChessboardCorners(improj, boardsize, Mat(refcorners), found ); // Draw expected corners on the projected image
      imshow("Reprojected image", improj);
      // # WARP #*/

      // Ask if the user is satisfied with the result
      bool correct = false, answered = false;
      char key;

      cout << "Is the result correct? Y/N" << endl;
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
        cout << "Performing reprojection with the reverse order..." << endl;
        reverse(imcorners.begin(), imcorners.end()); // Try with the points in the reversed order
        M = findHomography(imcorners, refcorners); // Get the new transformation matrix
      
        //* # WARP # Display the new reprojected image
        warpPerspective(ims, improj, M, Size(1200, 1200));  // Apply the new transformation on the whole image
        drawChessboardCorners(improj, boardsize, Mat(refcorners), found ); // Draw expected corners on the projected image
        imshow("Reprojected image", improj);
        // # WARP #*/

        // Ask if the user is satisfied with the new result
        correct = false;
        answered = false;
        
        cout << "Is the result correct? Y/N" << endl;
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
}



#define param 3
#define bound "# -----------------------------------"
#define acc true

int main(int args, char* argv[])
{
  if (args != param + 1) {
    if (args < param + 1) {
      cout << "Too few arguments!";
    }
    else {
      cout << "Too many arguments!";
    }
    cerr << " Number given: " << args - 1 << endl << "Usage: chess-calib <ims-name> <chess-data.yml> <calib-data.yml>" << endl;
    return EXIT_FAILURE;
  }
  else {
    cout << bound << endl << "Camera calibration with chessboard" << endl << bound << endl << endl;
    return process(argv[1], argv[2], argv[3], acc);
  }
}
