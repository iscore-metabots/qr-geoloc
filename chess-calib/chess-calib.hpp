#pragma once

using namespace std;
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
    Returns if the import was successful
*/
bool readRef( const char* filename, vector< Point2f >& corners, Size& boardsize);



/*
  readScene
  Function importing scene reference data from a YML file
    filename: input
      Full path and name to the YML file to read
    scnsize: output
      Dimensions of the scene
    Returns if the import was successful
*/
bool readScene( const char* filename, Size& scnsize);



/*
  openCam
  Function attempting to connect to a camera up to index 9
    videocap: output
      VideoCapture object corresponding to the first found camera
    index: input output
      As input: first camera index to try
      As output: last camera index tried, index of the first found camera if applicable
    Returns if the program could connect to a camera
*/
bool openCam(VideoCapture& videocap, int& index);



/*
  getCap
  Function loading a calibration image from the given source
    source: input
      String indicating which source will be used : image file or camera
      source should be a full path to a JPG or PNG file
      or an integer corresponding to the index of the first camera to try to connect to
    ims: output
      Loaded or captured image which will be used for calibration
    Returns if the program could open the image file
*/
bool getCap( const char* source, Mat& ims);



/*
  loadData
  Function loading and checking all required data
    refname: input
      Full path and name to the YML file from which to get chessboard parameters
      About required YML structure, refer to example file
    scnname: input
      Full path and name to the YML file from which to get scene reference data
      About required YML structure, refer to example file
    source: input
      String indicating which source will be used : image file or camera
    refcorners: output
      Positions of the reference corners
    boardsize: output
      Loaded dimensions of the chessboard
    scnsize: output
      Loaded dimensions of the scene
    ims: output
      Image corresponding to the loaded calibration capture
    Returns if all the data loadings was successful
*/
bool loadData(const char* refname, const char* scnname, const char* source, vector< Point2f >& refcorners, Size& boardsize, Size& scnsize, Mat& ims);



/*
  saveCalibData
  Function writing an YML file to save the transformation matrix needed to warp the image taken from the camera into a plane map
    M: input
      Transformation matrix to export
    filename: input
      Full path and name to the YML file to write
    Returns if the export was successful
*/
bool saveCalibData(Mat M, char* filename);



/*
  calibrateChess
  Function processing an image containing a chessboard into a reprojected image to calibrate a camera
    refcorners: input
      Positions of the reference corners within the scene plane
    boardsize: input
      Dimensions of the "inner" chessboard to detect
    scnsize: input
      Dimensions of the scene
    ims: input
      Calibration image
      This image should be taken from a fixed camera
    savename:
      Full path and name to the calibration data to save as a YML file
    extra_acc: input
      Corner detection accuracy improvement option
*/
int calibrateChess(vector< Point2f >& refcorners, Size boardsize, Size scnsize, Mat ims, char* savename, bool extra_acc);