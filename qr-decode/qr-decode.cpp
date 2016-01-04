
#include <iostream>
#include <stdlib.h>
#include <string>
using namespace std;

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/imgproc/imgproc.hpp"
using namespace cv;

#include <zbar.h>
using namespace zbar;



/*
  ItemData
  Structure aggregating geometric and ID data about each detected item in the image
    X: horizontal coordinate of the item
    Y: vertical coordinate of the item
    theta: orientation of the item
    ID: number identifying the item, as a positive integer
*/
struct ItemData {
  int X;
  int Y;
  float theta;
  unsigned int ID;
} ;



/*
  readProj
  Function importing reprojection data as a transformation matrix from a YML file
    filename: input
      Full path and name to the YML file to read
    M: output
      OpenCV matrix to return transformation matrix
*/
static bool readProj( const char* filename, Mat& M)
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
    w: output
      Width of the scene
    h: output
      Height of the scene
*/
static bool readScene( const char* filename, int& w, int& h)
{
  FileStorage fs(filename, FileStorage::READ);
  if ( !fs.isOpened() )
    return false;
  
  FileNode wn = fs["Width"], hn = fs["Height"];
  if ( wn.empty() || hn.empty() )
    return false;

  // Get the scene's dimensions
  wn >> w;
  hn >> h;
  return true;
}



/*
  process
  Function scanning an image taken from a calibrated camera to identify QR or bar codes
    imsname: input
      Full path and name to the image to read
      The image should be taken with a fixed camera already calibrated
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
*/
int process(const char* projname, const char* scnname, char* imsname)
{
  // # Configuration phase #

  // Load transformation matrix and scene data from reference files
  Mat M;
  int width, height;
  bool proj_loaded = readProj(projname, M);
  cout << ( proj_loaded ? "Reprojection data successfully loaded from: " : "Failed to load reprojection data from: ") << projname << endl;
  bool scn_loaded = readScene(scnname, width, height);
  cout << ( scn_loaded ? "Scene data successfully loaded from: " : "Failed to load scene data from: ") << scnname << endl;

  // Open the source image
  Mat ims = imread(imsname);

  // In case of error during configuration phase, the program exits
  if (! (ims.data && proj_loaded && scn_loaded) ) {
    cerr << "Aborting scanning..." << endl;
    exit(EXIT_FAILURE);
  }
  else {
    Mat gray; // Grayscale reprojected image

    ImageScanner scanner; // Code scanner
    scanner.set_config(ZBAR_NONE, ZBAR_CFG_ENABLE, 1);

    //* # SHOW # Display reprojected image in a window
    namedWindow("Reprojected image", 1);
    // # SHOW # */

    warpPerspective(ims, ims, M, Size(width, height)); // Apply the transformation on the whole image
    cvtColor(ims, gray, CV_BGR2GRAY); // Get grayscale image for scanning phase

    //* # SHOW #
    imshow("Reprojected image", ims);
    waitKey();
    // # SHOW # */

    uchar *raw = (uchar*) gray.data; // Raw image data
    Image image(width, height, "Y800", raw, width * height);
    
    // Scan for codes in the image
    int nsyms = scanner.scan(image);

    // Extract results

    //* # DATA # Write symbols' data in the console
    if (nsyms >= 0) {
      cout << nsyms << " symbol(s) found in the given image" << endl;
    }
    else {
      cerr << "An error occured when scanning the image!" << endl;
      exit(EXIT_FAILURE);
    } 
    // # DATA # */

    //* # HIGHLIGHT # Delimit detected symbols in the reprojected image
    Scalar color(0, 0, 255); // BGR pure red to highlight detected symbols
    namedWindow("Found symbols", 1);
    // # HIGHLIGHT # */
    
    for(Image::SymbolIterator symbol = image.symbol_begin(); symbol != image.symbol_end(); ++symbol) {
      vector<Point> vp; // Build a vector of points delimiting the symbol
      int n = symbol->get_location_size();
      for(int i = 0; i < n; i++) {
        vp.push_back(Point(symbol->get_location_x(i),symbol->get_location_y(i)));
      }

      RotatedRect rect = minAreaRect(vp); // Find the smallest rectangle containing the symbol
      Point2f center = rect.center;
      float angle = rect.angle;
      
      //* # DATA #
      cout << "Data: \"" << symbol->get_data() << "\" - Angle: " << angle << " - Center: " << center << endl;
      // # DATA # */

      //* # HIGHLIGHT #
      Point2f pts[4]; // Draw the rectangle's edges on the reprojected image
      rect.points(pts);
      for(int i = 0; i < 4; i++)
        line(ims, pts[i], pts[(i + 1) % 4], color, 2);
      // # HIGHLIGHT # */
    }

    //* # HIGHLIGHT #
    imshow("Found symbols", ims);
    waitKey();
    // # HIGHLIGHT # */

    return EXIT_SUCCESS;
  }
}



#define param 3
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
    cerr << " Number given: " << args - 1 << endl << "Usage: qr-decode <calib-data.yml> <scn-data.yml> <imsname>" << endl;
    exit(EXIT_FAILURE);
  }
  else {
    cout << bound << endl << "QR decoder based on reprojection data" << endl << bound << endl << endl;
    return process(argv[1], argv[2], argv[3]);
  }
}
