
/*
    Copyright 2012 Andrew Perrault and Saurav Kumar.

    This file is part of DetectText.

    DetectText is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    DetectText is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with DetectText.  If not, see <http://www.gnu.org/licenses/>.
*/
#include <cassert>
#include <fstream>
#include "TextDetection.h"
#include <opencv/highgui.h>
#include <exception>

void convertToFloatImage ( IplImage * byteImage, IplImage * floatImage )
{
  cvConvertScale ( byteImage, floatImage, 1 / 255., 0 );
}

class FeatureError : public std::exception
{
std::string message;
public:
FeatureError ( const std::string & msg, const std::string & file )
{
  std::stringstream ss;

  ss << msg << " " << file;
  message = msg.c_str ();
}
~FeatureError () throw ( )
{
}
};

IplImage * loadByteImage ( const char * name )
{
  IplImage * image = cvLoadImage ( name );

  if ( !image )
  {
    return 0;
  }
  cvCvtColor ( image, image, CV_BGR2RGB );
  return image;
}

IplImage * loadFloatImage ( const char * name )
{
  IplImage * image = cvLoadImage ( name );

  if ( !image )
  {
    return 0;
  }
  cvCvtColor ( image, image, CV_BGR2RGB );
  IplImage * floatingImage = cvCreateImage ( cvGetSize ( image ),
                                             IPL_DEPTH_32F, 3 );
  cvConvertScale ( image, floatingImage, 1 / 255., 0 );
  cvReleaseImage ( &image );
  return floatingImage;
}

int mainTextDetection ( int argc, char * * argv )
{

  DetectionParams params = detection_default_params;

  char* input_filename = argv[1];
  char* output_filename = argv[2];
  if (atoi(argv[3]) > -1)
    params.dark_on_light = atoi(argv[3]);
  if (atoi(argv[4]) > -1)
    params.canny_size = atoi(argv[4]);
  if (atof(argv[5]) > -1)
    params.canny_low = atof(argv[5]);
  if (atof(argv[6]) > -1)
    params.canny_high = atof(argv[6]);
  if (atoi(argv[7]) > -1)
    params.save_intermediate = atoi(argv[7]);


  IplImage * byteQueryImage = loadByteImage ( input_filename );
  if ( !byteQueryImage )
  {
    printf ( "couldn't load query image\n" );
    return -1;
  }

  // Detect text in the image
  IplImage * output = textDetection ( byteQueryImage, params );
  cvReleaseImage ( &byteQueryImage );

  // save output
  cvSaveImage ( output_filename, output );
  cvReleaseImage ( &output );
  return 0;
}

int main ( int argc, char * * argv )
{
  if ( ( argc != 8 ) )
  {
    printf ( "usage: %s imagefile resultImage dark_on_light canny_size canny_low canny_high save_intermediate\n",
             argv[0] );

    return -1;
  }
  return mainTextDetection ( argc, argv );
}
