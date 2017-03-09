#ifndef _CAMERA_HPP_
#define _CAMERA_HPP_

#include <OpenNI.h>
#include <OniCTypes.h>
#include <NiTE.h>
#include <NiteCTypes.h>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

class Camera
{
public:
  Camera();
  ~Camera();
  std::vector< cv::Rect > getFrame( cv::Mat3b & rgb, cv::Mat1w & depth );
private:
  openni::Device m_device;
  openni::VideoStream m_colour, m_depth;
  nite::UserTracker m_userTracker;
};

#endif
