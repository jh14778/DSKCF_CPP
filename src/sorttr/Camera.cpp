#include "Camera.hpp"

#include <cassert>
#include <vector>

#include <opencv2/opencv.hpp>

Camera::Camera()
{
  openni::Array< openni::DeviceInfo > devices;
  openni::OpenNI::enumerateDevices( &devices );

  assert( devices.getSize() > 0 );

  openni::Status status = this->m_device.open( devices[ 0 ].getUri() );

  assert( status == openni::STATUS_OK );

  this->m_device.setDepthColorSyncEnabled( true );

  assert( this->m_device.isValid() );

  openni::VideoMode depthMode, colourMode;
  depthMode.setFps( 30 );
  colourMode.setFps( 30 );
  depthMode.setPixelFormat( openni::PIXEL_FORMAT_DEPTH_1_MM );
  colourMode.setPixelFormat( openni::PIXEL_FORMAT_RGB888 );
  depthMode.setResolution( 640, 480 );
  colourMode.setResolution( 640, 480 );

  this->m_depth.create( this->m_device, openni::SENSOR_DEPTH );
  this->m_colour.create( this->m_device, openni::SENSOR_COLOR );
  this->m_depth.setVideoMode( depthMode );
  this->m_colour.setVideoMode( colourMode );

  this->m_depth.setMirroringEnabled( true );
  this->m_colour.setMirroringEnabled( true );

  this->m_device.setImageRegistrationMode( openni::IMAGE_REGISTRATION_DEPTH_TO_COLOR );

  openni::Status depthStatus = this->m_depth.start();
  openni::Status colourStatus = this->m_colour.start();
  nite::Status niteStatus = this->m_userTracker.create( &this->m_device );

  assert( depthStatus == openni::STATUS_OK && colourStatus == openni::STATUS_OK && niteStatus == nite::STATUS_OK );
}

Camera::~Camera()
{
  this->m_userTracker.destroy();
  this->m_colour.destroy();
  this->m_depth.destroy();
  this->m_device.close();
}

std::vector< cv::Rect > Camera::getFrame( cv::Mat3b & rgb, cv::Mat1w & depth )
{
  //assert( rgb.cols == 640 && rgb.rows == 480 );
  //assert( depth.cols == 640 && depth.rows == 480 );

  std::vector< cv::Rect > result;
  openni::VideoFrameRef colourFrame, depthFrame;
  nite::UserTrackerFrameRef userFrame;
  openni::VideoStream * streams[ 2 ] = { &this->m_colour, &this->m_depth };
  int which = 0;

  if( openni::OpenNI::waitForAnyStream( streams, 2, &which, 5000 ) == openni::STATUS_OK )
  {
    this->m_colour.readFrame( &colourFrame );
    this->m_depth.readFrame( &depthFrame );
    this->m_userTracker.readFrame( &userFrame );

    memcpy( rgb.data, colourFrame.getData(), (size_t)colourFrame.getDataSize() );
    memcpy( depth.data, depthFrame.getData(), (size_t)depthFrame.getDataSize() );

    const nite::Array< nite::UserData > & users = userFrame.getUsers();

    for( int i = 0; i < users.getSize(); ++i )
    {
      if( users[ i ].isVisible() && !users[ i ].isNew() )
      {
        const nite::BoundingBox & bb = users[ i ].getBoundingBox();

        result.push_back(
          { bb.min.x, bb.min.y, bb.max.x - bb.min.x, bb.max.y - bb.min.y }
        );
      }
    }
  }

  colourFrame.release();
  depthFrame.release();
  userFrame.release();

  cv::cvtColor( rgb, rgb, cv::COLOR_BGR2RGB );

  auto newEnd = std::partition( result.begin(), result.end(),
    []( const cv::Rect & r ) -> bool
    {
      return ( r.area() < 320 * 240 ) &&
        ( r.area() > 160 * 120 ) &&
        ( rand() % 6 == 0 ) &&
        ( static_cast< float >( r.width ) / static_cast< float >( r.height ) > 0.3f ) &&
        ( static_cast< float >( r.width ) / static_cast< float >( r.height ) < 0.6f );
    }
  );

  result.erase( newEnd, result.end() );

  return result;
}
