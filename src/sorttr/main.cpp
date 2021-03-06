#include <iostream>
#include <memory>

#include "Camera.hpp"
#include "SORTTR.hpp"
#include "../cf_libs/dskcf/dskcf_tracker.hpp"

struct Tracker
{
  boost::optional< cv::Rect > update( const cv::Mat3b & rgb, const cv::Mat1w & depth, const cv::Rect & rect )
  {
    cv::Rect_< double > result = { rect.x, rect.y, rect.width, rect.height };
    if( this->m_tracker->update( std::array< cv::Mat, 2 >{ rgb, depth }, result ) )
    {
      return cv::Rect{
        result.x, result.y, result.width, result.height
      };
    }
    else
    {
      return {};
    }
  }

  float detect( const cv::Mat3b & rgb, const cv::Mat1w & depth, const cv::Rect & rect )
  {
    cv::Rect_< double > r = { rect.x, rect.y, rect.width, rect.height };
    return this->m_tracker->detect( std::array< cv::Mat, 2 >{ rgb, depth }, r );
  }

  std::shared_ptr< DskcfTracker > m_tracker;
};

cv::Rect toSmall( const cv::Rect input )
{
  return { input.x / 4, input.y / 4, input.width / 4, input.height / 4 };
}

cv::Rect toBig( const cv::Rect input )
{
  return { input.x * 4, input.y * 4, input.width * 4, input.height * 4 };
}

int main()
{
  openni::OpenNI::initialize();
  nite::NiTE::initialize();

  std::vector< cv::Rect > detections;

  //Generic detector
  auto detector = [&]( const cv::Mat3b & rgb, const cv::Mat1w & depth ) -> std::vector< cv::Rect >
  {
    if( detections.size() > 0 )
    {
      return { toSmall( detections[ 0 ] ) };
    }
    else
    {
      return {};
    }
  };

  //Generic tracker factory
  auto factory = []( const cv::Mat3b & rgb, const cv::Mat1w & depth, const cv::Rect & rect ) -> Tracker
  {
    Tracker result;

    result.m_tracker = std::make_shared< DskcfTracker >();
    cv::Rect_< double > r = { rect.x, rect.y, rect.width, rect.height };
    result.m_tracker->reinit( std::array< cv::Mat, 2 >{ rgb, depth }, r );

    return result;
  };

  std::unique_ptr< Camera > camera = std::make_unique< Camera >();
  SORTTR<decltype(detector),decltype(factory),Tracker> sorttr( detector, factory );

  cv::namedWindow( "SORT-TR", cv::WINDOW_NORMAL );
  cv::setWindowProperty( "SORT-TR", cv::WND_PROP_FULLSCREEN, cv::WINDOW_FULLSCREEN );
  cv::Mat3b rgb = cv::Mat3b::zeros( 480, 640 );
  cv::Mat1w d   = cv::Mat1w::zeros( 480, 640 );

  cv::Mat3b rgb_small;
  cv::Mat1w d_small;

  do
  {
    detections = camera->getFrame( rgb, d );

    cv::resize( rgb, rgb_small, {160,120} );
    cv::resize( d, d_small, {160,120} );

    auto tracks = sorttr.update( rgb_small, d_small );

    for( auto detection : detections )
    {
      cv::rectangle( rgb, detection, cv::Scalar{ 0, 255, 0 }, 4 );
    }

    for( auto track : tracks )
    {
      cv::rectangle( rgb, toBig( track.boundingBox ), cv::Scalar{ 0, 0, 255 }, 4 );
      cv::putText( rgb, std::to_string( track.identity ), toBig( track.boundingBox ).tl(), cv::FONT_HERSHEY_PLAIN, 5.0, { 0, 0, 255 }, 4.0 );
    }

    cv::imshow( "SORT-TR", rgb );
  } while( cv::waitKey( 15 ) != 27 );

  camera = nullptr;
  nite::NiTE::shutdown();
  openni::OpenNI::shutdown();

  return 0;
}
