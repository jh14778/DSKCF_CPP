#include <iostream>
#include <memory>

#include "SORTTR.hpp"
#include "../cf_libs/dskcf/dskcf_tracker.hpp"

struct Tracker
{
  boost::optional< cv::Rect > update( const cv::Mat3b & rgb, const cv::Mat1i & depth, const cv::Rect & rect )
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

  float detect( const cv::Mat3b & rgb, const cv::Mat1i & depth, const cv::Rect & rect )
  {
    cv::Rect_< double > r = { rect.x, rect.y, rect.width, rect.height };
    return this->m_tracker->detect( std::array< cv::Mat, 2 >{ rgb, depth }, r );
  }

  std::shared_ptr< DskcfTracker > m_tracker;
};

int main()
{
  //Generic detector
  auto detector = []( const cv::Mat3b & rgb, const cv::Mat1i & depth ) -> std::vector< cv::Rect >
  {
    //TODO: insert psuedo-detector here...
    return {};
  };

  //Generic tracker factory
  auto factory = []( const cv::Mat3b & rgb, const cv::Mat1i & depth, const cv::Rect & rect ) -> Tracker
  {
    Tracker result;

    result.m_tracker = std::make_shared< DskcfTracker >();
    cv::Rect_< double > r = { rect.x, rect.y, rect.width, rect.height };
    result.m_tracker->reinit( std::array< cv::Mat, 2 >{ rgb, depth }, r );

    return result;
  };

  SORTTR<decltype(detector),decltype(factory),Tracker> sorttr( detector, factory );

  cv::Mat3b rgb;
  cv::Mat1i d;
  sorttr.update( rgb, d );

  return 0;
}
