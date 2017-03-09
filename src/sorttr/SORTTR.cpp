#include "SORTTR.hpp"

float overlapRatio( const cv::Rect & a, const cv::Rect & b )
{
  const float intersectionArea = ( a & b ).area();
  const float unionArea        = ( a.area() + b.area() ) - intersectionArea;

  return intersectionArea / unionArea;
}

std::vector< Assignment > max_cost_assignment( const dlib::matrix< int > & cost )
{
  const std::vector< long > assignment = dlib::max_cost_assignment( cost );

  std::vector< Assignment > result( assignment.size() );

  for( std::size_t a_index = 0; a_index < assignment.size(); ++a_index )
  {
    result.push_back(
      {
        static_cast< int >( a_index ),
        static_cast< int >( assignment[ a_index ] ),
        cost( a_index, assignment[ a_index ] )
      }
    );
  }

  return result;
}
