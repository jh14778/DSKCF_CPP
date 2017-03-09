#ifndef _SORTTR_HPP_
#define _SORTTR_HPP_

#include <algorithm>
#include <iterator>
#include <map>
#include <numeric>
#include <unordered_map>
#include <vector>

#include <boost/optional.hpp>
#include <dlib/optimization/max_cost_assignment.h>
#include <opencv2/core.hpp>

#include "transform_if.hpp"

struct Track
{
  int identity;
  cv::Rect boundingBox;
};

struct Assignment
{
  int detectionIndex;
  int trackIndex;
  int cost;
};

/**
 * Calculate overlapping ratio of two rectangles
 *
 * @param a The first rectangles
 * @param b The second rectangles
 *
 * @returns The ratio between the overlapping area and the total area of both
 *          bounding boxes.
 */
float overlapRatio( const cv::Rect & a, const cv::Rect & b );

/**
 * Calculate assignment which maximises the cost
 *
 * @param cost The matrix which represents the edge weights
 *
 * @returns A vector of assignments with their respective costs
 */
std::vector< Assignment > max_cost_assignment( const dlib::matrix< int > & cost );

/**
 * Simple Online Real-time Tracking with Target Reaquisiton
 */
template< typename DetectorType, typename TrackerFactoryType, typename TrackerType >
class SORTTR
{
public:
  SORTTR( DetectorType detector, TrackerFactoryType trackerFactory ) :
    m_nextID( 0 ), m_detector( detector ), m_trackerFactory( trackerFactory )
  {
  }

  //TODO: Refactor this function into stages as given in the paper, plus my contribution
  std::vector< Track > update( const cv::Mat3b & rgb, const cv::Mat1i & depth )
  {
    // Run the detector on the given frame
    std::vector< Track > result;
    const std::vector< cv::Rect > detections = this->m_detector( rgb, depth );
    std::unordered_map< int, std::pair< cv::Rect, TrackerType > > activeTrackers = this->m_activeTrackers;
    this->m_activeTrackers.clear();

    // Update the trackers on the given frame
    std::vector< std::pair< int, cv::Rect > > tracks;
    for( auto & pair : activeTrackers )
    {
      if( auto rect = pair.second.second.update( rgb, depth, pair.second.first ) )
      {
        tracks.push_back( { pair.first, *rect } );
      }
    }

    // Assign detections to tracks
    const std::size_t n = std::max< std::size_t >( tracks.size(), detections.size() );
    dlib::matrix< int > cost( n, n );
    for( std::size_t i = 0; i < detections.size(); ++i )
    {
      for( std::size_t j = 0; j < tracks.size(); ++j )
      {
        cost( i, j ) = std::floor(
          100.0f * overlapRatio( detections[ j ], tracks[ i ].second )
        );
      }
    }
    std::vector< Assignment > assignments = max_cost_assignment( cost );

    // Partition the assignments which are aboe the overlap threshold (0.3)
    auto threshold_part = std::partition( assignments.begin(), assignments.end(),
      []( const Assignment & assignment ){ return assignment.cost >= 30; }
    );

    // Add the trackers which were assigned to the active tracker map
    std::for_each( assignments.begin(), threshold_part,
      [&]( const Assignment & assignment )
      {
        const int trackID = tracks[ assignment.trackIndex ].first;
        this->m_activeTrackers[ trackID ] = activeTrackers[ trackID ];
      }
    );

    // Partition out the unassigned tracks and detections
    std::vector< cv::Rect > unassignedDetections;
    std::vector< std::pair< int, cv::Rect > > unassignedTracks;
    unassignedTracks.reserve( std::distance( threshold_part, assignments.end() ) );
    unassignedDetections.reserve( std::distance( threshold_part, assignments.end() ) );

    transform_if( threshold_part, assignments.end(),
      std::insert_iterator< std::vector< std::pair< int, cv::Rect > > >(
        unassignedTracks, unassignedTracks.begin()
      ),
      [&]( const Assignment & assignment ){ return assignment.trackIndex < static_cast< int >( tracks.size() ); },
      [&]( const Assignment & assignment ){ return tracks[ assignment.trackIndex ]; }
    );

    transform_if( threshold_part, assignments.end(),
      std::insert_iterator< std::vector< cv::Rect > >(
        unassignedDetections, unassignedDetections.begin()
      ),
      [&]( const Assignment & assignment ){ return assignment.detectionIndex < static_cast< int >( detections.size() ); },
      [&]( const Assignment & assignment ){ return detections[ assignment.detectionIndex ]; }
    );

    // Suspend unassigned tracks
    for( auto & track : unassignedTracks )
    {
      this->m_suspendedTrackers[ track.first ] = activeTrackers[ track.first ].second;
    }

    // Evaluate suspeded tracks against unassigned detections
    const std::size_t m = std::max< std::size_t >( this->m_suspendedTrackers.size(), unassignedDetections.size() );
    dlib::matrix< int > tr_cost( m, m );
    for( std::size_t detectionIndex = 0; detectionIndex < unassignedDetections.size(); ++detectionIndex )
    {
      for( auto itr = this->m_suspendedTrackers.begin(); itr != this->m_suspendedTrackers.end(); ++itr )
      {
        const auto trackerIndex = std::distance( this->m_suspendedTrackers.begin(), itr );

        tr_cost( detectionIndex, trackerIndex ) = std::floor(
          100.0f * itr->second.detect(
            rgb, depth, unassignedDetections[ detectionIndex ]
          )
        );
      }
    }
    std::vector< Assignment > tr_assignments = max_cost_assignment( tr_cost );

    // Partition assignments where the response is greater than 0.2
    auto tr_threshold = std::partition( tr_assignments.begin(), tr_assignments.end(),
      []( const Assignment & assignment ) -> bool
      {
        return assignment.cost > 20;
      }
    );

    // Re-activate suspended and assigned tracks
    std::for_each( tr_assignments.begin(), tr_threshold,
      [&]( const Assignment & assignment )
      {
        auto itr = this->m_suspendedTrackers.begin();
        std::advance( itr, assignment.trackIndex );
        this->m_activeTrackers[ itr->first ] = {
          unassignedDetections[ assignment.detectionIndex ], itr->second
        };

        if( auto rect = itr->second.update( rgb, depth, unassignedDetections[ assignment.detectionIndex ] ) )
        {
          result.push_back( { itr->first, *rect } );
        }
      }
    );

    // Initialise tracks on unassigned detections
    std::for_each( tr_threshold, tr_assignments.end(),
      [&]( const Assignment & assignment )
      {
        if( assignment.detectionIndex < static_cast< int >( unassignedDetections.size() ) )
        {
          const int id = ++this->m_nextID;
          this->m_activeTrackers[ id ] = {
            unassignedDetections[ assignment.detectionIndex ],
            this->m_trackerFactory(
              rgb, depth, unassignedDetections[ assignment.detectionIndex ]
            )
          };

          result.push_back(
            { id, unassignedDetections[ assignment.detectionIndex ] }
          );
        }
      }
    );

    // Remove active tracks from suspended tracks
    for( auto & tracks : this->m_activeTrackers )
    {
      this->m_suspendedTrackers.erase( tracks.first );
    }

    return result;
  }
private:
  int m_nextID;
  DetectorType m_detector;
  TrackerFactoryType m_trackerFactory;
  std::unordered_map< int, std::pair< cv::Rect, TrackerType > > m_activeTrackers;
  std::unordered_map< int, TrackerType > m_suspendedTrackers;
};

#endif
