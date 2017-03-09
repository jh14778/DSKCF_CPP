#include "dskcf_tracker.hpp"

#include "GaussianKernel.hpp"
#include "HOGFeatureExtractor.hpp"
#include "ConcatenateFeatureChannelProcessor.h"

typedef cv::Rect_< double > Rect;

DskcfTracker::DskcfTracker()
{
	std::shared_ptr< Kernel > kernel = std::make_shared< GaussianKernel >();
	std::shared_ptr< FeatureExtractor > features = std::make_shared< HOGFeatureExtractor >();
	std::shared_ptr< FeatureChannelProcessor > processor = std::make_shared< ConcatenateFeatureChannelProcessor >();

	this->m_occlusionHandler = std::make_shared< OcclusionHandler >(KcfParameters(), kernel, features, processor);
}

DskcfTracker::~DskcfTracker()
{
}

float DskcfTracker::detect( const std::array< cv::Mat, 2 > & frame, cv::Rect_< double > & boundingBox )
{
	Point position = centerPoint( boundingBox );

	return this->m_occlusionHandler->score( frame, position );
}

bool DskcfTracker::update(const std::array< cv::Mat, 2 > & frame, Rect & boundingBox)
{
	Point position = centerPoint( boundingBox );

	if( auto bb = this->m_occlusionHandler->detect( frame, position ) )
	{
		boundingBox = *bb;
		position = centerPoint( boundingBox );
		this->m_occlusionHandler->update( frame, position );

		return static_cast< bool >( bb );
	}
	else
	{
		return static_cast< bool >( bb );
	}

	//return !this->m_occlusionHandler->isOccluded();
	//return true;
}

bool DskcfTracker::reinit(const std::array< cv::Mat, 2 > & frame, Rect & boundingBox)
{
	std::shared_ptr< Kernel > kernel = std::make_shared< GaussianKernel >();
	std::shared_ptr< FeatureExtractor > features = std::make_shared< HOGFeatureExtractor >();
	std::shared_ptr< FeatureChannelProcessor > processor = std::make_shared< ConcatenateFeatureChannelProcessor >();

	this->m_occlusionHandler = std::make_shared< OcclusionHandler >(KcfParameters(), kernel, features, processor);

	this->m_occlusionHandler->init(frame, boundingBox);

	return true;
}

TrackerDebug* DskcfTracker::getTrackerDebug()
{
	return nullptr;
}

const std::string DskcfTracker::getId()
{
	return "DSKCF";
}
