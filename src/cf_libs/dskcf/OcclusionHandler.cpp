#include "OcclusionHandler.hpp"

#include <tbb/concurrent_vector.h>

OcclusionHandler::OcclusionHandler( KcfParameters paras, std::shared_ptr< Kernel > & kernel, std::shared_ptr< FeatureExtractor > & featureExtractor, std::shared_ptr< FeatureChannelProcessor > & featureProcessor )
{
  this->m_paras = paras;
  this->m_kernel = kernel;
  this->m_featureExtractor = featureExtractor;
  this->m_featureProcessor = featureProcessor;
  this->m_depthSegmenter = std::make_shared< DepthSegmenter >();
  this->m_scaleAnalyser = std::make_shared< ScaleAnalyser >( this->m_depthSegmenter.get(), paras.padding );

  for( int i = 0; i < 2; i++ )
  {
    this->m_targetTracker[ i ] = std::make_shared< DepthWeightKCFTracker >( paras, kernel );
  }

  //this->m_occluderTracker = std::make_shared< KcfTracker >( paras, kernel );

  this->m_lambdaOcc = 0.35;
  this->m_lambdaR1 = 0.4;
  this->m_lambdaR2 = 0.2;
  //this->m_isOccluded = false;

  this->singleFrameProTime = std::vector<int64>(8,0);
}

OcclusionHandler::~OcclusionHandler()
{
  this->m_depthSegmenter = nullptr;
}

void OcclusionHandler::init( const std::array< cv::Mat, 2 > & frame, const Rect & target )
{
  std::vector< std::shared_ptr< FC > > features( 2 );
  //this->m_isOccluded = false;
  this->m_initialSize = target.size();

  this->m_scaleAnalyser->clearObservers();
  this->m_scaleAnalyser->registerScaleChangeObserver( this );
  this->m_scaleAnalyser->registerScaleChangeObserver( this->m_targetTracker[ 0 ].get() );
  this->m_scaleAnalyser->registerScaleChangeObserver( this->m_targetTracker[ 1 ].get() );
  this->m_depthSegmenter->init( frame[ 1 ], target );
  this->m_scaleAnalyser->init( frame[ 1 ], target );

  Point position = centerPoint( target );
  Rect window = boundingBoxFromPointSize( position, this->m_windowSize );

  //Extract features
  for( int i = 0; i < 2; i++ )
  {
    features[ i ] = this->m_featureExtractor->getFeatures( frame[ i ], window );
    FC::mulFeatures( features[ i ], this->m_cosineWindow );
  }

  features = this->m_featureProcessor->concatenate( features );

  for( uint i = 0; i < features.size(); i++ )
  {
    this->m_targetTracker[ i ]->init( frame[ i ], features[ i ], position );
  }

  this->m_filter.initialise( position );
}

const boost::optional< Rect > OcclusionHandler::detect( const std::array< cv::Mat, 2 > & frame, const Point & position )
{
  return this->visibleDetect( frame, position );
}

void OcclusionHandler::update( const std::array< cv::Mat, 2 > & frame, const Point & position )
{
  return this->visibleUpdate( frame, position );
}

const float OcclusionHandler::score( const std::array< cv::Mat, 2 > & frame, const Point & position )
{
  std::vector< double > responses;
  std::vector< std::shared_ptr< FC > > features( 2 );
  std::vector< Point > positions;

  //Rect target = boundingBoxFromPointSize( position, this->m_targetSize );
  Rect window = boundingBoxFromPointSize( position, this->m_windowSize );

  tbb::parallel_for< uint >( 0, 2, 1,
	  [this,&frame,&features,&window]( uint index ) -> void
	  {
		  features[ index ] = this->m_featureExtractor->getFeatures( frame[ index ], window );
		  FC::mulFeatures( features[ index ], this->m_cosineWindow );
	  }
  );

  features = this->m_featureProcessor->concatenate( features );
  std::vector< cv::Mat > frames_ = this->m_featureProcessor->concatenate( std::vector< cv::Mat >( frame.begin(), frame.end() ) );

  for( uint i = 0; i < features.size(); i++ )
  {
    DetectResult result = this->m_targetTracker[ i ]->detect( frames_[ i ], features[ i ], position, this->m_depthSegmenter->getTargetDepth(), this->m_depthSegmenter->getTargetSTD() );
    positions.push_back( result.position );
    responses.push_back( result.maxResponse );
  }

  return responses[ 0 ];
}

const boost::optional< Rect > OcclusionHandler::visibleDetect( const std::array< cv::Mat, 2 > & frame, const Point & position )
{
  std::vector< double > responses;
  std::vector< std::shared_ptr< FC > > features( 2 );
  std::vector< Point > positions;

  Rect target = boundingBoxFromPointSize( position, this->m_targetSize );
  Rect window = boundingBoxFromPointSize( position, this->m_windowSize );

  tbb::parallel_for< uint >( 0, 2, 1,
	  [this,&frame,&features,&window]( uint index ) -> void
	  {
		  features[ index ] = this->m_featureExtractor->getFeatures( frame[ index ], window );
		  FC::mulFeatures( features[ index ], this->m_cosineWindow );
	  }
  );

  features = this->m_featureProcessor->concatenate( features );
  std::vector< cv::Mat > frames_ = this->m_featureProcessor->concatenate( std::vector< cv::Mat >( frame.begin(), frame.end() ) );

  for( uint i = 0; i < features.size(); i++ )
  {
    DetectResult result = this->m_targetTracker[ i ]->detect( frames_[ i ], features[ i ], position, this->m_depthSegmenter->getTargetDepth(), this->m_depthSegmenter->getTargetSTD() );
    positions.push_back( result.position );
    responses.push_back( result.maxResponse );
  }
  //here the maximun response is calculated....
  //TO BE CHECKED IN CASE OF MULTIPLE MODELS...LINEAR ETC....WORKS ONLY FOR SINGLE (or concatenate) features
  target = boundingBoxFromPointSize( positions.back(), this->m_targetSize );
  int bin=this->m_depthSegmenter->update( frame[ 1 ], target );

  DepthHistogram histogram = this->m_depthSegmenter->getHistogram();

  double totalArea=target.area()*1.05;

	//here the maximun response is calculated....
  Point estimate = this->m_featureProcessor->concatenate( positions );

	estimate.x=(estimate.x -this->m_targetSize.width/2)<	frame[ 0 ].cols	 ? estimate.x : this->m_targetSize.width;
	estimate.y=(estimate.y -this->m_targetSize.height/2)<	frame[ 0 ].rows	 ? estimate.y : this->m_targetSize.height;
	estimate.x=(estimate.x +this->m_targetSize.width/2)>	0	 ? estimate.x : 1;
	estimate.y=(estimate.y +this->m_targetSize.height/2)>	0	 ? estimate.y : 1;
  return boundingBoxFromPointSize( estimate, this->m_initialSize * this->m_scaleAnalyser->getScaleFactor() );
}

void OcclusionHandler::visibleUpdate( const std::array< cv::Mat, 2 > & frame, const Point & position )
{
	//EVALUATE CHANGE OF SCALE....
	int64 tStartScaleCheck=cv::getTickCount();
	std::vector< std::shared_ptr< FC > > features( 2 );
	Rect window = boundingBoxFromPointSize( position, this->m_windowSize );

	this->m_scaleAnalyser->update( frame[ 1 ], window );


	int64 tStopScaleCheck = cv::getTickCount();
	this->singleFrameProTime[5]=tStopScaleCheck-tStartScaleCheck;



	int64 tStartModelUpdate=tStopScaleCheck;
	window = boundingBoxFromPointSize( position, this->m_windowSize );

	tbb::parallel_for< uint >( 0, 2, 1,
		[this,&frame,&features,&window]( uint index ) -> void
	{
		features[ index ] = this->m_featureExtractor->getFeatures( frame[ index ], window );
		FC::mulFeatures( features[ index ], this->m_cosineWindow );
	}
	);

	features = this->m_featureProcessor->concatenate( features );

	for( size_t i = 0; i < features.size(); i++ )
	{
		this->m_targetTracker[ i ]->update( frame[ i ], features[ i ], position );
	}

	int64 tStopModelUpdate = cv::getTickCount();
	this->singleFrameProTime[6]=tStopModelUpdate-tStartModelUpdate;

}


void OcclusionHandler::onVisible( const std::array< cv::Mat, 2 > & frame, std::vector< std::shared_ptr< FC > > & features, const Point & position )
{
}

bool OcclusionHandler::evaluateOcclusion( const DepthHistogram & histogram, const int objectBin, const double maxResponse )
{
  // ( f(z)_max < λ_r1 ) ∧ ( Φ( Ω_obj ) > λ_occ l)
  return ( ( maxResponse < this->m_lambdaR1 ) && ( this->phi( histogram, objectBin ) > this->m_lambdaOcc ) );
}

bool OcclusionHandler::evaluateOcclusion( const DepthHistogram & histogram, const int objectBin, const double maxResponse,const double totalArea )
{
  // ( f(z)_max < λ_r1 ) ∧ ( Φ( Ω_obj ) > λ_occ l)
  return ( ( maxResponse < this->m_lambdaR1 ) && ( this->phi( histogram, objectBin,totalArea ) > this->m_lambdaOcc ) );
}

bool OcclusionHandler::evaluateVisibility( const DepthHistogram & histogram, const int objectBin, const double maxResponse ) const
{
  //( f(z)_n > λ_r2 ) ∧ ( Φ( Ω_Tbc ) < λ_occ )
  return ( ( maxResponse > this->m_lambdaR2 ) && ( this->phi( histogram, objectBin ) < this->m_lambdaOcc ) );
}

double OcclusionHandler::phi( const DepthHistogram & histogram, const int objectBin )const
{
  double totalArea = 0.0;
  double occluderArea = 0.0;

  for( uint i = 0; i < histogram.size(); i++ )
  {
    if( i < objectBin )
    {
      occluderArea += histogram[ i ];
    }

    totalArea += histogram[ i ];
  }

  return occluderArea / totalArea;
}


double OcclusionHandler::phi( const DepthHistogram & histogram, const int objectBin,const double totalArea ) const
{
  double occluderArea = 0.0;

  for( uint i = 0; i < histogram.size(); i++ )
  {
    if( i < objectBin )
    {
      occluderArea += histogram[ i ];
    }
	  else
		{
      break;
    }
  }

  return occluderArea / totalArea;
}

void OcclusionHandler::onScaleChange( const Size & targetSize, const Size & windowSize, const cv::Mat2d & yf, const cv::Mat1d & cosineWindow )
{
  this->m_targetSize = targetSize;
  this->m_windowSize = windowSize;
  this->m_cosineWindow = cosineWindow;
}
