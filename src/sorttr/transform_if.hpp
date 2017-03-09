#ifndef _TRANSFORM_IF_HPP_
#define _TRANSFORM_IF_HPP_

template< typename IIter, typename OIter, typename Map, typename Pred >
void transform_if( IIter begin, IIter end, OIter out, Pred pred, Map map )
{
  for( IIter itr = begin; itr != end; ++itr )
  {
    if( pred( *itr ) )
    {
      *out = map( *itr );
      ++out;
    }
  }
}

#endif
