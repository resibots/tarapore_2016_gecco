#ifndef MEAN_ARCHIVE_HPP_
#define MEAN_ARCHIVE_HPP_
namespace bayesian_optimization
{
  namespace mean_functions
  {
    template<typename Params>
    struct MeanArchive_Map
    {
      MeanArchive_Map()
      {}
      float operator()(const Eigen::VectorXf& v)const
      {
	std::vector<float> key(v.size(),0);
	for(int i=0;i<v.size();i++)
	  key[i]=v[i];
	return  Params::archiveparams::archive.at(key).fit;
      }
    };
  }
}

#endif
