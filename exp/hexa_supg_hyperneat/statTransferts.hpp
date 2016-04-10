#ifndef STAT_TRANSFERTS_HPP_
#define STAT_TRANSFERTS_HPP_


namespace bayesian_optimization
{
  namespace stat
  {

    template<typename Params>
    struct StatTransferts : public Stat<Params>
    {
      std::ofstream _ofs;
      StatTransferts(){

      }

      template<typename BO>
      void operator()(const BO& bo)
      {
	this->_create_log_file(bo, "transferts.dat");	
	if (!bo.dump_enabled())
	  return;

	(*this->_log_file)<< bo.iteration()<<" : ";
	std::vector<float> sample;
	for(int i=0;i<bo.samples()[0].size();i++)
	  {
	    (*this->_log_file)<<bo.samples()[bo.samples().size()-1][i]<<" ";
	    sample.push_back(bo.samples()[bo.samples().size()-1][i]);
	  }

	(*this->_log_file)<<" : "<<Params::archiveparams::archive[sample].fit<<" : ";


	Eigen::VectorXf v=bo.samples()[bo.samples().size()-1];
	if(bo.samples().size()==1)
	  {
	    (*this->_log_file)<<bo.meanfunction()(v)+Params::ucb::alpha*sqrt(bo.kernelfunction()(v,v))<<" : ";
	  }
	else
	  {
	    

	    // compute k, WITHOUT the last sample
	    Eigen::VectorXf k(bo.samples().size()-1);
	    for(int i=0;i<k.size();i++)
	      {
		k[i]=bo.kernelfunction()(bo.samples()[i],v);
	      }


	    Eigen::VectorXf mean_vector(bo.samples().size()-1);
	    for(int i=0;i<mean_vector.size();i++)
	      mean_vector[i]=bo.meanfunction()(bo.samples()[i]);

        //Eigen::MatrixXf inverted_kernel=bo.kernel().corner(Eigen::TopLeft, bo.kernel().rows()-1,bo.kernel().cols()-1).inverse();
        Eigen::MatrixXf inverted_kernel=bo.kernel().topLeftCorner(bo.kernel().rows()-1,bo.kernel().cols()-1).inverse();

        //Eigen::VectorXf observations=bo.observations().start(bo.observations().size()-1);
        Eigen::VectorXf observations=bo.observations().head(bo.observations().size()-1);

	    float mu=bo.meanfunction()(v)+ (k.transpose()*inverted_kernel*(observations-mean_vector))[0];
	    float sigma=bo.kernelfunction()(v,v) - (k.transpose()*inverted_kernel*k)[0];
	    (*this->_log_file)<<mu<<" : "<<mu+Params::ucb::alpha*sqrt(sigma)<<" : ";

	  }
	(*this->_log_file)<<bo.observations()[bo.observations().size()-1]<<std::endl;


    /*for(int i=0;i<Params::archiveparams::archive[sample].controller.size();i++)
      (*this->_log_file)<<Params::archiveparams::archive[sample].controller[i]<<" ";*/
    (*this->_log_file)<< "     " << Params::archiveparams::archive[sample].controller_index; //index of the controller in the archive

	(*this->_log_file)<<std::endl;
	(*this->_log_file)<<std::endl;

      }
    };
  }
}
#endif
