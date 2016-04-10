#ifndef STAT_OBSERVATIONS_HPP_
#define STAT_OBSERVATIONS_HPP_


namespace bayesian_optimization
{
  namespace stat
  {

    template<typename Params>
    struct StatObservations : public Stat<Params>
    {
      std::ofstream _ofs;
      StatObservations(){

      }

      template<typename BO>
      void operator()(const BO& bo)
      {
	this->_create_log_file(bo, "observations.dat");	
	if (!bo.dump_enabled())
	  return;

	(*this->_log_file)<< bo.iteration()<<"  ";
	(*this->_log_file)<<bo.observations()[bo.observations().size()-1]<<" "<< bo.observations().maxCoeff()<<std::endl;


      }
    };
  }
}
#endif
