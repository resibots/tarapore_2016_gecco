#ifndef IDW_MODIFIER_HPP
#define IDW_MODIFIER_HPP

#include "idw.hpp"

int generation = -1;



namespace sferes
{

    namespace modifier_div
    {
        template<typename Phen,typename Center >
        struct _parallel_disparity
        {
            typedef std::vector<boost::shared_ptr<Phen> > pop_t;
            pop_t _pop;
            std::vector< Center  >& _centers;
            IDW _model;

            _parallel_disparity(pop_t& pop,std::vector< Center  >& centers,IDW model) : _pop(pop), _centers(centers),_model(model){}
            _parallel_disparity(const _parallel_disparity& ev) : _pop(ev._pop), _centers(ev._centers),_model(ev._model) {}

            void operator() (const parallel::range_t& r) const
            {
                for (size_t i = r.begin(); i != r.end(); ++i)
                {
                    float output = 0.0f;
                    if (_pop[i]->fit().objs()[0] <= -10000)
                            output=-1000;
                        else
                        {
                    _model.predict( _centers, _pop[i]->fit().legs_features, output);
                        }
                    int l = _pop[i]->fit().objs().size()-2;
                    _pop[i]->fit().set_obj(l, output);
                }
            }
        };
    }



    template<typename Center, typename Params = stc::_Params, typename Exact = stc::Itself>
    class IdwModifier : public stc::Any<Exact>
    {
    private:
        std::vector< Center > centers;
        IDW model;
        int transfer_number;

    public:



      template<typename Ea>
      void apply(Ea& ea)
      {
	if (generation == -1)
	  transfer_number = 0;
	
	++generation;
	
	std::cout << "G" << generation << std::endl;
	
	/*if(transfer_number >= Params::surrogate::max_transfer_number)
	  {
	  std::cout << "Max. number of transfers reached." << std::endl;
	  return;
	  }*/
	if (generation % 25 == 0)
	  {
	    ++transfer_number;
	    //RANDOM TRANSFER
	    unsigned int transferred;
	    
	    do
	      {
#ifdef MAX_TRANSFER //select an individual with at least 50% of max. fitness (on 0th obj.)
		transferred = 0;
		double max = ea.pop()[0]->fit().objs()[0];
		for (int i=1;i < ea.pop().size();++i)
		  {
		    if (max < ea.pop()[i]->fit().objs()[0])
		      {
			transferred = i;
			max = ea.pop()[i]->fit().objs()[0];
		      }
		  }
		
		do
		  {
		    transferred = sferes::misc::rand<unsigned int>(ea.pop().size());
		  }
		while(ea.pop()[transferred]->fit().objs()[0]<0.5*max);
#else //random selection
		transferred = sferes::misc::rand<unsigned int>(ea.pop().size());
		
#endif
	
                }
                while (ea.pop()[transferred]->fit().objs()[0]<=-10000);
                Center center = *ea.pop()[transferred];
                center.fit().transfer(center,transfer_number);

                centers.push_back(center);



                //if addition
                model.learn(centers);

            }

            // parallel compute disparity
            parallel::init();
            parallel::p_for(parallel::range_t(0, ea.pop().size()),
                            modifier_div::_parallel_disparity<typename Ea::phen_t,Center>(ea.pop(),centers,model));

        }
    };
}
#endif
