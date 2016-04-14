#ifndef SVM_MODIFIER_HPP
#define SVM_MODIFIER_HPP

//#define PRESAMP
#include "svm_model.hpp"
#include "behavior.hpp"



#include <boost/fusion/container.hpp>
#include <boost/fusion/algorithm.hpp>
#include <boost/fusion/support/is_sequence.hpp>
#include <boost/fusion/include/is_sequence.hpp>
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
            SVM _model;

             _parallel_disparity(pop_t& pop,std::vector< Center  >& centers,SVM model) : _pop(pop), _centers(centers),_model(model){}
            _parallel_disparity(const _parallel_disparity& ev) : _pop(ev._pop), _centers(ev._centers),_model(ev._model) {}

            void operator() (const parallel::range_t& r) const
            {
                for (size_t i = r.begin(); i != r.end(); ++i)
                {
		  
                    float output = 0.0f;
                    if (_pop[i]->fit().dead())
                        output=-1000;
                    else
                    {
                        std::vector<float> params;

                        _model.predict(params, _centers, _pop[i]->fit().features, output);

                    }
		  
                    int l = _pop[i]->fit().objs().size()-1;
                    //_pop[i]->fit().set_obj(l, output);
		  
		    _pop[i]->fit().behavior().transferability=output;
		  
                }
            }
        };

      
       template<typename Center >
        struct _parallel_disparity_archive
        {
	  typedef        std::vector< boost::shared_ptr<Behavior> > archive_t;
	  archive_t _archive;
	  std::vector< Center  >& _centers;
	  SVM _model;
	  
	  _parallel_disparity_archive(archive_t& archive,std::vector< Center  >& centers,SVM model) : _archive(archive), _centers(centers),_model(model){}
	  _parallel_disparity_archive(const _parallel_disparity_archive& ev) : _archive(ev._archive), _centers(ev._centers),_model(ev._model) {}
	  
	  void operator() (const parallel::range_t& r) const
	  {
	    for (size_t i = r.begin(); i != r.end(); ++i)
	      {
		float output = 0.0f;
		if (_archive[i]->position[0] == 0 && _archive[i]->position[1] == 0 && _archive[i]->drift == 0 )
		  output=-1000;
		else
		  {
		    std::vector<float> params;
		    
		    _model.predict(params, _centers, _archive[i]->features, output);
		    
		  }

		_archive[i]->transferability= output;
	      }
	  }
       };

      template<typename V1, typename V2>
      float dist_centers(const V1& v1, const V2& v2)
      {

	assert(v1.size() == v2.size());
	typename V1::const_iterator it1 = v1.begin(), it2 = v2.begin();
	float res = 0.0f;
	while (it1 != v1.end())
	  {
	    float v = (float)*it1 - (float)*it2;
	    res += v * v;
	    ++it1;
	    ++it2;
	  }

	return sqrtf(res);
      }
      

      template<typename T,typename T2>
      struct compare_dist_centers
      {
	compare_dist_centers(const T2& v) : _v(v) {}
	const T2 _v;
	bool operator()(const T& v1, const T& v2) const
	{
	  assert(v1.fit().behavior().position.size());
	  assert(v1.fit().behavior().position.size() == _v.position.size());
	  assert(v2.fit().behavior().position.size() == _v.position.size());



	  return dist_centers(v1.fit().behavior().position, _v.position) < dist_centers(v2.fit().behavior().position, _v.position);
	}
      };

      
    }
    int generation = -1;

    template<typename Center, typename Params = stc::_Params, typename Exact = stc::Itself>
    class SvmModifier : public stc::Any<Exact>
    {
    private:
      std::vector<Center > centers;
      SVM model;
      int transfer_number;
        
      struct timeval timev_prev;  // Previous tick absolute time

    public:

      template<typename Ea>
        void apply(Ea& ea)
        {
            if (generation == -1)
                transfer_number = 0;

            ++generation;

            std::cout << "G" << generation << std::endl;

        
#ifdef PRESAMP
            if (generation  == 0)
            {

                for (int i=0;i<100;i++)
                {
                    ++transfer_number;

                    do
                    {

                        ea.pop()[0]->random();
                        ea.pop()[0]->develop();
                        ea.pop()[0]->fit().eval(*ea.pop()[0]);


                    }
                    while (ea.pop()[0]->fit().objs()[0]==-10000);


                    Center center = *ea.pop()[0];

                    center.fit().transfer(center);

                    centers.push_back(center);
                    //if addition
                    model.learn(centers);
                }

            }
#else

            if (generation % 50== 0 && generation < 3000)//ADD
	      {
		std::ofstream ofile("results.dat", std::ios_base::app);
		ofile << transfer_number << " ";
		
		{
		  struct timeval timev_cur; 
		  struct timeval timev_diff; 
		  gettimeofday(&timev_cur, NULL);
		  timersub(&timev_cur, &timev_prev, &timev_diff);
		  ofile<< timev_diff.tv_sec+((float)timev_diff.tv_usec)/1000000<<" "; 
		}
                ++transfer_number;
                //RANDOM TRANSFER
                unsigned int transferred;
		bool in_archive=false;
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
#else 
		    transferred = sferes::misc::rand<unsigned int>(ea.pop().size()+boost::fusion:: at_c<1>(ea.fit_modifier()).archive().size());
		    if(transferred>=ea.pop().size())
		      {
		      in_archive=true;
		      std::cout<<"in archive"<<std::endl;
		      }
//Max behavioral novelty between transfered  //random selection
		    //transferred = sferes::misc::rand<unsigned int>(ea.pop().size());
		    /*		    transferred=0;
		    float max=0;
		    if( centers.size()<2)
		      transferred = sferes::misc::rand<unsigned int>(ea.pop().size());
		    else
		      {
			std::cout<<1<<std::endl;
			for(int i=0;i< ea.pop().size();i++)
			  {
			    if(!ea.pop()[i]->fit().dead())
			      { 
				tbb::parallel_sort(centers.begin(),
						   centers.end(),
						   modifier_div::compare_dist_centers<Center, Behavior >(ea.pop()[i]->fit().behavior()));
				
				
				if(max < modifier_div::dist_centers(ea.pop()[i]->fit().behavior().position, centers[0].fit().behavior().position))
				  {
				    max=modifier_div::dist_centers(ea.pop()[i]->fit().behavior().position, centers[0].fit().behavior().position);
				    transferred=i;
				  }
			      }
			  } 
						std::cout<<2<<std::endl;
			for(int i=0;i< boost::fusion:: at_c<1>(ea.fit_modifier()).archive().size();i++)
			  {

			    tbb::parallel_sort(centers.begin(),
					       centers.end(),
					       modifier_div::compare_dist_centers<Center, Behavior >(*boost::fusion:: at_c<1>(ea.fit_modifier()).archive()[i]));
			    
			    
			    if(max < modifier_div::dist_centers(boost::fusion:: at_c<1>(ea.fit_modifier()).archive()[i]->position, centers[0].fit().behavior().position))
			      {
				max=modifier_div::dist_centers(boost::fusion:: at_c<1>(ea.fit_modifier()).archive()[i]->position, centers[0].fit().behavior().position);
				in_archive=true;
				transferred=i;
			      }
			  }

		      }
		    std::cout<<"indiv transfered = "<<transferred<< " et max "<<max<<std::endl;
		    */		    
#endif
                }
                while ((!in_archive && (ea.pop()[transferred]->fit().objs()[0]<=-10000 ||ea.pop()[transferred]->fit().dead()|| was_transfered(*ea.pop()[transferred])))
		       || (in_archive && was_transfered(boost::fusion:: at_c<1>(ea.fit_modifier()).archive()[transferred-ea.pop().size()])));
		
		Center center;
		if(!in_archive)
		  {
		    std::cout<<"in pop"<<std::endl;
		    ofile << ea.pop()[transferred]->fit().objs()[0] << " " << ea.pop()[transferred]->fit().objs()[1] << " ";

		    center = *ea.pop()[transferred];
		  }
		else
		  {
		    ofile << 99 << " " <<99 << " ";
		    for(int k=0;k<center.gen().size();k++)
		      center.gen().set_data(k,boost::fusion:: at_c<1>(ea.fit_modifier()).archive()[transferred-ea.pop().size()]->controller[k]/0.25);
		    center.develop();
		    center.fit().eval(center);

		  }
               
		center.fit().transfer(center,transfer_number,ofile);
		
                centers.push_back(center);
		std::cout<<"end transfer"<<std::endl;
		/*	for(int i=0;i < ea.pop()[transferred]->gen().size();++i)
		  ofile << ea.pop()[transferred]->gen().data_index(i) << " ";
		ofile << std::endl;
		*/
                //if addition
                model.learn(centers);
		std::cout<<"end transfer2"<<std::endl;
		gettimeofday(&timev_prev, NULL);
            }
#endif


            parallel::init();
            parallel::p_for(parallel::range_t(0, ea.pop().size()),
                            modifier_div::_parallel_disparity<typename Ea::phen_t,Center>(ea.pop(),centers,model));

	    
	    
	    parallel::p_for(parallel::range_t(0,boost::fusion:: at_c<1>(ea.fit_modifier()).archive().size()), modifier_div::_parallel_disparity_archive<Center>(boost::fusion:: at_c<1>(ea.fit_modifier()).archive(),centers,model));


        }
    protected:
	    template<typename Phen >
      bool was_transfered(const Phen  phen)const
      {
	for(int i=0;i<centers.size();i++)
	  {
	    bool transfered = true;
	    for(int j=0;j<centers[i].size();j++)
	      {
		transfered=transfered && centers[i].data(j)==phen.data(j);
	      }
	    if(transfered)
	      return true;
	  }
	return false;
      }
      bool was_transfered(const boost::shared_ptr<Behavior> pbeha)const
      {
	for(int i=0;i<centers.size();i++)
	  {
	    bool transfered = true;
	    for(int j=0;j<centers[i].size();j++)
	      {
		transfered=transfered && centers[i].data(j)==pbeha->controller[j]/0.25;
	      }
	    if(transfered)
	      return true;
	  }
	return false;
      }
    };
}
#endif
