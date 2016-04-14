
//#define EVAL_ALL
#define EIGEN_USE_NEW_STDVECTOR
#include <Eigen/StdVector>
#include <Eigen/Core>
#include <Eigen/Array>
#include <time.h>
#include <netinet/in.h>
//#define NO_WHEEL
#define Z_OBSTACLE
//#define GRAPHIC
#include <unistd.h>
#include <iostream>
#include <numeric>
#include <tbb/parallel_reduce.h>
#include <boost/foreach.hpp>
#include <boost/assign/list_of.hpp>
#include <boost/thread.hpp>
#include <boost/asio.hpp>
#include <sferes/phen/parameters.hpp>
#include <sferes/gen/evo_float.hpp>
#include <sferes/gen/sampled.hpp>
//#include <sferes/ea/nsga2.hpp>

#include "nsga2_genocrowd.hpp"
#include <sferes/stat/pareto_front.hpp>
#include <sferes/modif/dummy.hpp>
#include <sferes/run.hpp>
#include <sferes/stc.hpp>


#include "mole_neigh.hpp"
#include "mole_grid.hpp"

#include "behavior.hpp"

#define NO_MPI


//#define NO_PARALLEL
//#define NODIV

#include "ode/box.hh"

//#define GRAPHIC


//#undef GRAPHIC
#ifdef GRAPHIC
#define NO_PARALLEL
#include "renderer/osg_visitor.hh"
#endif


#ifndef NO_PARALLEL
#include <sferes/eval/parallel.hpp>
#ifndef NO_MPI
#include <sferes/eval/mpi.hpp>
#endif
#else
#include <sferes/eval/eval.hpp>
#endif



#include "hexapod.hh"
#include "simu.hpp"
#ifdef ROBOT
#include "robotHexa.hpp"
#endif
#include "diversity_modifier.hpp"
//#include "svm_modifier.hpp"




#include <boost/fusion/sequence/intrinsic/at.hpp>



using namespace sferes;
using namespace boost::assign;
using namespace sferes::gen::evo_float;




struct Params
{
  struct surrogate
  {
    static const int nb_transf_max = 10;
    static const float tau_div = 0.05f;
    //SFERES_ARRAY(float, features_norm, 1.0f, 0.5f, 0.5f);
  };

  struct sampled
  {
    SFERES_ARRAY(float, values,0,0.25,0.5,0.75,1)
    static const float mutation_rate=0.05f;
    static const float cross_rate = 0.00f;
    static const bool ordered = true;
  };
  struct evo_float
  {
    static const float cross_rate = 0.0f;
    static const float mutation_rate = 1.0f/36.0f;
    static const float eta_m = 10.0f;
    static const float eta_c = 10.0f;
    static const mutation_t mutation_type = polynomial;
    static const cross_over_t cross_over_type = sbx;
  };
  struct pop
  {
    static const unsigned size = 200;
    static const unsigned nb_gen = 10001;
    static const int dump_period = 50;
    static const int initial_aleat = 1;

  };
  struct parameters
  {
    static const float min = 0.0f;
    static const float max = 1.0f;
  };
};



 typedef gen::EvoFloat<36, Params> genom_t;
//typedef gen::Sampled<36,Params> genom_t;



struct ordering
{
  bool operator ()(std::pair<int, float> const& a, std::pair<int, float> const& b)
  {
    return a.second < b.second;
  }
};




///variables globales------------


namespace global
{
  boost::shared_ptr<robot::Hexapod> robot;
  boost::shared_ptr<ode::Environment_hexa> env;
  boost::shared_ptr<ode::Environment_hexa> env2;
#ifndef ROBOT
  boost::shared_ptr<robot::Hexapod> real_robot;
#else
  boost::shared_ptr<RobotHexa> real_robot;



#endif
  std::vector<int> brokenLegs;
  std::vector<int> brokenLegs_simu;
     
};
///---------------------------



void init_simu(  int argc ,char** argv,bool master)
{
  global::env = boost::shared_ptr<ode::Environment_hexa>(new ode::Environment_hexa());
  global::env2 = boost::shared_ptr<ode::Environment_hexa>(new ode::Environment_hexa());

  //global::brokenLegs_simu.push_back(1);
  //global::brokenLegs_simu.push_back(4);
  global::robot = boost::shared_ptr<robot::Hexapod>(new robot::Hexapod(*global::env, Eigen::Vector3d(0, 0, 0.1),global::brokenLegs_simu));

#ifndef ROBOT


    //global::brokenLegs.push_back(0);
    //global::brokenLegs.push_back(1);
    //global::brokenLegs.push_back(2);
    //global::brokenLegs.push_back(3);
    //global::brokenLegs.push_back(4);
    //global::brokenLegs.push_back(5);


  global::real_robot = boost::shared_ptr<robot::Hexapod>(new robot::Hexapod(*global::env2, Eigen::Vector3d(0, 0, 0.1),global::brokenLegs));//a robot with the leg number 0 broken
#else
    if(master)
    global::real_robot->initRosNode(argc , argv);



#endif

  float step = 0.001;

  // low gravity to slow things down (eq. smaller timestep?)
  global::env->set_gravity(0, 0, -15);
  bool stabilized = false;
  int stab = 0;
  for (size_t s = 0; s < 1000 && !stabilized; ++s)
    {
      Eigen::Vector3d prev_pos = global::robot->pos();
      global::robot->next_step(step);
      global::env->next_step(step);


      if ((global::robot->pos() - prev_pos).norm() < 1e-5)
	stab++;
      else
	stab = 0;
      if (stab > 100)
	stabilized = true;
    }
  assert(stabilized);
#ifndef ROBOT



    global::env2->set_gravity(0, 0, -15);
    stabilized = false;
    stab = 0;
    for (size_t s = 0; s < 1000 && !stabilized; ++s)
    {
        Eigen::Vector3d prev_pos = global::real_robot->pos();
        global::real_robot->next_step(step);
        global::env2->next_step(step);

        if ((global::real_robot->pos() - prev_pos).norm() < 1e-5)
            stab++;
        else
            stab = 0;
        if (stab > 100)
            stabilized = true;
    }
    global::env2->set_gravity(0, 0, -9.81);
    assert(stabilized);
   
   
    #endif    
  global::env->set_gravity(0, 0, -9.81);



}






SFERES_FITNESS(FitAdapt, sferes::fit::Fitness)
{
 public:
  template<typename Indiv>

    void eval(Indiv& indiv, bool write_objs = false)
  {

    this->_objs.resize(2);
    std::fill(this->_objs.begin(), this->_objs.end(), 0);
    _dead=false;
    _eval(indiv, write_objs);
  }

    template<class Archive>
      void serialize(Archive & ar, const unsigned int version)
    {
      dbg::trace trace("fit", DBG_HERE);
      //ar & BOOST_SERIALIZATION_NVP(this->_value);
      //ar & BOOST_SERIALIZATION_NVP(this->_objs);

      ar & boost::serialization::make_nvp("_value", this->_value);
      ar & boost::serialization::make_nvp("_objs", this->_objs);

      //ar & BOOST_SERIALIZATION_NVP(legs_features);
      //ar & BOOST_SERIALIZATION_NVP(features);
      //ar & BOOST_SERIALIZATION_NVP(disparity);

      //ar & BOOST_SERIALIZATION_NVP(_dist);
      ar & BOOST_SERIALIZATION_NVP(_behavior);
      //ar & BOOST_SERIALIZATION_NVP(_arrival_angle);
      //ar & BOOST_SERIALIZATION_NVP(_direction);
      ar & BOOST_SERIALIZATION_NVP(_covered_distance);
    ar & BOOST_SERIALIZATION_NVP(_dead);
    }
    /*  template<typename Indiv>
    void transfer(Indiv& indiv,int nbtransf,std::ofstream& ofile)
  {
    _transfer(indiv,nbtransf,ofile);
    }*/
    //std::vector<Eigen::Vector3d> get_traj(){return _traj;}
  void set_novelty(float n){this->_objs[0]=n;}
  Behavior& behavior(){return this->_behavior;}
  const Behavior& behavior() const{return this->_behavior;}
  //  float direction(){return this-> _direction;}
  float covered_distance(){return this-> _covered_distance;}
  //float arrival_angle(){return this->_arrival_angle;}
  bool dead(){return this->_dead;}


  //std::vector<std::vector<float> > legs_features;
  //std::vector<float>  features;
  //std::vector<float>  disparity;

 protected:
  //float _dist;
  Behavior _behavior;
  //float _arrival_angle;
  //float _direction;
  float _covered_distance;
  bool _dead;
  //std::vector<Eigen::Vector3d> _traj;  
  template<typename Indiv>
    void _eval(Indiv& indiv, bool write_objs)
  {
    //          std::cout <<"debut eval "<<indiv.size()<<std::endl;
    // copy of controler's parameters

    //std::cout<<"simu eval"<<std::endl;
    _behavior.controller.clear();

    for (int i=0;i < indiv.size();++i)
      {
	//std::cout<<indiv.data(i)<<" ";
	_behavior.controller.push_back(indiv.data(i));
      }
    //std::cout<<std::endl;

    if (this->mode() == sferes::fit::mode::view)
      {

	//            return;
      }



    // std::cout<<std::endl;


    //launching the simulation
    
    //    std::cout <<"debut simu eval "<<std::endl;
    Simu simu = Simu(_behavior.controller, global::robot,global::brokenLegs_simu);
    //std::cout <<"fin simu"<<std::endl;
    _covered_distance=simu.covered_distance();
    
    _behavior.covered_distance=_covered_distance;

    //        std::cout<<"fin recup donnés"<<std::endl;



	
	   
	if(this->_covered_distance<-1000)
      {
	_dead=true;
	//mort subite
	_behavior.duty_cycle.resize(6);
	_behavior.duty_cycle[0]=0;
	_behavior.duty_cycle[1]=0;
	_behavior.duty_cycle[2]=0;
	_behavior.duty_cycle[3]=0;
	_behavior.duty_cycle[4]=0;
	_behavior.duty_cycle[5]=0;
	_behavior.covered_distance=-10000;

      }
    else
      _behavior.duty_cycle= simu.get_duty_cycle();

    //	this->_objs[0]=this->_covered_distance;
    //writting objectives if needed
    if (write_objs)
      {
	std::cout<<std::endl<<"fitness" << this->_objs[0] << std::endl;

      }
    //        std::cout<<"fin eval"<<std::endl;
  }
  /*  template<typename Indiv>
    void _transfer(Indiv& indiv,int nbtransf, std::ofstream& ofile)
  {
    
    std::vector<std::vector<float> > features_transf;
    

    std::vector<float> ctrl;
    for (int i=0;i < indiv.size();++i)
      {

	ctrl.push_back(indiv.data(i));
      }
    //std::cout <<"transfered controler"<< ctrl[0] << " " << ctrl[1] << " " << ctrl[2] << std::endl;
    std::cout<<"simu transf"<<std::endl;
    std::cout<<"result simu X: "<<this->behavior().position[0]<<" Y: "<<this->behavior().position[1]<<std::endl;

    //	assert(global::real_robot);
    Simu simu = Simu(ctrl, global::real_robot,std::vector<int>(),false,3,nbtransf);
    std::cout<<"end simu"<<std::endl;


    disparity.resize(1);


    if (this->_objs[0]<=-10000)
      disparity[0]=-10000;
    else
      {
	std::cout<<this->behavior().position.size() <<" taille "<<std::endl;
	disparity[0]=-(rapport_dist(this->behavior().position,simu.final_pos()));

      }
    std::cout<<"disp :"<<disparity[0]<< " fit real :"<<simu.covered_distance()<<" simu :"<<this->_dist<<"disparité lié a la fit"<<fabs(simu.covered_distance()-this->_dist)/fabs(this->_dist)<<  std::endl;

    ofile << " x_real: "<<simu.final_pos()[0]<< " y_real:  "<<simu.final_pos()[1]<<" x_simu: "<<this->behavior().position[0]<<" y_simu: "<<this->behavior().position[1]<<" distance(disparity): " << disparity[0] <<" Final Angle "<<simu.arrival_angle() <<" slam duration" << simu.slam_duration() << " "<<std::endl;

    _traj=simu.get_traj();




  }
  */


  float rapport_dist(const std::vector<float>& v1, const std::vector<float>& v2)
{
  assert(v1.size()>=2 && v2.size()>=2);
    std::cout<<"simu ";
for(int i=0;i<v1.size();i++)
    std::cout<< v1[i]<<" ";
  std::cout<<std::endl;
    std::cout<<"real ";
  for(int i=0;i<v2.size();i++)
    std::cout<< v2[i]<<" ";
  std::cout<<std::endl;


	    
  float B= sqrt((v1[0]/2)*(v1[0]/2)+(v1[1]/2)*(v1[1]/2));
  float alpha=atan2(v1[1],v1[0]);
  float A= B/cos(alpha);
	    
  float beta=atan2(v1[1],v1[0]-A);
  float angle;
  if(v1[0]>=0)
    angle=beta-M_PI;
  else
    angle=beta;
  while(angle<-M_PI)
    angle+=2*M_PI;
  while(angle>M_PI)
    angle-=2*M_PI;
	    


  float curv=fabs(A*angle);


  typename std::vector<float>::const_iterator it1 = v1.begin(), it2 = v2.begin();
  float res = 0.0f;
  while (it1 != v1.end())
    {
      float v = (float)*it1 - (float)*it2;
      res += v * v;
      ++it1;
      ++it2;
    }
  return sqrtf(res);// /curv;
}




};








template<typename V1, typename V2>
float dist(const V1& v1, const V2& v2)
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




template<typename T>
struct compare_dist_f
{
  compare_dist_f(const T& v) : _v(v) {}
  const T _v;
  bool operator()(const T& v1, const T& v2) const
  {
    assert(v1.duty_cycle.size() == _v.duty_cycle.size());
    assert(v2.duty_cycle.size() == _v.duty_cycle.size());
    return dist(v1.duty_cycle, _v.duty_cycle) < dist(v2.duty_cycle, _v.duty_cycle);
  }
};

template<typename T, typename T2>
struct compare_dist_p
{
  compare_dist_p(const T2& v) : _v(v) {}
  const T2 _v;
  bool operator()(const T& v1, const T& v2) const
  {
    assert(v1->duty_cycle.size() == _v.duty_cycle.size());
    assert(v2->duty_cycle.size() == _v.duty_cycle.size());
    return dist(v1->duty_cycle, _v.duty_cycle) < dist(v2->duty_cycle, _v.duty_cycle);
  }
};



SFERES_CLASS(LogModifier)
{
 public:
  template<typename Ea>
    void apply(Ea& ea)
  {
    _log_behaviors(ea);
  }

 protected:
  boost::shared_ptr<std::ofstream> _log_file;

  template<typename E>
    void _log_behaviors(const E& ea)
  {
    this->_create_log_file(ea, "all_behaviors.dat",_log_file);
    

    for (size_t i = 0; i < ea.pop().size(); ++i)
      {
	const Behavior& v = ea.pop()[i]->fit().behavior(); 
	(*this->_log_file)<<ea.gen()<< " ";
	for (size_t j = 0; j < v.duty_cycle.size(); ++j)
	  (*this->_log_file)<<v.duty_cycle[j]<<" ";
	(*this->_log_file)<<"    "<<v.covered_distance<<"        "<<ea.pop()[i]->fit().obj(0)<<"         "<<ea.pop()[i]->fit().obj(1)<<std::endl;
      }
  }
  template<typename E>
    void _create_log_file(const E& ea, const std::string& name, boost::shared_ptr<std::ofstream>& log_file)
  {
    if (!log_file && ea.dump_enabled())
      {
	std::string log = ea.res_dir() + "/" + name;
	log_file = boost::shared_ptr<std::ofstream>(new std::ofstream(log.c_str()));
      }
 }
  
};


SFERES_CLASS_D(NoveltyArchive, LogModifier)
{
 public:
  NoveltyArchive() : _rho_min(0.45f), _not_added(0) {}
  template<typename Ea>
    void apply(Ea& ea)
  {
    //std::cout<<"debut modif novelty"<<std::endl;
    static const    size_t  k = 15; //voisinage
     
    float max_n = 0;
    int added = 0;
    int swapped=0;
    std::vector<Behavior> archive;
    for (size_t i = 0; i <_archive.size(); ++i)
      archive.push_back(*_archive[i]);

#ifndef NOVELTY_NO_DIV 
    for (size_t i = 0; i < ea.pop().size(); ++i)
      archive.push_back(ea.pop()[i]->fit().behavior());
#endif
    for (size_t i = 0; i < ea.pop().size(); ++i)
      {


	if(ea.pop()[i]->fit().dead())
	   {
	     ea.pop()[i]->fit().set_novelty(-10000);
	     ea.pop()[i]->fit().set_obj(1,-10000);//local comp1
	     //	     ea.pop()[i]->fit().set_obj(2,-10000);//local comp2
	     continue;
	   }
	const Behavior& behavior = ea.pop()[i]->fit().behavior();
	const float covered_distance=behavior.covered_distance;

 	tbb::parallel_sort(archive.begin(),
			   archive.end(),
			   compare_dist_f<Behavior >(behavior));
	float n = 0;
	int rank_distance=0;

	

	//voisinage géometrique
	int j=0;

	/*	while(j<archive.size() && dist(archive[j].position,behavior.position)<_rho_min*2)
	  {
	    n+= dist(archive[j].position, behavior.position);
	    if(drift>fabs(archive[j].drift))// Attention valeur positive qu'on veut minimiser (nulle)
	      rank_direction --;
	    if(transferability>fabs(archive[j].transferability)) //Attention valeur positive  qu'on veut minimiser (nulle) 
	      rank_transferability --;
	    
	    j++;
	  }
	n/=j;
	std::cout<<"concerned indiv="<< j<< "for archive size="<< archive.size()<<std::endl;*/
	
	//voisinage par nombre de voisin
	//	std::cout<<covered_distance<<"  current"<<std::endl;
	if (archive.size() > k)
	  for (size_t j = 0; j < k; ++j)
	    {
	      // std::cout<<archive[j].covered_distance<<std::endl;
	      n += dist(archive[j].duty_cycle, behavior.duty_cycle);
	      if(covered_distance<archive[j].covered_distance)// Attention valeur positive qu'on veut maximiser (nulle)
		rank_distance --;

	    }

	else
	  n += 1;
	n /= k;
	
	//	std::cout<<rank_distance <<"  rank distance"<<std::endl<<std::endl;
	max_n = std::max(n, max_n);
	
	ea.pop()[i]->fit().set_novelty(n);
	ea.pop()[i]->fit().set_obj(1,rank_distance);//local comp1

	

	if (_archive.size() < k || n > _rho_min)
	  {
	    //std::cout<<"ajout"<<std::endl;
	    
	    tbb::parallel_sort(_archive.begin(),
			       _archive.end(),
			       compare_dist_p<boost::shared_ptr<Behavior>,Behavior >(behavior));
	    if(_archive.size()==0 || _archive[0]->duty_cycle[0]!=behavior.duty_cycle[0] ||_archive[0]->duty_cycle[1]!=behavior.duty_cycle[1] ||_archive[0]->duty_cycle[2]!=behavior.duty_cycle[2] ||_archive[0]->duty_cycle[3]!=behavior.duty_cycle[3] ||_archive[0]->duty_cycle[4]!=behavior.duty_cycle[4] ||_archive[0]->duty_cycle[5]!=behavior.duty_cycle[5] ||  _archive[0]->covered_distance!=behavior.covered_distance )// A REFAIRE PLUS PROPREMENT
	      {
		/*
		//assert(behavior.size() == 2);
		std::pair<std::vector<float>,std::vector<float> > indiv;
		indiv.first=behavior;
		for (int p=0;p < ea.pop()[i]->size();++p)
		{
		//  std::cout<<indiv.data(i)<<" ";
		indiv.second.push_back(ea.pop()[i]->data(p));
		}
		*/


		_archive.push_back(boost::shared_ptr<Behavior>(new Behavior(behavior)));
	
		_not_added = 0;
		++added;
	      }
	  }
	else
	  {
	    ++_not_added;
	    // change individu if it better than the nearest
	    //std::cout<<"change ?"<<std::endl;
	    tbb::parallel_sort(_archive.begin(),
			       _archive.end(),
			       compare_dist_p<boost::shared_ptr<Behavior>,Behavior >(behavior));
	    //	    if( (fabs(behavior.transferability)>0.05 && fabs(behavior.transferability) <fabs( _archive[0]->transferability)) || (behavior.transferability <= 0.05 && fabs(_archive[0]->drift) > fabs(behavior.drift) ) )
	    if(fabs(behavior.covered_distance)>fabs(_archive[0]->covered_distance))
	      {
		//std::cout<<"yes"<<std::endl;;
		_archive[0]=boost::shared_ptr<Behavior>(new Behavior(behavior));
		swapped++;
		/*for (int p=0;p < ea.pop()[i]->size();++p)
		  {
		  //  std::cout<<indiv.data(i)<<" ";
		  _archive[0].second[p]=(ea.pop()[i]->data(p));
		  }
		*/
	      }
	    
	  }
      }
   
    /*    if (_not_added > 25000)
      _rho_min *= 0.95;
    if (_archive.size() > k  && added > 4)
    _rho_min *= 1.05f;*/

    std::cout<<"archive size:"<<_archive.size()<<" rho:"<<_rho_min<<" max="<<max_n<< " add=" << added<<" swapped="<< swapped<<std::endl;
    if (ea.gen() % Params::pop::dump_period == 0)
      {
	// créer un flux de sortie
	std::ostringstream oss;
	// écrire un nombre dans le flux
	oss << ea.gen();
	// récupérer une chaîne de caractères
	std::ofstream ofs((ea.res_dir() + "/archive"+oss.str()+".dat").c_str());
	for (size_t i = 0; i < _archive.size(); ++i)
	  {
	    //	    assert(_archive[i].size() == 2);
	    
	    ofs<<_archive[i]->duty_cycle[0]<<" "<<_archive[i]->duty_cycle[1]<<" "<<_archive[i]->duty_cycle[2]<<" "<<_archive[i]->duty_cycle[3]<<" "<<_archive[i]->duty_cycle[4]<<" "<<_archive[i]->duty_cycle[5]<<"    "<<_archive[i]->covered_distance<<"    ";
	    for(int j=0;j<_archive[i]->controller.size();j++)
	      ofs<<_archive[i]->controller[j]<<" ";
	    ofs<<std::endl;
	  }
      }


    this->_log_behaviors(ea);
    this->log_best_indiv(ea);
  }
      
  std::vector<boost::shared_ptr<Behavior> > & archive() { return _archive; }
 protected:
  std::vector<boost::shared_ptr<Behavior> > _archive; 

  float _rho_min;
  size_t _not_added;
  boost::shared_ptr<std::ofstream> _log_file_best_indiv;

  template<typename E>
  void log_best_indiv(const E& ea)
  { 
    this->_create_log_file(ea, "best_behaviors.dat",_log_file_best_indiv);
    float best=0;
    for(int i=0; i<_archive.size();i++)
      if (_archive[i]->covered_distance>best)
	best=_archive[i]->covered_distance;
   
    (*this->_log_file_best_indiv)<<ea.gen()<< " "<<best<<std::endl;    
  }
    
};

struct elem_archive
{
  std::vector<float> duty_cycle;
  float fit;
  std::vector<float> controller;

};



  typedef FitAdapt<Params> fit_t;
  typedef phen::Parameters<genom_t, fit_t, Params> phen_t;


void lecture(int argc, char **argv)
{
#ifdef ROBOT
  global::real_robot = boost::shared_ptr<RobotHexa>(new RobotHexa);
#endif
  //initilisation of the simulation and the simulated robot
  init_simu(argc, argv,true);
  
  
  
  std::vector<elem_archive> archive;
  std::ifstream monFlux(argv[1]);  //Ouverture d'un fichier en lecture 
  if(monFlux)
    {
      while(!monFlux.eof())
        {
	  elem_archive elem;
          for(int i =0;i<43;i++)
            {
              if(monFlux.eof())
                break;
              float data;
              monFlux>>data;
	      if(i<=5)
		elem.duty_cycle.push_back(data);
	      if(i==6)
		elem.fit=data;
	      if(i>=7)
		elem.controller.push_back(data);
          
            }
          if(elem.controller.size()==36)
            archive.push_back(elem);
          
        }
    }
  else
    {
      std::cout << "ERREUR: Impossible d'ouvrir le fichier en lecture." << std::endl;
      return ;
    }

  std::cout<<archive.size()<<" controllers loaded"<<std::endl;
  std::ofstream ofile("evaluation.dat", std::ios_base::app);  
  int i=0;
  int error=0;
  if(argc==3)
    i=atoi(argv[2]);
  for(;i<archive.size();i++)
    {
      if(i%100==0)
	std::cout<<"progression: "<<(float)i/archive.size()<<std::endl;
      //      ofile<<i<<": ";                                                                                                               
      phen_t indiv;
      for(int k=0;k<indiv.gen().size();k++)
        indiv.gen().data(k,archive[i].controller[k]);
      indiv.develop();
      indiv.fit().eval(indiv);

      if(fabs(indiv.fit().behavior().covered_distance-archive[i].fit)>0.01)
        {
	  error++;
	  std::cout<<error<< " : "<<fabs(indiv.fit().behavior().covered_distance-archive[i].fit)<<std::endl;
        }



      /*      for(int j=0;j<archive[i].duty_cycle.size();j++)
	ofile<< archive[i].duty_cycle[j]<<" ";
      ofile<< "\t"<<archive[i].fit<<"\t "<<indiv.fit().behavior().covered_distance<<std::endl; 
      float data;*/
      //      std::cout<<"x:simu :"<<indiv.fit().behavior().position[0]<<" x archive: "<<archive[i].x_simu<<" y: simu : "<<indiv.fit().behavior().position[1]<<" y archive: "<<archive[i].y_simu<<std::endl;

      //ofile<<"x:simu :"<<indiv.fit().behavior().position[0]<<" x:real: "<<archive[i].x_simu<<" y:simu : "<<indiv.fit().behavior().position[1]<<" y:real: "<<archive[i].y_simu<<std::endl;


      
  

      /*int format_width = (int) log10(archive.size()) +1;
      std::string fname =  std::string("traj_") + boost::str(boost::format("%1%")%boost::io::group(std::setfill('0'),std::setw(format_width),i));//format_width %gen);//+ boost::lexical_cast<std::string>(gen);
      std::ofstream trajfile(fname.c_str(),std::ios_base::app);


      std::vector<Eigen::Vector3d> traj=indiv.fit().get_traj();
      	for (int i =0;i<traj.size();i++)
	  {
	    trajfile<<traj[i][0]<<" "<<traj[i][1]<<" "<<traj[i][2]<<std::endl;
	    }*/


    }
  

}

int main(int argc, char **argv)
{





  

   //initialisation of the simulator
  dInitODE();






#ifndef NO_PARALLEL
#ifndef NO_MPI
    typedef eval::Mpi<Params> eval_t;    
#else
    typedef eval::Parallel<Params> eval_t;
#endif
#else

    typedef eval::Eval<Params> eval_t;
#endif





    //    typedef boost::fusion::vector<sferes::stat::ParetoFront<phen_t, Params> >  stat_t;
    typedef boost::fusion::vector<>  stat_t;
  //typedef CenterModifier<Params> modifier_t;

  //typedef boost::fusion::vector< SvmModifier<phen_t, Params>, NoveltyArchive<Params > > modifier_t;
    //typedef boost::fusion::vector<  NoveltyArchive<Params > > modifier_t;


  //typedef NoveltyArchive<Params> modifier_t;
    //  typedef ea::Nsga2<phen_t, eval_t, stat_t, modifier_t, Params> ea_t;
    typedef ea::MoleGrid<phen_t, eval_t, stat_t, modif::Dummy<Params>, Params> ea_t;

  if(argc ==37)
    { 

#ifdef ROBOT
	 std::cout<<"init ROBOT"<<std::endl;
	global::real_robot = boost::shared_ptr<RobotHexa>(new RobotHexa);
#endif
      
	init_simu(argc, argv,true);
	//init_simu(argc, argv,false);                                                   
      genom_t indiv;
      std::cout << "LOADING..." << std::endl;
      std::vector<float> ctrl;
      for(int i=0;i < 36;++i)
	   {
	     indiv.data(i, atof(argv[i+1]));
	     //indiv.set_data(i, atof(argv[i+1])*4);
	     ctrl.push_back(atof(argv[i+1]));
	   }
      fit_t fitness;
      std::cout << "fin loading..." << std::endl;
      fitness.eval(indiv);
      Simu simu = Simu(ctrl, global::real_robot,std::vector<int>(),true,5);
  
      std::cout<<"covered_distance simu: "<<fitness.covered_distance()<<std::endl;
      std::cout<<"covered_distance real: "<<simu.covered_distance()<<std::endl;
      std::cout<<"duty cycles:"<<std::endl;
      
      for(int i=0;i< fitness.behavior().duty_cycle.size();i++)
	std::cout<<fitness.behavior().duty_cycle[i]<<" ";
      std::cout<<std::endl;
      
      global::robot.reset();

      global::env.reset();
      dCloseODE();

      return 0;
    }

  // lecture fichier results.dat
    if(argc == 2 || argc ==3)
    lecture(argc, argv);
  


    
  ea_t ea;

#ifndef NO_MPI
    if (ea.eval().rank() == 0)
      {
#ifdef ROBOT
	 std::cout<<"init ROBOT"<<std::endl;
	global::real_robot = boost::shared_ptr<RobotHexa>(new RobotHexa);
#endif
	//initilisation of the simulation and the simulated robot
	 std::cout<<"init SIMU"<<std::endl;
	init_simu(argc, argv,true);
      }
    else
      init_simu(argc, argv,false);
#else
#ifdef ROBOT
	global::real_robot = boost::shared_ptr<RobotHexa>(new RobotHexa);
#endif
	//initilisation of the simulation and the simulated robot
	init_simu(argc, argv,true);

#endif
 std::cout<<"debut run"<<std::endl;

  run_ea(argc, argv, ea);
  std::cout <<"fin run"<<std::endl;

  global::robot.reset();

  global::env.reset();
  dCloseODE();
  std::cout <<"fin"<<std::endl;
  return 0;
}
