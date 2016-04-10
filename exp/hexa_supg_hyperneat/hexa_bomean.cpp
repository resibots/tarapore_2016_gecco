#define SHOW_TIMER
#include <bayesian_optimization/bayesian_optimization.hpp>
#include <bayesian_optimization/inner_cmaes.hpp>
#include "exhaustiveSearchMap.hpp"
#include "meanMap.hpp"
#include "statTransferts.hpp"
#include "statObservations.hpp"
#include "tbb/tbb.h"

#ifdef GRAPHIC
#define NO_PARALLEL
#include "renderer/osg_visitor.hh"
#endif

#include "hexapod.hh"
#include "simu.hpp"

#define NO_PARALLEL
#include "sferes/parallel.hpp"
#define NOISEADDED
//=======================

#include <Eigen/Core>
#include <time.h>
#include <netinet/in.h>
#define Z_OBSTACLE
#include <unistd.h>
#include <numeric>
#include <tbb/parallel_reduce.h>
#include <boost/foreach.hpp>
#include <boost/assign/list_of.hpp>
#include <boost/thread.hpp>
#include <boost/asio.hpp>

#include <fstream>
#include <iostream>
#include <bitset>
#include <limits>
#include <bitset>

#include <boost/random.hpp>
#include <boost/multi_array.hpp>
#include <boost/serialization/bitset.hpp>

#include <sferes/dbg/dbg.hpp>
#include <sferes/stc.hpp>
#include <sferes/run.hpp>
#include <sferes/misc.hpp>
#include <sferes/phen/parameters.hpp>
#include <sferes/gen/evo_float.hpp>
#include <sferes/ea/ea.hpp>
#include <sferes/fit/fitness.hpp>
#include <sferes/modif/dummy.hpp>
#include <sferes/eval/parallel.hpp>
#include <sferes/eval/eval.hpp>
#include <sferes/stat/best_fit.hpp>

#include <modules/nn2/phen_dnn.hpp>

#include "behavior.hpp"
#include "gen_hnn.hpp"
#include "phen_hnn.hpp"

#include <modules/map_elite/map_elite.hpp>
#include <modules/map_elite/fit_map.hpp>
#include <modules/map_elite/stat_map.hpp>

#define NO_MPI

#ifdef GRAPHIC
#define NO_PARALLEL
#include "renderer/osg_visitor.hh"
#endif

#include "hexapod.hh"
#include "simu.hpp"
#include <boost/fusion/sequence/intrinsic/at.hpp>

//=======================
#ifdef ROBOT
#include <ros/ros.h>
#include <std_msgs/String.h>
#include <std_msgs/Float32.h>
#include <hexa_control/Transfert.h>
#endif
//=======================




using namespace bayesian_optimization;

using namespace sferes;
using namespace boost::assign;
using namespace sferes::gen::dnn;
using namespace sferes::gen::evo_float;

struct Params
{    
    struct panne{
        static int number;
    };

    struct ea // for loading the population using map_elite/stat_map
    {
        SFERES_CONST size_t behav_dim = 6; SFERES_ARRAY(size_t, behav_shape, 5, 5, 5, 5, 5, 5);
    };
    struct pop //Parameters of the population // for loading the population using sferes // just kept to avoid compile time error. values not actually used
    {
        SFERES_CONST size_t init_size = 0; //Using the same parameters as Antoine to compare results //1000
        SFERES_CONST size_t size = 0; //Using the same parameters as Antoine to compare results //64
        SFERES_CONST size_t nb_gen = 0; // number of evaluations is nb_gen * size of batch
        static constexpr int dump_period = -1;
    };

    struct boptimizer
    {
        static constexpr int dim=6;
        //static constexpr int dim=3;

        static constexpr int bd_res=5;
        //static constexpr int bd_res=20;

        static constexpr float noise=0.001;
        static constexpr int dump_period=1;

        struct init
        {
            static constexpr bool initial_sampling=false;
            static constexpr bool continuous=false;
            static constexpr int nb_initial_samples=0;
            static constexpr int nb_pts=5;
            static constexpr bool random=false;
        };
    };
    struct maxiterations
    {
        static  int n_iterations;//=20;
    };

    struct maxpredictedvalue
    {
        static constexpr float ratio=0.9;
    };
    struct kf_exp
    {
        static constexpr float sigma=3;
    };
    struct kf_maternthreehalfs
    {
        static constexpr float sigma=1;
        static constexpr float l=1;
    };
    struct kf_maternfivehalfs
    {
        static constexpr float sigma=1;
        static float l;//=0.5; // 0.25
    };

    struct ucb
    {
        static float alpha;//=0.05;
    };
    struct meanconstant
    {
        static constexpr float constant=1.2258;
    };
    struct meanarchive
    {
        //static const char filename[];
        char filename[];
    };
    struct exhaustivesearch
    {

        static constexpr int nb_pts=4;
    };

    struct archiveparams
    {

        struct elem_archive
        {
            std::vector<float> duty_cycle;
            float fit;
            size_t controller_index;
        };

        struct classcomp {
            bool operator() (const std::vector<float>& lhs, const std::vector<float>& rhs) const
            {
                assert(lhs.size()==Params::boptimizer::dim && rhs.size()==Params::boptimizer::dim);
                //assert(lhs.size()==dim && rhs.size()==6);
                int i=0;
                while(i<(Params::boptimizer::dim - 1) && round(lhs[i]*(Params::boptimizer::bd_res-1))==round(rhs[i]*(Params::boptimizer::bd_res-1))) //lhs[i]==rhs[i])
                    i++;
                return round(lhs[i]*(Params::boptimizer::bd_res-1))<round(rhs[i]*(Params::boptimizer::bd_res-1));//lhs[i]<rhs[i];
            }

        };
        typedef std::map<std::vector<float>,elem_archive,classcomp> archive_t;
        static std::map<std::vector<float>,elem_archive,classcomp> archive;
    };



    struct cppn
    {
        // params of the CPPN
        struct sampled
        {
            SFERES_ARRAY(float, values, 0, 1, 2, 3, 4); //sine; sigmoid; gaussian; linear; copy
            static constexpr float mutation_rate = 0.01f;
            static constexpr float cross_rate = 0.25f;
            static constexpr bool ordered = false;
        };
        struct evo_float
        {
            static constexpr float mutation_rate = 0.1f;//per conn mut rate
            static constexpr float eta_m = 10.0f;  // perturbation of the order O(1/eta_m)

            static constexpr float cross_rate = 0.1f; //we don't use this
            static constexpr mutation_t mutation_type = polynomial;
            static constexpr cross_over_t cross_over_type = sbx;
            static constexpr float eta_c = 10.0f; // A large value ef eta gives a higher probablitity for creating a `near-parent' solutions and a small value allows distant solutions to be selected as offspring.
        };
        struct parameters //Min and max weights of CPPN and of HNN
        {
            static constexpr float min = -2.0f;
            static constexpr float max = 2.0f;
        };

        struct dnn //the parameters for the cppn
        {
#ifndef ORIENTFB
            //x1 and y1, the position of the supg; time since last trigger event + bias input
            static constexpr size_t nb_inputs = 4;
#else
            //x1 and y1, the position of the supg; time since last trigger event, heading orientation f/b, + bias input
            static constexpr size_t nb_inputs = 5;
#endif

            //outputs: supg output and offset to timer and duty factor
            static constexpr size_t nb_outputs = 3;
            static constexpr init_t init = ff; //feedforward nn
            static constexpr float m_rate_add_conn = 0.05f; //0.1f
            static constexpr float m_rate_del_conn = 0.1f;
            static constexpr float m_rate_change_conn = 0.0f; //move conn to new neurons
            static constexpr float m_rate_add_neuron = 0.05f; //0.2f
            static constexpr float m_rate_del_neuron = 0.01f;

            //these only count w/ random init, instead of feed forward
            static constexpr size_t min_nb_neurons = 6;
            static constexpr size_t max_nb_neurons = 20;
            static constexpr size_t min_nb_conns = 30;
            static constexpr size_t max_nb_conns = 150;
        };
    };
    struct hnn
    { //not used as SUPG output (output of cppn) directly drives the motors
        static constexpr size_t nb_rand_mutations = 0;
        static constexpr size_t nb_layers = 1;

        static constexpr size_t nb_inputs = 1;
        static constexpr size_t nb_outputs  = 1;

        typedef float weight_t;
        typedef float io_t;
        typedef nn::params::Vectorf<1> params_t;
        typedef nn::Neuron<nn::PfWSum<io_t>,
        nn::AfSigmoidBias<params_t>, io_t > neuron_t;
        typedef nn::Connection<weight_t, io_t> connection_t;
    };
};

Params::archiveparams::archive_t  load_archive(std::string archive_name);
//std::map<std::vector<float>,Params::mapparams::elem_archive,Params::mapparams::classcomp>  load_archive(std::string archive_name);





namespace global
{

struct timeval timev_selection;  // Initial absolute time (static)

#ifdef ROBOT
boost::shared_ptr<ros::NodeHandle> node;
ros::ServiceClient hexapod;
#endif
std::string res_dir;

std::vector<int> brokenLegs;
boost::shared_ptr<robot::Hexapod> global_robot;
boost::shared_ptr<ode::Environment_hexa> global_env;

size_t ind_index; // index of ind in pop. used in lecture()
};


///---------------------------

#ifdef ROBOT
void init_ros_node(int argc ,char** argv)
{
    ros::init(argc, argv, "hexap_bomean",ros::init_options::NoSigintHandler);
    global::node=boost::shared_ptr<ros::NodeHandle>(new ros::NodeHandle());
    global::hexapod = global::node->serviceClient<hexa_control::Transfert>("Transfert");
}
#endif


void init_simu(int argc ,char** argv,bool master,std::vector<int> broken_legs=std::vector<int>())
{
    global::global_env = boost::shared_ptr<ode::Environment_hexa>(new ode::Environment_hexa(0));

    if(broken_legs.size()==0)
    {
        //global::brokenLegs.push_back(0);
        //global::brokenLegs.push_back(1);
        //global::brokenLegs.push_back(2);
        //global::brokenLegs.push_back(3);
        //global::brokenLegs.push_back(4);
        //global::brokenLegs.push_back(5);
    }
    else
        global::brokenLegs=broken_legs;

    if(global::brokenLegs.size()==0)
        std::cout<<"global_real_robot is undamaged"<<std::endl;
    else
    {
        std::cout<< "legs ";
        for(int i=0;i<global::brokenLegs.size();i++)
            std::cout<<global::brokenLegs[i]<<" ";
        std::cout<<" are removed from global_real_robot"<<std::endl;
    }

    global::global_robot = boost::shared_ptr<robot::Hexapod>(new robot::Hexapod(*global::global_env, Eigen::Vector3d(0, 0, 0.5),global::brokenLegs));


    float step = 0.001;
    // low gravity to slow things down (eq. smaller timestep?)
    global::global_env->set_gravity(0, 0, -9.81);
    bool stabilized = false;
    int stab = 0;
    for (size_t s = 0; s < 1000 && !stabilized; ++s)
    {

        Eigen::Vector3d prev_pos = global::global_robot->pos();
        global::global_robot->next_step(step);
        global::global_env->next_step(step);

        if ((global::global_robot->pos() - prev_pos).norm() < 1e-4)
            stab++;
        else
            stab = 0;
        if (stab > 100)
            stabilized = true;
    }
    assert(stabilized);
    global::global_env->set_gravity(0, 0, -9.81);
}



/*SFERES_FITNESS(BomeanFit, FitMap)
{
    public:
    float covered_distance;

    template<typename Indiv>
    void eval(Indiv& indiv)
    {
        this->_objs.resize(1);
        std::fill(this->_objs.begin(), this->_objs.end(), 0);
        _eval(indiv);
    }

    protected:
    template<typename Indiv>
    void _eval(Indiv& indiv)
    {
        indiv.gen().init();

        if (this->mode() == sferes::fit::mode::view) //is used to view individuals
        {}

        Simu <typename Indiv::gen_t> simu(indiv.gen(), global::global_robot, global::brokenLegs);
        covered_distance= simu.covered_distance();

        if(this->mode() == sferes::fit::mode::view)
        {}
    }
};*/


FIT_MAP(BomeanFit)
{
    public:

    //sum of vector
    template<typename V1>
    float sumofvector(const V1& v1)
    {
        typename V1::const_iterator it1 = v1.begin();
        float res = 0.0f;
        while (it1 != v1.end())
        {
            res += (float)*it1;
            ++it1;
        }
        return res;
    }

    template<typename Indiv>
    void eval(Indiv& indiv)
    {
        this->_objs.resize(1);
        std::fill(this->_objs.begin(), this->_objs.end(), 0);
        _eval(indiv);
    }

    std::vector<std::vector<float> > angles_forfft;
    float servo_frequencies_max;
    float dist;

    protected:

    template<typename Indiv>
    void _eval(Indiv& indiv)
    {
        indiv.gen().init();


        if (this->mode() == sferes::fit::mode::view) //is used to  view individuals in pareto front (their ids. can be got from pareto.data)
        { }

#if defined GRND_OBSTACLES || defined GRND_OBSTACLES_EVO
	std::vector <std::vector <float> > noobstacles;
	float floorangle=0.0;
        Simu <typename Indiv::gen_t> simu(indiv.gen(), global::global_robot, global::brokenLegs, 5.0, noobstacles, floorangle);
#else
        Simu <typename Indiv::gen_t> simu(indiv.gen(), global::global_robot, global::brokenLegs, 5.0, 0.0);
#endif




        this->servo_frequencies_max = simu.servo_frequencies_max;



           std::vector<float> data = {(float)round((sumofvector(simu.get_contact(0)) / (float)simu.get_contact(0).size()) * 100.0f) / 100.0f,
                                      (float)round((sumofvector(simu.get_contact(1)) / (float)simu.get_contact(1).size()) * 100.0f) / 100.0f,
                                      (float)round((sumofvector(simu.get_contact(2)) / (float)simu.get_contact(2).size()) * 100.0f) / 100.0f,
                                      (float)round((sumofvector(simu.get_contact(3)) / (float)simu.get_contact(3).size()) * 100.0f) / 100.0f,
                                      (float)round((sumofvector(simu.get_contact(4)) / (float)simu.get_contact(4).size()) * 100.0f) / 100.0f,
                                      (float)round((sumofvector(simu.get_contact(5)) / (float)simu.get_contact(5).size()) * 100.0f) / 100.0f};
           this->set_desc(data);


        this->_value   = simu.covered_distance();
       /*if(floor(this->servo_frequencies_max) >= 1.0f)
	     this->_value   = simu.covered_distance() / floor(this->servo_frequencies_max);
	   else
         this->_value   = simu.covered_distance();*/


        angles_forfft = simu.angles_forfft;
        servo_frequencies_max = simu.servo_frequencies_max;
        this->dist = simu.covered_distance();


        if(this->mode() == sferes::fit::mode::view)
        {        }
    }
};

namespace global_popstat
{
    using namespace nn;
    typedef BomeanFit<Params> fit_t;
    typedef phen::Parameters<gen::EvoFloat<1, Params::cppn>, fit::FitDummy<>, Params::cppn> weight_t;
    typedef gen::Hnn<weight_t, Params, Params::cppn> gen_t;
    typedef phen::Hnn<gen_t, fit_t, Params> phen_t;

    typedef sferes::stat::Map<phen_t, Params> t_mapstat;

    t_mapstat mapstat;
    gen_t supg_ctrl;
}


template<typename Params>
struct fit_eval_map
{
    fit_eval_map()
    {
        timerclear(&global::timev_selection);
        gettimeofday(&global::timev_selection, NULL);
    }

    float operator()(Eigen::VectorXf x) const
    {
        std::cout<<"start eval"<<std::endl;
        std::vector<float> key(x.size(),0);
        for(int i=0;i<x.size();i++)
            key[i]=x[i];

        if(Params::archiveparams::archive.count(key)==0)
            return -1000;


#ifdef ROBOT
        std::vector<float> ctrl=Params::archiveparams::archive.at(key).controller;

        struct timeval timev_init;  // Initial absolute time (static)

        struct timeval timev_cur;   // Current absolute
        struct timeval timev_duration;  // Current tick position (curent - previous)
        timerclear(&timev_cur);
        gettimeofday(&timev_cur, NULL);
        timersub(&timev_cur, &global::timev_selection, &timev_duration);

        std::ofstream ofile((global::res_dir+"/times.dat").c_str(), std::ios_base::app);
        std::cout<<"selection "<<timev_duration.tv_sec+timev_duration.tv_usec/1e6<<"sec"<<std::endl;
        ofile<<"selection "<<timev_duration.tv_sec+timev_duration.tv_usec/1e6<<"sec"<<std::endl;



        ///- COMMUNICATION WITH ROS/HEXAPOD
        hexa_control::Transfert srv;


        srv.request.duration=-1; // pose zero
        if (global::hexapod.call(srv))
        {
            ROS_INFO("init pose");
        }
        else
        {
            ROS_ERROR("Failed to call service");
            return 1;
        }


        for(int i=0;i<ctrl.size();i++)
            srv.request.params[i]=ctrl[i];

        float obs=0;
        bool ok=false;

        do
        {
            timerclear(&timev_init);
            gettimeofday(&timev_init, NULL);

            srv.request.duration=5;// action
            if (global::hexapod.call(srv))
            {
                ROS_INFO("controller executed");
                obs=srv.response.covered_distance;
            }
            else
            {
                ROS_ERROR("Failed to call service");
                //return 1;
            }
            srv.request.duration=-1; // pose zero
            if (global::hexapod.call(srv))
            {
                ROS_INFO("init pose");
            }
            else
            {
                ROS_ERROR("Failed to call service");
                return 1;
            }
            timerclear(&timev_cur);
            gettimeofday(&timev_cur, NULL);
            timersub(&timev_cur, &timev_init, &timev_duration);

            std::cout<< "estimated distance: "<<obs<<std::endl;
            if(obs<0 || obs>2)
            {
                std::cout<<"measurement seems wrong, set to zero"<<std::endl;
                obs=0;
            }

            std::cin.clear();
            std::cout<<"transfert ok? :"<<std::endl;
            std::cin>> ok;
            std::cin.clear();
            std::cin.ignore( std::numeric_limits<std::streamsize>::max(), '\n' );
        }
        while (!ok);

        std::cout<<"action "<<timev_duration.tv_sec+timev_duration.tv_usec/1e6<<"sec"<<std::endl;
        ofile<<"action "<<timev_duration.tv_sec+timev_duration.tv_usec/1e6<<"sec"<<std::endl;

        gettimeofday(&global::timev_selection, NULL);
        return obs;
#else

        float covered_distance = 0.0;

        size_t controller_index = Params::archiveparams::archive.at(key).controller_index;
        //ea.pop()[controller_index]->fit().eval(*ea.pop()[controller_index]);
        //covered_distance = ea.pop()[controller_index]->fit().value();

        global_popstat::mapstat.show(std::cerr, controller_index);
        covered_distance = global_popstat::mapstat.archive()[controller_index]->fit().value();

        if(covered_distance<0 || covered_distance>2.5)
        {
            std::cout<<covered_distance<<" measurement seems wrong, set to zero"<<std::endl;
            return 0;
        }

#ifdef NOISEADDED
        return covered_distance*bayesian_optimization::misc::gaussian_rand(0.95,0.1); // multiplicative gaussian noise with mean 0.95 (instead of 1, as KINECT always underestimates distance) and std. dev. 1
#else
        return covered_distance;
#endif
#endif

    }
};




std::map<std::vector<float>,Params::archiveparams::elem_archive,Params::archiveparams::classcomp>  load_archive(std::string archive_name)
{    
    std::map<std::vector<float>,Params::archiveparams::elem_archive,Params::archiveparams::classcomp> archive;
    std::ifstream monFlux(archive_name.c_str());  //Ouverture d'un fichier en lecture
    if(monFlux)
    {
        while(!monFlux.eof())
        {
            Params::archiveparams::elem_archive elem;
            std::vector<float> candidate(6);

            for(int i =0;i<(Params::boptimizer::dim + 1 + 1);i++)
            {
                if(monFlux.eof())
                    break;
                float data;
                monFlux>>data;

                if(i<=(Params::boptimizer::dim-1))
                {
                    candidate[i]=data;
                    elem.duty_cycle.push_back(data);
                }

                if(i==Params::boptimizer::dim)
                    elem.fit=data;

                if(i>=(Params::boptimizer::dim+1))
                    elem.controller_index = data;
            }
            archive[candidate]=elem;
        }
    }
    else
    {
        std::cout << "ERREUR: Impossible d'ouvrir le fichier en lecture." << std::endl;
        return archive;
    }
    std::cout<<archive.size()<<" elements loaded"<<std::endl;
    return archive;
}


void lecture() // evaluate all the individuals in the grid
{
#ifdef ROBOT
    // input index to individual of population. If individual exists, we run it on the robot
#else

    std::ofstream workingFile("freqofarchive15000_0.dat");

    for (Params::archiveparams::archive_t::const_iterator it=Params::archiveparams::archive.begin(); it!=Params::archiveparams::archive.end(); ++it)
    {
        global::ind_index = it->second.controller_index;
        global_popstat::mapstat.show(std::cerr, global::ind_index);

        // std::vector<std::vector<float> > angles_forfft = global_popstat::mapstat.archive()[global::ind_index]->fit().angles_forfft;


//        std::ostringstream oss; oss << global::ind_index;
//        std::ofstream workingFile(("cppnoutput_ind" + oss.str() + std::string(".dat")).c_str());
//        if (workingFile)
//            for (int o =0; o < angles_forfft.size(); o++)
//            {
//                for (int p =0; p < angles_forfft[o].size(); p++)
//                    workingFile << angles_forfft[o][p] << " ";

//                workingFile << std::endl;
//            }


        if (workingFile)
            workingFile << global_popstat::mapstat.archive()[global::ind_index]->fit().desc()[0] << " " << global_popstat::mapstat.archive()[global::ind_index]->fit().desc()[1] << " " << global_popstat::mapstat.archive()[global::ind_index]->fit().desc()[2] << " " << global_popstat::mapstat.archive()[global::ind_index]->fit().desc()[3] << " " << global_popstat::mapstat.archive()[global::ind_index]->fit().desc()[4] << " " << global_popstat::mapstat.archive()[global::ind_index]->fit().desc()[5] << "   " << global_popstat::mapstat.archive()[global::ind_index]->fit().servo_frequencies_max << "    " << global::ind_index << std::endl;
    }

#endif
}




Params::archiveparams::archive_t Params::archiveparams::archive;
float Params::kf_maternfivehalfs::l;
int Params::maxiterations::n_iterations;
float Params::ucb::alpha;
int Params::panne::number;


int main(int argc,char** argv)
{       
    if(argc<2)
        return -1;


    using namespace nn;
    typedef BomeanFit<Params> fit_t;
    typedef phen::Parameters<gen::EvoFloat<1, Params::cppn>, fit::FitDummy<>, Params::cppn> weight_t;
    typedef gen::Hnn<weight_t, Params, Params::cppn> gen_t;
    typedef phen::Hnn<gen_t, fit_t, Params> phen_t;
    typedef boost::fusion::vector< sferes::stat::Map<phen_t, Params> > stat_t;
    typedef modif::Dummy<> modifier_t;
    typedef eval::Eval<Params> eval_t;
    typedef ea::MapElite<phen_t, eval_t, stat_t, modifier_t, Params> ea_t;
    ea_t ea;

    ea.load(argv[2]);
    typedef sferes::stat::Map<phen_t, Params> t_mapstat;
    const stat_t MyMapStat = dynamic_cast<const stat_t&>(ea.stat());
    global_popstat::mapstat = boost::fusion::at_c<0>(MyMapStat);



    Params::archiveparams::archive=load_archive(argv[1]);


    /*global_popstat::supg_ctrl = global_popstat::mapstat.archive()[7314]->gen();
    global_popstat::supg_ctrl.init();
    std::cerr << " global_popstat::mapstat.archive()[7314]->gen().returncppn().get_nb_neurons()  " << global_popstat::supg_ctrl.returncppn().get_nb_neurons() << std::endl;
    std::cerr << " global_popstat::mapstat.archive()[7314]->gen().returncppn().get_nb_connections()  " << global_popstat::supg_ctrl.returncppn().get_nb_connections() << std::endl;
    std::ofstream ofs(std::string("graph.dot").c_str());
    global_popstat::supg_ctrl.write_cppndot(ofs, global_popstat::supg_ctrl.returncppn());
    global_popstat::supg_ctrl.substrate();
    global_popstat::supg_ctrl.query(boost::make_tuple(0.0, 0.0, 0.0));
    return 0;*/


    if(argc>3)
        Params::kf_maternfivehalfs::l=atof(argv[3]);
    else
        Params::kf_maternfivehalfs::l=0.4;

    Params::ucb::alpha=0.05;

#ifdef NOSTOP
    Params::maxiterations::n_iterations=150;
#else
    Params::maxiterations::n_iterations=150;
#endif


    srand(time(NULL));

    typedef kernel_functions::KF_MaternFiveHalfs<Params> KF_t;

    typedef inner_optimization::ExhaustiveSearchArchive<Params> InnerOpt_t;

#ifdef NOSTOP
    typedef boost::fusion::vector< stopping_criterion::MaxIterations<Params> > Stop_t;
#else
    typedef boost::fusion::vector< stopping_criterion::MaxIterations<Params>, stopping_criterion::MaxPredictedValue<Params> > Stop_t;
#endif

    typedef mean_functions::MeanArchive_Map<Params> Mean_t;


#ifdef ROBOT
    typedef boost::fusion::vector< stat::Acquisitions<Params>,stat::StatArchive<Params>,stat::StatTransferts<Params>,stat::StatObservations<Params> > Stat_t;
#else
    //typedef boost::fusion::vector<  > BOStat_t;
    typedef boost::fusion::vector< bayesian_optimization::stat::Acquisitions<Params>, bayesian_optimization::stat::StatTransferts<Params>, bayesian_optimization::stat::StatObservations<Params> > BOStat_t;
#endif

    typedef acquisition_functions::UCB<Params,Mean_t,KF_t> Acqui_t;

#ifdef ROBOT
    init_ros_node(argc ,argv);
    hexa_control::Transfert srv;
    srv.request.duration=0; // init
    if (global::hexapod.call(srv))
    {
        ROS_INFO("executed");
    }
    else
    {
        ROS_ERROR("Failed to call service");
        return 1;
    }
#else
    dInitODE();
    std::vector<int> brokenleg;
    if(argc>4)
    {
        for(int i=4;i<argc;i++)
        {
            brokenleg.push_back(atoi(argv[i]));
        }
    }
    init_simu(argc, argv,true,brokenleg);
#endif


    /*lecture(); // evaluate all the individuals in the grid;
    return 0;*/


    BOptimizer<Params,KF_t,Acqui_t,Mean_t,InnerOpt_t,BOStat_t, Stop_t > opt;
    global::res_dir=opt.res_dir();


    Eigen::VectorXf result(1);
    float val=opt.optimize(fit_eval_map<Params>(),result);


    std::ofstream oofile((opt.res_dir()+std::string("/panne.dat")).c_str(), std::ios_base::app);
    for(int i=0;i<global::brokenLegs.size();i++)
        oofile<<global::brokenLegs[i]<<" ";

    if(argc>1)
    {
        std::ofstream oofile1((opt.res_dir()+std::string("/archive_name.dat")).c_str(), std::ios_base::app);
        oofile1 << argv[1];
    }


#ifdef ROBOT
    std::ofstream ofile((std::string("result_")+argv[2]+".dat").c_str(), std::ios_base::app);

    ofile<<argv[1]<<" "<<argv[2]<<"  ";
    for(int i =3;i<argc;i++)
        ofile<<argv[i];
    ofile<<" "<<val<<" "<<opt.iteration()<<std::endl;
#endif

    std::cout<<val<<" res  "<<result.transpose()<<std::endl;

#ifdef ROBOT
    srv.request.duration=-5; // relax
    if (global::hexapod.call(srv))
    {
        ROS_INFO("executed");
    }
    else
    {
        ROS_ERROR("Failed to call service");
        return 1;
    }


#else

    global::global_robot.reset();
    global::global_env.reset();

    dCloseODE();
#endif  

    std::cout <<"fin"<<std::endl;


    return 0;

}
