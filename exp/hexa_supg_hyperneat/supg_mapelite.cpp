
//#define EIGEN_USE_NEW_STDVECTOR
//#include <Eigen/StdVector>
#include <Eigen/Core>
//#include <Eigen/Array>


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
#include <sferes/ea/nsga2.hpp>
#include <sferes/fit/fitness.hpp>
#include <sferes/modif/dummy.hpp>
#include <sferes/eval/parallel.hpp>
#include <sferes/eval/eval.hpp>
#include <sferes/stat/best_fit.hpp>

#include <modules/nn2/phen_dnn.hpp>

//#include "stat_progress_archive.hpp"
#include "behavior.hpp"
#include "gen_hnn.hpp"
#include "phen_hnn.hpp"

//#include "pareto_front_constrsort.hpp"
#include <modules/map_elite/map_elite.hpp>
#include <modules/map_elite/fit_map.hpp>
#include <modules/map_elite/stat_map.hpp>

#define NO_MPI

//#include "ode/box.hh"

#ifdef GRAPHIC
#define NO_PARALLEL
#include "renderer/osg_visitor.hh"
#endif

#include "hexapod.hh"
#include "simu.hpp"
#include <boost/fusion/sequence/intrinsic/at.hpp>

using namespace sferes;
using namespace boost::assign;
using namespace sferes::gen::dnn;
using namespace sferes::gen::evo_float;


//#define AMPLITUDE_BD // using the amplitude behavior descriptor


#define MutMultiplier (1.0f / 10.0f)   //  mutation rates reduced by one order of magnitude

struct Params
{
    struct ea
    {
#ifdef AMPLITUDE_BD
      SFERES_CONST size_t behav_dim = 3;
      SFERES_ARRAY(size_t, behav_shape, 20, 20, 20);
#else
      SFERES_CONST size_t behav_dim = 6;
      SFERES_ARRAY(size_t, behav_shape, 5, 5, 5, 5, 5, 5);
#endif
    };
    struct pop //Parameters of the population
    {
        // number of initial random points
        SFERES_CONST size_t init_size = 200; //200; //Using the same parameters as Antoine to compare results //1000

        // size of a batch - evaluations to be run in parallel. could be about 64 (number of cores on cluster).
        SFERES_CONST size_t size = 200; //200  was Using the same parameters as Antoine to compare results //64
      SFERES_CONST size_t nb_gen = 25001; //200001; //100001; // number of evaluations is nb_gen * size of batch

        static constexpr int dump_period = 1000; //50  //logs are written every dump_period of 5001 generations
        static constexpr int dump_period_logs = 50; //50  //logs are written every dump_period of 5001 generations
        static constexpr int initial_aleat = 1; //initial population size at first generation is scaled by initial_aleat
    };
    struct cppn
    {
        // params of the CPPN
        struct sampled
        {
            SFERES_ARRAY(float, values, 0, 1, 2, 3, 4); //sine; sigmoid; gaussian; linear; copy/tanh (depending on if supg is constrained/not-constrained)
	  static constexpr float mutation_rate = 0.01f * MutMultiplier;;
            static constexpr float cross_rate = 0.0;
            static constexpr bool ordered = false;
        };
        struct evo_float
        {
            static constexpr float mutation_rate = 0.1f ;//per conn mut rate
	    static constexpr float eta_m = 10.0f / MutMultiplier;;  // perturbation of the order O(1/eta_m)

            static constexpr float cross_rate = 0.0f; //we don't use this
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
            //x1 and y1, the position of the supg; time since last trigger event, error in heading orientation f/b, + bias input
            static constexpr size_t nb_inputs = 5;
#endif

            //outputs: supg output and offset to timer and duty factor
            static constexpr size_t nb_outputs = 3;
            static constexpr init_t init = ff; //feedforward nn
	  static constexpr float m_rate_add_conn = 0.05f* MutMultiplier;; //0.1f
	  static constexpr float m_rate_del_conn = 0.1f* MutMultiplier;;
	  static constexpr float m_rate_change_conn = 0.0f* MutMultiplier;; //move conn to new neurons
	  static constexpr float m_rate_add_neuron = 0.05f* MutMultiplier;; //0.2f
	  static constexpr float m_rate_del_neuron = 0.01f* MutMultiplier;;

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

///variables globales------------
///global variables

namespace global
{//Pointer to robot Hexapod and the ode environment. Incase of transferability, there would be one more environment and one more robot (physical robot or detailed simulation of it) here
boost::shared_ptr<robot::Hexapod> robot;
boost::shared_ptr<ode::Environment_hexa> env;

std::vector<int> brokenLegs; // broken legs for the global::robot and not global::robot_dmg
//std::vector<int> brokenLegs(2,1); // middle leg and back leg of other side is broken
//std::vector<int> brokenLegs(1,1); // middle leg is broken
//std::vector<int> brokenLegs(1,5); // front-left leg is broken
//std::vector<int> brokenLegs(1,1);
};
///---------------------------

void init_simu(int argc ,char** argv, bool master)
{
    //global::brokenLegs[1] = 3;
    global::env = boost::shared_ptr<ode::Environment_hexa>(new ode::Environment_hexa());

    //passed environment robot is in, position and vector of broken legs
    global::robot =
            boost::shared_ptr<robot::Hexapod>(
                new robot::Hexapod(*global::env, Eigen::Vector3d(0, 0, 0.1), global::brokenLegs));


    float step = 0.001; //step size of robot, being dropped down from 100cm - to prevent any inital legs in floor artifacts

    // low gravity to slow things down (eq. smaller timestep?).
    // i beleive so, v = u + a*t
    global::env->set_gravity(0, 0, -9.81); //was 15m/s^2 but side effects seen at the second stabilization,after the evaluation gait is completed
    bool stabilized = false;
    int stab = 0;
    for (size_t s = 0; s < 1000 && !stabilized; ++s)
    {
        Eigen::Vector3d prev_pos = global::robot->pos();
        global::robot->next_step(step);
        global::env->next_step(step);

        //!Distance between last two positions. robot->pos() returns pos of center of main body
        if ((global::robot->pos() - prev_pos).norm() < 1e-4)//-5
            stab++;
        else
            stab = 0;
        if (stab > 100)
            stabilized = true;
    }
    assert(stabilized);
    global::env->set_gravity(0, 0, -9.81); //returning gravity to normal

    global::env->resetGRF();
}



//SFERES_FITNESS(FitSpace, sferes::fit::Fitness)
FIT_MAP(FitGrid)
{
    public:
    float servo_frequencies_max;
    float dist;

    template<typename Indiv>
    void eval(Indiv& indiv)
    {
        /*//_objs instead of _value for Multiobjective optimizations.
        //three objectives: performance - displacement, trajectory direction angle (wrt y-axis) and behavior diversity.
        this->_objs.resize(1);
        std::fill(this->_objs.begin(), this->_objs.end(), 0);
        _dead=false; //the dead robot is awarded a huge -ve performance value
        _eval(indiv, write_objs);*/

        this->_objs.resize(1);
        std::fill(this->_objs.begin(), this->_objs.end(), 0);

        _dead=false; //the dead robot is awarded a huge -ve performance value
        _eval(indiv);
    }

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

    //sum of vector
    std::vector <float> sumsof2Dvector(std::vector <std::vector <float> >& v1)
    {
      std::vector <float> vecsums(v1[0].size(),0);

      for (int o =0; o < v1.size(); o++)
	for (int p =0; p < v1[o].size(); p++)
	  vecsums[p] += v1[o][p];

      return vecsums;
    }


    std::vector<Eigen::Vector3d> get_traj(){return _traj;}
    void set_performance(float p){this->_value=p;} //Performance is stored in the first objective
    Behavior& behavior(){return this->_behavior;}
    const Behavior& behavior() const{return this->_behavior;}
    float direction(){return this-> _direction;}
    float covered_distance(){return this-> _covered_distance;}
    float arrival_angle(){return this->_arrival_angle;}
    bool dead(){return this->_dead;}

    size_t cppn_nodes() const { return _cppn_nodes; }
    size_t cppn_conns() const { return _cppn_conns; }
    size_t nn_nodes() const { return _nn_nodes; }
    size_t nn_conns() const { return _nn_conns; }

    std::vector<std::vector<float> > legs_features;
    std::vector<float>  features;

    protected:
    Behavior _behavior;
    float _arrival_angle;
    float _direction;
    float _covered_distance;
    bool _dead;
    std::vector<Eigen::Vector3d> _traj;

    size_t _cppn_nodes;
    size_t _cppn_conns;
    size_t _nn_nodes;
    size_t _nn_conns;

    template<typename Indiv>
    void _eval(Indiv& indiv)
    {
        indiv.gen().init();

        _cppn_nodes = indiv.gen().returncppn().get_nb_neurons();
        _cppn_conns = indiv.gen().returncppn().get_nb_connections();

        if (this->mode() == sferes::fit::mode::view) //is used to  view individuals in pareto front (their ids. can be got from pareto.data)
        {
            std::ofstream ofs(std::string("graph.dot").c_str());
            indiv.gen().write_cppndot(ofs, indiv.gen().returncppn());

            std::ofstream ofs_cppn(std::string("cppn.dat").c_str());
            ofs_cppn << indiv.gen().returncppn().get_nb_neurons() << " " << indiv.gen().returncppn().get_nb_connections() <<  " " << indiv.gen().returncppn().get_nb_inputs() <<  " " << indiv.gen().returncppn().get_nb_outputs() << std::endl;
        }



#if defined GRND_OBSTACLES || defined GRND_OBSTACLES_EVO
	std::vector <std::vector <float> > noobstacles;
        float floorangle=0.0;
        Simu <typename Indiv::gen_t> simu(indiv.gen(), global::robot, global::brokenLegs, 5.0, noobstacles, floorangle);
#else
        Simu <typename Indiv::gen_t> simu(indiv.gen(), global::robot, global::brokenLegs, 5.0, 0.0);
#endif

        this->dist=simu.covered_distance();
        this->_covered_distance= simu.covered_distance();

        this->_direction = simu.direction();
        this->_arrival_angle=simu.arrival_angle();//orientation of robots final pos wrt y axis

        this->servo_frequencies_max = simu.servo_frequencies_max;

        if(this->_covered_distance < -1000.0)
        {
            _dead=true;
            _behavior.position.resize(2);
            _behavior.position[0]=0.0;
            _behavior.position[1]=0.0;
            _behavior.performance = -1000.0;
            _behavior.arrivalangle = -1000.0;
            _behavior.direction = -1000.0;

            this->_value = -1000.0f;

#ifdef AMPLITUDE_BD
	    std::vector<float> data = {0.0f, 0.0f, 0.0f};
            this->set_desc(data);
#else
	    std::vector<float> data = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
            this->set_desc(data);
#endif
        }
        else
        {
            legs_features.clear();
            legs_features.push_back(simu.get_contact(0));
            legs_features.push_back(simu.get_contact(1));
            legs_features.push_back(simu.get_contact(2));
            legs_features.push_back(simu.get_contact(3));
            legs_features.push_back(simu.get_contact(4));
            legs_features.push_back(simu.get_contact(5));

            _behavior.features.clear();
            features.clear();
            for(int i=0;i<legs_features.size();i++)
                for(int j=0;j<legs_features[i].size();j++)
                {
                    features.push_back(legs_features[i][j]);
                    _behavior.features.push_back(legs_features[i][j]);
                }


#ifdef AMPLITUDE_BD
            float MAX_AMPLITUDE = 0.20f; //max range of 20 cm
	    std::vector<float> data = {std::min(simu.amp_x / MAX_AMPLITUDE, 1.0f),
                                       std::min(simu.amp_y / MAX_AMPLITUDE, 1.0f),
                                       std::min(simu.amp_z / MAX_AMPLITUDE, 1.0f)};
#else
            /*==========Duty factor behavior descriptor ============ */
	    /*std::vector<float> data = {(float)round((sumofvector(legs_features[0]) / (float)legs_features[0].size()) * 100.0f) / 100.0f,
	    			       (float)round((sumofvector(legs_features[1]) / (float)legs_features[1].size()) * 100.0f) / 100.0f,
	    			       (float)round((sumofvector(legs_features[2]) / (float)legs_features[2].size()) * 100.0f) / 100.0f,
	    			       (float)round((sumofvector(legs_features[3]) / (float)legs_features[3].size()) * 100.0f) / 100.0f,
	    			       (float)round((sumofvector(legs_features[4]) / (float)legs_features[4].size()) * 100.0f) / 100.0f,
	    			       (float)round((sumofvector(legs_features[5]) / (float)legs_features[5].size()) * 100.0f) / 100.0f}; */
            /*========================================================*/


            /*==========Orientation behavior descriptor ============ */
            //            using main body orientation behavior descriptor
            /*            float perc_threshold = 0.5f; // > +perc_threshold is considered +ve // < -perc_threshold is considered -ve
			  std::vector<float> data = simu.get_orientation_bd(perc_threshold);*/
            /*========================================================*/



            /*==========Relative GRF behavior descriptor ============ */
	    std::vector <float> grf_eachleg = sumsof2Dvector(simu.grf_z);
            assert(grf_eachleg.size() == 6);

            float total_force = sumofvector(grf_eachleg);
            float total_force_y = sumofvector(sumsof2Dvector(simu.grf_y));
            float total_force_x = sumofvector(sumsof2Dvector(simu.grf_x));

            if(isnan(total_force) || isnan(total_force_x) || isnan(total_force_y) || total_force > 1.0e+5 || std::min(grf_eachleg[0],grf_eachleg[1]) < -0.1 || std::min(grf_eachleg[2],grf_eachleg[3]) < -0.1 || std::min(grf_eachleg[4],grf_eachleg[5]) < -0.1) 
	    // the z component of the GRF sometimes takes very small negative values. if the negative value is too large the leg maybe penetrating the ground. so we kill the individual. // in rare cases we get a NaN for the x and y component of the GRF. We catch it here
	      {
		_dead=true;
		_behavior.position.resize(2);
		_behavior.position[0]=0.0;
		_behavior.position[1]=0.0;
		_behavior.performance = -1000.0;
		_behavior.arrivalangle = -1000.0;
		_behavior.direction = -1000.0;

		this->_value = -1000.0f;

		std::vector<float> data = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
		this->set_desc(data);
		return;
	      }

             float rel_grd_0 = std::max(grf_eachleg[0] / total_force, 0.0f); // the z component of the GRF sometimes takes very small negative values (< 0.01)
             float rel_grd_1 = std::max(grf_eachleg[1] / total_force, 0.0f);
             float rel_grd_2 = std::max(grf_eachleg[2] / total_force, 0.0f);
             float rel_grd_3 = std::max(grf_eachleg[3] / total_force, 0.0f);
             float rel_grd_4 = std::max(grf_eachleg[4] / total_force, 0.0f);
             float rel_grd_5 = std::max(grf_eachleg[5] / total_force, 0.0f);


	     std::vector<float> data = {rel_grd_0, rel_grd_1, rel_grd_2, rel_grd_3, rel_grd_4, rel_grd_5};
            /*========================================================*/
#endif

            this->set_desc(data);

            _behavior.performance = simu.covered_distance();
            //this->_value   = _behavior.performance;

	    if (floor(this->servo_frequencies_max) >= 1.99999f)
	      this->_value   = _behavior.performance / floor(this->servo_frequencies_max);
	      else 
	    this->_value   = _behavior.performance;

            _behavior.direction = round(this->_direction * 100) / 100.0f; //two decimal places
        }

        if(this->mode() == sferes::fit::mode::view)
        {
	  std::cout << " _behavior.performance " << _behavior.performance << std::endl;
	  std::cout << " _direction " << _behavior.direction << std::endl;
	  std::cout << " frequency " << simu.servo_frequencies_max << std::endl;

	  std::cout << std::endl << " Descriptor:  ";
	  for(size_t bd_index=0;bd_index<6;++bd_index)
	    std::cout  << this->desc()[bd_index] << " ";
	  std::cout  << std::endl;

	  static std::ofstream ofs(std::string("performance_metrics.dat").c_str());
	  ofs << _behavior.performance << std::endl;
	  ofs << _behavior.direction << std::endl;
	  ofs.close();
        }
    }
};


int main(int argc, char **argv)
{
    dInitODE();

#ifndef NO_PARALLEL
    typedef eval::Parallel<Params> eval_t;
#else
    typedef eval::Eval<Params> eval_t;
#endif


    using namespace nn;
    typedef FitGrid<Params> fit_t;
    typedef phen::Parameters<gen::EvoFloat<1, Params::cppn>,
            fit::FitDummy<>, Params::cppn> weight_t;
    typedef gen::Hnn<weight_t, Params, Params::cppn> gen_t;
    typedef phen::Hnn<gen_t, fit_t, Params> phen_t;
    typedef boost::fusion::vector< sferes::stat::Map<phen_t, Params> > stat_t;

    typedef modif::Dummy<> modifier_t;


    if(argc >= 25)
    {
    }


    typedef ea::MapElite<phen_t, eval_t, stat_t, modifier_t, Params> ea_t;

    ea_t ea;

    //initilisation of the simulation and the simulated robot
    if(argc>7)
    {
        global::brokenLegs.push_back(atoi(argv[7]));
        std::cout << "Leg damaged: " << atoi(argv[7]) << std::endl;
    }
    init_simu(argc, argv, true);

    std::cout<<"debut run"<<std::endl;
    run_ea(argc, argv, ea);
    std::cout <<"fin run"<<std::endl;

    global::robot.reset();
    global::env.reset();

    dCloseODE();
    std::cout <<"fin"<<std::endl;


    return 0;
}
