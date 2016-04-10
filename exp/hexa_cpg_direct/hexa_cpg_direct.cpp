/*================================================================*/
//#define SEEDED_ENTIREPOP // The entire population is seeded as it is
/*================================================================*/
//#define SEEDED_NSGA // Seed population with mutants of best individual

#ifdef SEEDED_NSGA
    #include <modules/seeded_nsga2/seeded_nsga2.hpp>
    #include <modules/nn2/trait.hpp>
#endif
/*================================================================*/

#define EIGEN_USE_NEW_STDVECTOR
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
#include "pareto_front_constrsort.hpp"

#include <modules/nn2/phen_dnn.hpp>

#include "behavior.hpp"

#define NO_MPI

#include "ode/box.hh"

#ifdef GRAPHIC
#define NO_PARALLEL
#include "renderer/osg_visitor.hh"
#endif


#include "hexapod.hh"
#include "simu.hpp"
#include <boost/fusion/sequence/intrinsic/at.hpp>


/*#if MutationRate==Class1
#define MutMultiplier 4.0 // high mutation rates - an order of magnitude higher
#elif MutationRate==Class2
#define MutMultiplier 1.0/4.0 // low mutation rates - an order of magnitude lower
#else
#define MutMultiplier 1.0   // standard mutation rates
#endif*/

#define MutMultiplier 1.0

using namespace sferes;
using namespace boost::assign;
using namespace sferes::gen::evo_float;

struct Params
{
#ifdef SEEDED_NSGA
    struct seed
    {
      static constexpr unsigned stat_num = 0; //number of statistics; 0 -> just one stat, the pareto front
      static unsigned gen_num;            //id of the individual in the pareto front, to be loaded
      static std::string file_name;       //generation file name
    };
#endif

    struct pop //Parameters of the population
    {
        static constexpr unsigned size = 100; //population size
#ifdef SEEDED_NSGA
        static constexpr unsigned nb_gen = 10001; //10001; //total number of generations of evolution
        static constexpr int dump_period = 10;    //logs are written every dump_period generations
#else

#ifdef SEEDED_ENTIREPOP
        static constexpr unsigned nb_gen = 8000 + 10001;
#else
        static constexpr unsigned nb_gen = 10001; //10001;
#endif
        static constexpr int dump_period = 50;  //logs are written every dump_period generations
#endif

        static constexpr int initial_aleat = 1; //initial population size at first generation is scaled by initial_aleat
    };

    struct evo_float //Parameters for the "evo_float" genotype. gene values have higher degree of precision
    {
        static constexpr float mutation_rate = 0.1f * MutMultiplier; //per gene mut rate
        static constexpr float eta_m = 10.0f / MutMultiplier; // perturbation of the order O(1/eta_m)

        static constexpr float cross_rate = 0.1f; //we don't use this
        static constexpr mutation_t mutation_type = polynomial;
        static constexpr cross_over_t cross_over_type = sbx;
        static constexpr float eta_c = 10.0f; // A large value ef eta gives a higher probablitity for creating a `near-parent' solutions and a small value allows distant solutions to be selected as offspring.
    };

    struct parameters //!Min. and max. parameters of genes represented with EvoFloat
    {
        static constexpr float min = -1.0f;
        static constexpr float max =  1.0f;
    };
};

#ifdef SEEDED_NSGA
    unsigned Params::seed::gen_num = 0;
    std::string Params::seed::file_name = "test.sferes";
#endif

///variables globales------------
///global variables

namespace global
{//Pointer to robot Hexapod and the ode environment. Incase of transferability, there would be one more environment and one more robot (physical robot or detailed simulation of it) here
#ifndef GRND_OBSTACLES_EVO
boost::shared_ptr<robot::Hexapod> robot;
boost::shared_ptr<ode::Environment_hexa> env;
#else
boost::shared_ptr<robot::Hexapod> robot1, robot2, robot3;
boost::shared_ptr<ode::Environment_hexa> env1, env2, env3;
#endif

std::vector<int> brokenLegs; // broken legs for the global::robot and not global::robot_dmg
//std::vector<int> brokenLegs(2,1); // middle leg and back leg of other side is broken
//std::vector<int> brokenLegs(1,1); // middle leg is broken


#ifdef SEEDED_ENTIREPOP
    unsigned gen=0;
    unsigned dmgatgen = 8000;

    //Pointer to another robot Hexapod (DAMAGED) and the ode environment, to continue evolution when the ROBOT GET DAMAGED.
    boost::shared_ptr<robot::Hexapod> robot_dmg;
    boost::shared_ptr<ode::Environment_hexa> env_dmg;

    std::vector<int> brokenLegs_dmg(1,1); // middle leg of damaged robot is broken // (global::brokenLegs_dmg used with global::robot_dmg to continue evolution with damaged robot);
#endif

#if defined GRND_OBSTACLES || defined GRND_OBSTACLES_EVO
    std::vector < std::vector<float> > obstacle_pos_rad;
    std::vector < std::vector<float> > obstacle_pos_rad1, obstacle_pos_rad2, obstacle_pos_rad3;
    size_t numberobstacles;
#endif
    float floorangle;
};
///---------------------------

void init_simu(int argc ,char** argv, bool master)
{
#ifdef GRND_OBSTACLES
    time_t t = time(0) + ::getpid(); srand(t); //std::cout<<"seed: " << t << std::endl;
    std::ofstream workingFile("seed.dat");
    workingFile << t;

    for(size_t i=0; i<global::numberobstacles;++i)
    {
        global::obstacle_pos_rad.push_back(std::vector<float>(4, 0.0f));

        global::obstacle_pos_rad[i][0] = misc::rand<float>() * 1.2 - 0.6;  //x
        global::obstacle_pos_rad[i][1] = misc::rand<float>() * 1.5 - 0.5;  //y
        global::obstacle_pos_rad[i][3] = misc::rand<float>() * (0.06-0.01) + 0.01; // radius

        //if(global::obstacle_pos_rad[i][3] < 0.05f)
            global::obstacle_pos_rad[i][2] = 0.0f; // misc::rand<float>(0.10 - global::obstacle_pos_rad[i][3], 0.0); //z
        /*else
            global::obstacle_pos_rad[i][2] = misc::rand<float>(0.01 - global::obstacle_pos_rad[i][3], 0.05 - global::obstacle_pos_rad[i][3]); //z*/
    }

    global::floorangle = 0.0;
    global::env = boost::shared_ptr<ode::Environment_hexa>(new ode::Environment_hexa(global::floorangle, global::obstacle_pos_rad));
#elif defined GRND_SLOPES
    global::floorangle = global::floorangle*3.14/180.0;
    global::env = boost::shared_ptr<ode::Environment_hexa>(new ode::Environment_hexa(global::floorangle));
#elif defined GRND_OBSTACLES_EVO
    srand(111);
    global::numberobstacles = 200;
    for(size_t i=0; i<global::numberobstacles;++i)
    {
        global::obstacle_pos_rad1.push_back(std::vector<float>(4, 0.0f));
        global::obstacle_pos_rad1[i][0] = misc::rand<float>() * 1.2 - 0.6;  //x
        global::obstacle_pos_rad1[i][1] = misc::rand<float>() * 1.5 - 0.5;  //y
        global::obstacle_pos_rad1[i][3] = misc::rand<float>() * (0.06-0.01) + 0.01; // radius
        global::obstacle_pos_rad1[i][2] = 0.0f; // misc::rand<float>(0.10 - global::obstacle_pos_rad[i][3], 0.0); //z
    }
    global::floorangle = 0.0;
    global::env1 = boost::shared_ptr<ode::Environment_hexa>(new ode::Environment_hexa(global::floorangle, global::obstacle_pos_rad1));


    for(size_t i=0; i<global::numberobstacles;++i)
    {
        global::obstacle_pos_rad2.push_back(std::vector<float>(4, 0.0f));
        global::obstacle_pos_rad2[i][0] = misc::rand<float>() * 1.2 - 0.6;  //x
        global::obstacle_pos_rad2[i][1] = misc::rand<float>() * 1.5 - 0.5;  //y
        global::obstacle_pos_rad2[i][3] = misc::rand<float>() * (0.06-0.01) + 0.01; // radius
        global::obstacle_pos_rad2[i][2] = 0.0f; // misc::rand<float>(0.10 - global::obstacle_pos_rad[i][3], 0.0); //z
    }
    global::env2 = boost::shared_ptr<ode::Environment_hexa>(new ode::Environment_hexa(global::floorangle, global::obstacle_pos_rad2));


    for(size_t i=0; i<global::numberobstacles;++i)
    {
        global::obstacle_pos_rad3.push_back(std::vector<float>(4, 0.0f));
        global::obstacle_pos_rad3[i][0] = misc::rand<float>() * 1.2 - 0.6;  //x
        global::obstacle_pos_rad3[i][1] = misc::rand<float>() * 1.5 - 0.5;  //y
        global::obstacle_pos_rad3[i][3] = misc::rand<float>() * (0.06-0.01) + 0.01; // radius
        global::obstacle_pos_rad3[i][2] = 0.0f; // misc::rand<float>(0.10 - global::obstacle_pos_rad[i][3], 0.0); //z
    }
    global::env3 = boost::shared_ptr<ode::Environment_hexa>(new ode::Environment_hexa(global::floorangle, global::obstacle_pos_rad3));
#else
    global::floorangle = 0.0;
    global::env = boost::shared_ptr<ode::Environment_hexa>(new ode::Environment_hexa(global::floorangle));
#endif

    //passed environment robot is in, position and vector of broken legs
#ifdef GRND_SLOPES
    global::robot =
            boost::shared_ptr<robot::Hexapod>(
                new robot::Hexapod(*global::env, Eigen::Vector3d(0, 0, 0.5), global::brokenLegs));
#else
#ifndef GRND_OBSTACLES_EVO
    global::robot =
            boost::shared_ptr<robot::Hexapod>(
                new robot::Hexapod(*global::env, Eigen::Vector3d(0, 0, 0.1), global::brokenLegs));
#else
    global::robot1 =
            boost::shared_ptr<robot::Hexapod>(
                new robot::Hexapod(*global::env1, Eigen::Vector3d(0, 0, 0.1), global::brokenLegs));
    global::robot2 =
            boost::shared_ptr<robot::Hexapod>(
                new robot::Hexapod(*global::env2, Eigen::Vector3d(0, 0, 0.1), global::brokenLegs));
    global::robot3 =
            boost::shared_ptr<robot::Hexapod>(
                new robot::Hexapod(*global::env3, Eigen::Vector3d(0, 0, 0.1), global::brokenLegs));
#endif
#endif

#ifndef GRND_OBSTACLES_EVO
    float step = 0.001; //step size of robot, being dropped down from 100cm - to prevent any inital legs in floor artifacts
    global::env->set_gravity(0, 0, -9.81); //was 15m/s^2 but side effects seen at the second stabilization,after the evaluation gait is completed
    bool stabilized = false;
    int stab = 0;
    for (size_t s = 0; s < 1000 && !stabilized; ++s)
    {
        Eigen::Vector3d prev_pos = global::robot->pos();
        global::robot->next_step(step);
        global::env->next_step(step);

        //!Distance between last two positions. robot->pos() returns pos of center of main body
        if ((global::robot->pos() - prev_pos).norm() < 1e-5)
            stab++;
        else
            stab = 0;
        if (stab > 100)
            stabilized = true;
    }
#ifndef GRND_SLOPES
    assert(stabilized);
#endif
    global::env->set_gravity(0, 0, -9.81); //returning gravity to normal
#else

    float step = 0.001; //step size of robot, being dropped down from 100cm - to prevent any inital legs in floor artifacts
    global::env1->set_gravity(0, 0, -9.81); //was 15m/s^2 but side effects seen at the second stabilization,after the evaluation gait is completed
    bool stabilized = false;
    int stab = 0;
    for (size_t s = 0; s < 1000 && !stabilized; ++s)
    {
        Eigen::Vector3d prev_pos = global::robot1->pos();
        global::robot1->next_step(step);
        global::env1->next_step(step);
        if ((global::robot1->pos() - prev_pos).norm() < 1e-5)
            stab++;
        else
            stab = 0;
        if (stab > 100)
            stabilized = true;
    }
    //assert(stabilized);
    global::env1->set_gravity(0, 0, -9.81); //returning gravity to normal


    global::env2->set_gravity(0, 0, -9.81); //was 15m/s^2 but side effects seen at the second stabilization,after the evaluation gait is completed
    stabilized = false;
    stab = 0;
    for (size_t s = 0; s < 1000 && !stabilized; ++s)
    {
        Eigen::Vector3d prev_pos = global::robot2->pos();
        global::robot2->next_step(step);
        global::env2->next_step(step);
        if ((global::robot2->pos() - prev_pos).norm() < 1e-5)
            stab++;
        else
            stab = 0;
        if (stab > 100)
            stabilized = true;
    }
    //assert(stabilized);
    global::env2->set_gravity(0, 0, -9.81); //returning gravity to normal


    step = 0.001; //step size of robot, being dropped down from 100cm - to prevent any inital legs in floor artifacts
    global::env3->set_gravity(0, 0, -9.81); //was 15m/s^2 but side effects seen at the second stabilization,after the evaluation gait is completed
    stabilized = false;
    stab = 0;
    for (size_t s = 0; s < 1000 && !stabilized; ++s)
    {
        Eigen::Vector3d prev_pos = global::robot3->pos();
        global::robot3->next_step(step);
        global::env3->next_step(step);
        if ((global::robot3->pos() - prev_pos).norm() < 1e-5)
            stab++;
        else
            stab = 0;
        if (stab > 100)
            stabilized = true;
    }
    //assert(stabilized);
    global::env3->set_gravity(0, 0, -9.81); //returning gravity to normal
#endif

#ifdef SEEDED_ENTIREPOP
    global::env_dmg = boost::shared_ptr<ode::Environment_hexa>(new ode::Environment_hexa());
    global::robot_dmg =
            boost::shared_ptr<robot::Hexapod>(
                new robot::Hexapod(*global::env_dmg, Eigen::Vector3d(0, 0, 0.1), global::brokenLegs_dmg));

    global::env_dmg->set_gravity(0, 0, -9.81); //was 15m/s^2 but side effects seen at the second stabilization,after the evaluation gait is completed
    stabilized = false;
    stab = 0;
    for (size_t s = 0; s < 1000 && !stabilized; ++s)
    {
        Eigen::Vector3d prev_pos = global::robot_dmg->pos();
        global::robot_dmg->next_step(step);
        global::env_dmg->next_step(step);

        //!Distance between last two positions. robot_dmg->pos() returns pos of center of main body
        if ((global::robot_dmg->pos() - prev_pos).norm() < 1e-5)
            stab++;
        else
            stab = 0;
        if (stab > 100)
            stabilized = true;
    }
    assert(stabilized);
    global::env_dmg->set_gravity(0, 0, -9.81); //returning gravity to normal
#endif
}


SFERES_FITNESS(FitSpace, sferes::fit::Fitness)
{
    public:
    template<typename Indiv>

    void eval(Indiv& indiv, bool write_objs = false)
    {
        //_objs instead of _value for Multiobjective optimizations.
        //three objectives: performance - displacement, trajectory direction angle (wrt y-axis) and behavior diversity.
        this->_objs.resize(3);
        std::fill(this->_objs.begin(), this->_objs.end(), 0);
        _dead=false; //the dead robot is awarded a huge -ve performance value
        _eval(indiv, write_objs);
    }

    //!TODO what does serialize do. Used mainly when using MPI, to shared attributes.
    //! Else will reset value to 0. In this case, sharing the fitness object
    //! This is not needed, if you are not using MPI (MPI not supported on the ISIR cluster). The bug with loading certain generations was that you were serializing unintialized variables of the fitness and the _behavior object. If the value was unusual like a NAN, you might have a problem, loading the file
    /*template<class Archive>
    void serialize(Archive & ar, const unsigned int version)
    {
        dbg::trace trace("fit", DBG_HERE);
        //ar & BOOST_SERIALIZATION_NVP(this->_value);
        //ar & BOOST_SERIALIZATION_NVP(this->_objs);

        ar & boost::serialization::make_nvp("_value", this->_value);
        ar & boost::serialization::make_nvp("_objs",  this->_objs);

        //! TODO What are these attributes storing
        ar & BOOST_SERIALIZATION_NVP(legs_features);// temporal binary features of the leg contacts
        ar & BOOST_SERIALIZATION_NVP(features); // temporal binary features of the leg contacts
        ar & BOOST_SERIALIZATION_NVP(disparity); // the inverse of transferability

        //the curvilinear distance, the distance of the perfect curved trajectory passing through the start
        //and end positions, with the center on the x axis
        ar & BOOST_SERIALIZATION_NVP(_dist);

        ar & BOOST_SERIALIZATION_NVP(_behavior);//the behavior structure

        //final orientation of the robot wrt its initial orientation (i.e. y axis)
        ar & BOOST_SERIALIZATION_NVP(_arrival_angle);

        // direction of trajectory (straight line passing through start and end points) wrt y axis (probably)
        ar & BOOST_SERIALIZATION_NVP(_direction);
        ar & BOOST_SERIALIZATION_NVP(_covered_distance);// same as _dist, i think
        ar & BOOST_SERIALIZATION_NVP(_dead);

        ar & BOOST_SERIALIZATION_NVP(_cppn_nodes);
        ar & BOOST_SERIALIZATION_NVP(_cppn_conns);
        ar & BOOST_SERIALIZATION_NVP(_nn_nodes);
        ar & BOOST_SERIALIZATION_NVP(_nn_conns);     
        ar & BOOST_SERIALIZATION_NVP(_rewards);
    }*/

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


    std::vector<Eigen::Vector3d> get_traj(){return _traj;}
    void set_performance(float p){this->_objs[0]=p;} //Performance is stored in the first objective
    Behavior& behavior(){return this->_behavior;}
    const Behavior& behavior() const{return this->_behavior;}
    float direction(){return this-> _direction;}
    float covered_distance(){return this-> _covered_distance;}
    float arrival_angle(){return this->_arrival_angle;}
    bool dead(){return this->_dead;}


    std::vector<std::vector<float> > legs_features;
    std::vector<float>  features;

   protected:
    float _dist;
    Behavior _behavior;
    float _arrival_angle;
    float _direction;
    float _covered_distance;
    bool _dead;
    std::vector<Eigen::Vector3d> _traj;


    template<typename Indiv>
    void _eval(Indiv& indiv, bool write_objs)
    {
        _behavior.controller.clear();

        if (this->mode() == sferes::fit::mode::view) //is used to  view individuals in pareto front (their ids. can be got from pareto.data)
        {
            // Introducing mutations to measure evolvability - introduced in pareto_front_constrsort.hpp - show(...)

            static std::ofstream ofs_mutated(std::string("genome_mutated.dat").c_str());
            for(size_t osc_index = 0; osc_index < indiv.size(); ++ osc_index)
                ofs_mutated << indiv.data(osc_index) << " ";
            ofs_mutated.close();
        }

        assert(indiv.size() == 6 * 2 * 2);
        size_t osc_index = 0;
        for (size_t leg=0; leg < 6;++leg)
            for (size_t oscl=0; oscl < 2;++oscl)
            {
                _behavior.controller.push_back(indiv.data(osc_index++));
                _behavior.controller.push_back(indiv.data(osc_index++));

                if(oscl==1)
                {
                    _behavior.controller.push_back(indiv.data(osc_index-2));
                    _behavior.controller.push_back(indiv.data(osc_index-1));
                }
            }
        assert(_behavior.controller.size() == 6 * 3 * 2); // 3 oscillators on each leg, with the third oscillator having same parameters as the middle one

        //launching the simulation
#ifdef SEEDED_ENTIREPOP
        Simu simu = Simu(_behavior.controller, global::gen < global::dmgatgen ? global::robot : global::robot_dmg, global::gen < global::dmgatgen ? global::brokenLegs : global::brokenLegs_dmg);
#else
#ifdef GRND_OBSTACLES
       Simu simu = Simu(_behavior.controller, global::robot, global::brokenLegs, 5.0, global::obstacle_pos_rad, global::floorangle);
#elif defined GRND_OBSTACLES_EVO
       Simu simu1(_behavior.controller, global::robot1, global::brokenLegs, 5.0, global::obstacle_pos_rad1, global::floorangle);
       Simu simu2(_behavior.controller, global::robot2, global::brokenLegs, 5.0, global::obstacle_pos_rad2, global::floorangle);
       Simu simu3(_behavior.controller, global::robot3, global::brokenLegs, 5.0, global::obstacle_pos_rad3, global::floorangle);
#else
       Simu simu = Simu(_behavior.controller, global::robot, global::brokenLegs, 5.0, global::env->angle);
#endif
#endif

#ifndef GRND_OBSTACLES_EVO
        this->_dist=simu.covered_distance();
        this->_covered_distance= simu.covered_distance();//euclidean dist
        this->_direction = simu.direction();
        this->_arrival_angle=simu.arrival_angle();//orientation of robots final pos wrt y axis

        if(this->_covered_distance<-1000.0)
        {
            _dead=true;
            _behavior.position.resize(2);
            _behavior.position[0]=0.0;
            _behavior.position[1]=0.0;
            _behavior.performance = -1000.0;
            _behavior.arrivalangle = -1000.0;
            _behavior.direction = -1000.0;
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

           /* _behavior.dutycycles.clear();
            _behavior.dutycycles.push_back(sumofvector(legs_features[0])/legs_features[0].size());
            _behavior.dutycycles.push_back(sumofvector(legs_features[1])/legs_features[1].size());
            _behavior.dutycycles.push_back(sumofvector(legs_features[2])/legs_features[2].size());
            _behavior.dutycycles.push_back(sumofvector(legs_features[3])/legs_features[3].size());
            _behavior.dutycycles.push_back(sumofvector(legs_features[4])/legs_features[4].size());
            _behavior.dutycycles.push_back(sumofvector(legs_features[5])/legs_features[5].size());*/

            _behavior.features.clear();
            features.clear();
            for(int i=0;i<legs_features.size();i++)
                for(int j=0;j<legs_features[i].size();j++)
                {
                    features.push_back(legs_features[i][j]);
                    _behavior.features.push_back(legs_features[i][j]);
                }

            _behavior.position = simu.final_pos();
            std::vector <float> goal_pos; goal_pos.resize(2); goal_pos[0] = 0.0; goal_pos[1] = 25.0; // a very far out goal (e.g. at 400 m) would not penalize the variant turning phenotypes enough


#if defined GRND_OBSTACLES || defined ROB_DAMAGES || defined GRND_SLOPES
            _behavior.performance = simu.covered_distance();
            this->_objs[0] = _behavior.performance;
#else
            _behavior.performance = -round(sqrt(
                                             (_behavior.position[0]-goal_pos[0])*(_behavior.position[0]-goal_pos[0]) +
                                             (_behavior.position[1]-goal_pos[1])*(_behavior.position[1]-goal_pos[1]))*100) / 100.0f; //-ve sign so as to maximize this quantity
            this->_objs[0] = _behavior.performance;
#endif

//            _behavior.arrivalangle = round(this->_arrival_angle * 100) / 100.0f; //two decimal places
//            this->_objs[1] = -fabs(_behavior.arrivalangle);

            _behavior.direction = round(this->_direction * 100) / 100.0f; //two decimal places
            this->_objs[1] = -fabs(_behavior.direction);
        }
#else

       this->_dist=(simu1.covered_distance() + simu2.covered_distance() + simu3.covered_distance()) / 3.0;
       this->_covered_distance= this->_dist;//euclidean dist
       this->_direction = (simu1.direction() + simu2.direction() + simu3.direction()) / 3.0;
       this->_arrival_angle = (simu1.arrival_angle() + simu2.arrival_angle() + simu3.arrival_angle()) / 3.0;//orientation of robots final pos wrt y axis


       if(simu1.covered_distance()<-1000.0 || simu2.covered_distance()<-1000.0 || simu3.covered_distance()<-1000.0)
       {
           _dead=true;
           _behavior.position.resize(2);
           _behavior.position[0]=0.0;
           _behavior.position[1]=0.0;
           _behavior.performance = -1000.0;
           _behavior.arrivalangle = -1000.0;
           _behavior.direction = -1000.0;
       }
       else
       {
           legs_features.clear();
           legs_features.push_back(simu1.get_contact(0));
           legs_features.push_back(simu1.get_contact(1));
           legs_features.push_back(simu1.get_contact(2));
           legs_features.push_back(simu1.get_contact(3));
           legs_features.push_back(simu1.get_contact(4));
           legs_features.push_back(simu1.get_contact(5));
           _behavior.features_simu1.clear();
           for(int i=0;i<legs_features.size();i++)
               for(int j=0;j<legs_features[i].size();j++)
                   _behavior.features_simu1.push_back(legs_features[i][j]);

           legs_features.clear();
           legs_features.push_back(simu2.get_contact(0));
           legs_features.push_back(simu2.get_contact(1));
           legs_features.push_back(simu2.get_contact(2));
           legs_features.push_back(simu2.get_contact(3));
           legs_features.push_back(simu2.get_contact(4));
           legs_features.push_back(simu2.get_contact(5));
           _behavior.features_simu2.clear();
           for(int i=0;i<legs_features.size();i++)
               for(int j=0;j<legs_features[i].size();j++)
                   _behavior.features_simu2.push_back(legs_features[i][j]);

           legs_features.clear();
           legs_features.push_back(simu3.get_contact(0));
           legs_features.push_back(simu3.get_contact(1));
           legs_features.push_back(simu3.get_contact(2));
           legs_features.push_back(simu3.get_contact(3));
           legs_features.push_back(simu3.get_contact(4));
           legs_features.push_back(simu3.get_contact(5));
           _behavior.features_simu3.clear();
           for(int i=0;i<legs_features.size();i++)
               for(int j=0;j<legs_features[i].size();j++)
                   _behavior.features_simu3.push_back(legs_features[i][j]);


           _behavior.performance = 0.0;
           std::vector <float> goal_pos; goal_pos.resize(2); goal_pos[0] = 0.0; goal_pos[1] = 25.0; // a very far out goal (e.g. at 400 m) would not penalize the variant turning phenotypes enough
           _behavior.position = simu1.final_pos();
           _behavior.performance += -round(sqrt(
                                            (_behavior.position[0]-goal_pos[0])*(_behavior.position[0]-goal_pos[0]) +
                                            (_behavior.position[1]-goal_pos[1])*(_behavior.position[1]-goal_pos[1]))*100) / 100.0f; //-ve sign so as to maximize this quantity
           _behavior.position = simu2.final_pos();
           _behavior.performance += -round(sqrt(
                                            (_behavior.position[0]-goal_pos[0])*(_behavior.position[0]-goal_pos[0]) +
                                            (_behavior.position[1]-goal_pos[1])*(_behavior.position[1]-goal_pos[1]))*100) / 100.0f; //-ve sign so as to maximize this quantity
           _behavior.position = simu3.final_pos();
           _behavior.performance += -round(sqrt(
                                            (_behavior.position[0]-goal_pos[0])*(_behavior.position[0]-goal_pos[0]) +
                                            (_behavior.position[1]-goal_pos[1])*(_behavior.position[1]-goal_pos[1]))*100) / 100.0f; //-ve sign so as to maximize this quantity
           this->_objs[0] = _behavior.performance / 3.0;


           _behavior.direction = round(this->_direction * 100) / 100.0f; //two decimal places
           this->_objs[1] = -fabs(_behavior.direction);
       }

#endif


        if(this->mode() == sferes::fit::mode::view)
        {
            std::cout << " _behavior.performance " << _behavior.performance << std::endl;
            std::cout << " _direction " << _behavior.direction << std::endl;

#ifndef GRND_OBSTACLES_EVO
            std::cout << " corridor time " << simu.end_time;
#else
            std::cout << " corridor time " << (simu1.end_time + simu2.end_time + simu3.end_time) / 3.0;
#endif

            static std::ofstream ofs(std::string("performance_metrics.dat").c_str());
            ofs << _behavior.performance << std::endl;
            ofs << _behavior.direction << std::endl;
            ofs << this->_covered_distance << std::endl;
#ifndef GRND_OBSTACLES_EVO
            ofs << simu.end_time << std::endl;
#else
            ofs << (simu1.end_time + simu2.end_time + simu3.end_time) / 3.0 << std::endl;
#endif
            ofs.close();
        }

        if (write_objs)
        {
            std::cout << std::endl << "fitness" << this->_objs[0] << std::endl << "direction angle" << this->_objs[1] << std::endl  << "diversity" << this->_objs[2] << std::endl; //! _objs[0] is performance
        }
    }

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


        float curv=fabs(A*angle);//curv is length of the perfect curvilinear trajectory

        //!None of the above computations are used in calculating res. In earlier versions, sqrtf(res)/curv was returned, to take into account the length of the trajectory into the comparision of discrepancies
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


//euclidean distance
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
        assert(v1.position.size() == _v.position.size());
        assert(v2.position.size() == _v.position.size());
        return dist(v1.position, _v.position) < dist(v2.position, _v.position);
    }
};

template<typename T, typename T2>
struct compare_dist_p
{
    compare_dist_p(const T2& v) : _v(v) {}
    const T2 _v;
    bool operator()(const T& v1, const T& v2) const
    {
        assert(v1->position.size() == _v.position.size());
        assert(v2->position.size() == _v.position.size());
        return dist(v1->position, _v.position) < dist(v2->position, _v.position);
    }
};


//hamming distance
template<typename V1, typename V2>
float hdist(const V1& v1, const V2& v2)
{
    assert(v1.size() == v2.size());

    typename V1::const_iterator it1 = v1.begin(), it2 = v2.begin();
    float hdist = 0.0f;
    while (it1 != v1.end())
    {
        hdist += (float)((bool)*it1 ^ (bool)*it2);
        ++it1; ++it2;
    }
    return hdist;
}


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
        this->_create_log_file(ea, "all_behaviors.dat");

        float best_performance = -100000.0; int best_ind = -1;
        for (size_t indindex = 0; indindex < ea.pop().size(); ++indindex) //!TODO: Size of population 100, but log file writes 300 ind entries
        {
            const Behavior& v = ea.pop()[indindex]->fit().behavior();
            if(v.performance > best_performance && -fabs(v.direction) >= -ANGLE_THRESHOLD)
            {
                best_performance = v.performance;
                best_ind = (int)indindex;
            }

            (*this->_log_file)<<ea.gen()<< " "<<indindex<< " ";
            for (size_t i = 0; i < v.position.size(); ++i) //Final position of the robot, x and y coordinates
              (*this->_log_file)<<v.position[i]<<" ";

            (*this->_log_file)<<std::endl;
        }

        if(best_ind != -1)
        {
            const Behavior& v = ea.pop()[best_ind]->fit().behavior();
            (*this->_log_file)<<ea.gen()<< " "<<best_ind<< " ";
            for (size_t i = 0; i < v.position.size(); ++i) //Final position of the robot, x and y coordinates
                (*this->_log_file)<<v.position[i]<<" ";

           (*this->_log_file)<<v.performance <<" "<<v.direction<<" -> ";

            for (size_t k = 0; k < ea.pop()[best_ind]->size(); ++k) //parameters of driving signal
                (*this->_log_file)<<ea.pop()[best_ind]->data(k)<<" ";
        }
    }

    template<typename E>
    void _create_log_file(const E& ea, const std::string& name)
    {
        if (!_log_file && ea.dump_enabled())
        {
            std::string log = ea.res_dir() + "/" + name;
            _log_file = boost::shared_ptr<std::ofstream>(new std::ofstream(log.c_str()));
        }
    }
};

//!SFERES_CLASS_D Arguments is the class NoveltyArchive dervied from LogModifier
SFERES_CLASS_D(NoveltyArchive, LogModifier)
{
    public:
    NoveltyArchive() {}
    template<typename Ea>
    void apply(Ea& ea)
    {
        std::vector<Behavior> archive;

        for (size_t i = 0; i < ea.pop().size(); ++i)
        {
            if(ea.pop()[i]->fit().dead())
            {   //setting all objectives to very low values
                ea.pop()[i]->fit().set_obj(0,-10000.0);//distance traversed
                ea.pop()[i]->fit().set_obj(1,-10000.0);//trajectory angle
                ea.pop()[i]->fit().set_obj(2,-10000.0);//behavior diversity
                continue;
            }

            const Behavior& behavior1 = ea.pop()[i]->fit().behavior();
            float behavdiversity = 0.0;
            for (size_t j = 0; j < ea.pop().size(); ++j)
            {
                if(i==j || ea.pop()[j]->fit().dead())
                    continue;

                const Behavior& behavior2 = ea.pop()[j]->fit().behavior();

#ifndef GRND_OBSTACLES_EVO
                assert(behavior1.features.size() == behavior2.features.size());
                behavdiversity += hdist(behavior1.features, behavior2.features);
#else
                assert(behavior1.features_simu1.size() == behavior2.features_simu1.size());
                behavdiversity += (hdist(behavior1.features_simu1, behavior2.features_simu1) +
                                   hdist(behavior1.features_simu2, behavior2.features_simu2) +
                                   hdist(behavior1.features_simu3, behavior2.features_simu3)) / 3.0;
#endif
            }
            behavdiversity /= ea.pop().size();
            ea.pop()[i]->fit().set_obj(2,  behavdiversity);
        }
        this->_log_behaviors(ea);

#ifdef SEEDED_ENTIREPOP
        if (global::gen == global::dmgatgen)
        {
          for (size_t i = 0; i < ea.pop().size(); ++i)
          {
            ea.pop()[i]->fit().eval(*ea.pop()[i]);
          }
        } // <- should add the above 7 lines in your other expt files for damage recovery where the seed population remains unchanged. this evaluates the individuals again. needed because of elitism enabled in nsga2

        global::gen++;
#endif
    }

    std::vector<boost::shared_ptr<Behavior> > & archive() { return _archive; }
    protected:
    std::vector<boost::shared_ptr<Behavior> > _archive;
};


int main(int argc, char **argv)
{
    //initialisation of the simulator
    dInitODE();

#ifndef NO_PARALLEL
    typedef eval::Parallel<Params> eval_t;
#else
    typedef eval::Eval<Params> eval_t;
#endif

    typedef FitSpace<Params> fit_t;
    typedef gen::EvoFloat<24, Params> genom_t;
    typedef phen::Parameters<genom_t, fit_t, Params> phen_t;

    typedef boost::fusion::vector<
            sferes::stat::ParetoFrontConstraintSort<phen_t, Params>
            >  stat_t;

    typedef boost::fusion::vector<NoveltyArchive<Params> > modifier_t;

    if(argc >= 25)
    {
        // test the gait, parameters of which are provided on command line
        init_simu(argc, argv,false);
        genom_t indiv;
        std::cout << "LOADING..." << std::endl;
        for(int i=0;i < 24;++i) //36
        {
            indiv.data(i, atof(argv[i+1])); //36
            //Translated Antoine's tripod gait parameter values from [0,1] to [-1,1]
            /* 1.0   -1.0   -0.5   -0.5    1.0   0   -0.50    0.50    1.0   -1.0   -0.5   -0.5  1.0   -1.0   -0.5    0.5    1.0 0   -0.5   -0.5    1.0   -1.0   -0.5    0.5*/

            // 1.0   0   -0.50    0.50
        }

        fit_t fitness;
        fitness.eval(indiv);

        /*std::ofstream ofss("simu_fit");
        boost::archive::text_oarchive  of(ofss);
        of<<fitness.behavior().position;*/

        std::ofstream workingFile("simu_fit");
        workingFile << fitness.behavior().performance;


        std::ofstream ofs("simu_contacts");
        boost::archive::text_oarchive  oa(ofs);
        oa<<fitness.legs_features;
        // write class instance to archive
        std::cout<<"fin eval"<<std::endl;

#ifndef GRND_OBSTACLES_EVO
        global::robot.reset();
        global::env.reset();
#else
        global::robot1.reset();
        global::env1.reset();
        global::robot2.reset();
        global::env2.reset();
        global::robot3.reset();
        global::env3.reset();
#endif

#ifdef SEEDED_ENTIREPOP
        global::robot_dmg.reset();
        global::env_dmg.reset();
#endif
        dCloseODE();

        return 0;
    }


#ifdef SEEDED_NSGA
    // can provide values as arguments. In json file the arguments can then be given separately with the experiment
    Params::seed::file_name = std::string(argv[1]); //generation filename
    Params::seed::gen_num = atoi(argv[2]);          //id of ind. on pareto front

    typedef ea::Nsga2Seeded<phen_t, eval_t, stat_t, modifier_t, Params> ea_t;
#else
    typedef ea::Nsga2<phen_t, eval_t, stat_t, modifier_t, Params> ea_t;
#endif


    ea_t ea;

    //initilisation of the simulation and the simulated robot
#ifdef GRND_OBSTACLES
    if(argc>7)
    {
        global::numberobstacles = atoi(argv[7]);
    }
#elif defined ROB_DAMAGES
    if(argc>7)
    {
        global::brokenLegs.push_back(atoi(argv[7]));
        std::cerr << "Leg damaged: " << atoi(argv[7]);
    }
#elif defined GRND_SLOPES
    if(argc>7)
    {
        global::floorangle = atoi(argv[7]);
        global::floorangle = global::floorangle - 20.0;
        std::cout << "Floor slope (degrees): " << global::floorangle << std::endl;
    }
#endif

    init_simu(argc, argv,true);

    //#endif
    std::cout<<"debut run"<<std::endl;

    run_ea(argc, argv, ea);
    std::cout <<"fin run"<<std::endl;

#ifndef GRND_OBSTACLES_EVO
    global::robot.reset();
    global::env.reset();
#else
    global::robot1.reset();
    global::env1.reset();
    global::robot2.reset();
    global::env2.reset();
    global::robot3.reset();
    global::env3.reset();
#endif

#ifdef SEEDED_ENTIREPOP
    global::robot_dmg.reset();
    global::env_dmg.reset();
#endif

    dCloseODE();
    std::cout <<"fin"<<std::endl;
    return 0;
}
