/*================================================================*/
//#define SEEDED_ENTIREPOP // The entire population is seeded as it is
/*================================================================*/
//#define SEEDED_NSGA

#ifdef SEEDED_NSGA
    #include <modules/seeded_nsga2/seeded_nsga2.hpp>
    #include <modules/nn2/trait.hpp>
#endif
/*================================================================*/

#define EIGEN_USE_NEW_STDVECTOR
#include <Eigen/StdVector>
#include <Eigen/Core>
#include <Eigen/Array>
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
#include "oscillator.hpp"


#define NO_MPI

#include "ode/box.hh"

#ifdef GRAPHIC
#define NO_PARALLEL
#include "renderer/osg_visitor.hh"
#endif


#include "hexapod.hh"
#include "simu.hpp"
#include <boost/fusion/sequence/intrinsic/at.hpp>


#define DEFAULTVALUE -10000.0 // default value for phase bias couplings
#define NumOsc 12 //Number of oscillators


using namespace sferes;
using namespace boost::assign;
using namespace sferes::gen::evo_float;


struct Params
{
#ifdef SEEDED_NSGA
    struct seed
    {
      static const unsigned stat_num = 0; //number of statistics; 0 -> just one stat, the pareto front
      static unsigned gen_num;            //id of the individual in the pareto front, to be loaded
      static std::string file_name;       //generation file name
    };
#endif

    struct pop //Parameters of the population
    {
        static const unsigned size = 100; //population size
#ifdef SEEDED_NSGA
        static const unsigned nb_gen = 10001; //10001; //total number of generations of evolution
        static const int dump_period = 10;    //logs are written every dump_period generations
#else

#ifdef SEEDED_ENTIREPOP
        static const unsigned nb_gen = 8000 + 10001;
#else
        static const unsigned nb_gen = 10001; //10001;
#endif
        static const int dump_period = 50;  //logs are written every dump_period generations
#endif

        static const int initial_aleat = 1; //initial population size at first generation is scaled by initial_aleat
    };

    struct evo_float
    {
      static const float mutation_rate = 0.1f;//per conn mut rate
      static const float cross_rate    = 0.1f; //we don't use this
      static const mutation_t mutation_type = polynomial;
      static const cross_over_t cross_over_type = sbx;
      static const float eta_m = 10.0f;
      static const float eta_c = 10.0f;
    };

    struct parameters //!Min. and max. parameters of genes represented with EvoFloat
    {
        static const float min = -1.0f;
        static const float max =  1.0f;
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
boost::shared_ptr<robot::Hexapod> robot;
boost::shared_ptr<ode::Environment_hexa> env;

//std::vector<int> brokenLegs; // broken legs for the global::robot and not global::robot_dmg
std::vector<int> brokenLegs(2,1); // middle leg and back leg of other side is broken
//std::vector<int> brokenLegs(1,1); // middle leg is broken


#ifdef SEEDED_ENTIREPOP
    unsigned gen=0;
    unsigned dmgatgen = 8000;

    //Pointer to another robot Hexapod (DAMAGED) and the ode environment, to continue evolution when the ROBOT GET DAMAGED.
    boost::shared_ptr<robot::Hexapod> robot_dmg;
    boost::shared_ptr<ode::Environment_hexa> env_dmg;

    std::vector<int> brokenLegs_dmg(1,1); // middle leg of damaged robot is broken // (global::brokenLegs_dmg used with global::robot_dmg to continue evolution with damaged robot);
    //std::vector<int> brokenLegs_dmg(2,1); // middle right leg  + rear left leg of damaged robot is brokena // (global::brokenLegs_dmg used with global::robot_dmg to continue evolution with damaged robot);
#endif
};
///---------------------------


void init_simu(int argc ,char** argv, bool master)
{

/*#ifdef SEEDED_ENTIREPOP
    global::brokenLegs_dmg[1] = 3;
#endif*/
    global::brokenLegs[1] = 3;

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
        if ((global::robot->pos() - prev_pos).norm() < 1e-5)
            stab++;
        else
            stab = 0;
        if (stab > 100)
            stabilized = true;
    }
    assert(stabilized);
    global::env->set_gravity(0, 0, -9.81); //returning gravity to normal



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

    int querycouplings(int n1, int n2)
    {
        if(n2 < 0 || n2 >= NumOsc)
            return -1;

        for(size_t i=0; i<NumOsc;++i)
            if(_oscillator.adjacency[n1][i] == -1)
            {
                _oscillator.adjacency[n1][i] = n2;
                break;
            }

        if(_oscillator.phasebias_couplings[n1][n2] == DEFAULTVALUE)
            return 1;

        return -1;
    }

    std::vector<std::vector<float> > legs_features;
    std::vector<float>  features;
    std::vector<float>  disparity;

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
        _oscillator.amp_osc.clear();
        _oscillator.phasebias_couplings.clear();
        _oscillator.adjacency.clear();

        //inter-oscillator couplings
        for (size_t n = 0; n < NumOsc; ++n)
        {
            std::vector<float> params_tmp(NumOsc, DEFAULTVALUE);
            _oscillator.phasebias_couplings.push_back(params_tmp);

            std::vector<int> params_tmp1(NumOsc, -1);
            _oscillator.adjacency.push_back(params_tmp1);
        }

         assert(_oscillator.phasebias_couplings.size()==NumOsc);
         assert(_oscillator.phasebias_couplings[0].size()==NumOsc);
         assert(_oscillator.adjacency.size()==NumOsc);
         assert(_oscillator.adjacency[0].size()==NumOsc);

        assert(indiv.size() == NumOsc + 17); // amplitude of oscillators (two oscillators on each of six legs) + inter-oscillator phase bias couplings
        size_t osc_index = 0;
        for (size_t leg=0; leg < 6;++leg)
        {
            for (size_t oscl=0; oscl < 2;++oscl)
            {
                _oscillator.amp_osc.push_back(indiv.data(osc_index++));
            }
        }
        assert(_oscillator.amp_osc.size() == NumOsc); // 2 oscillators on each of six legs


        for (int n1 = 0; n1 < (int)NumOsc; ++n1)
        {
            if(!(n1 == 3 || n1 == 7 || n1 == 11)) //exceptions at left and right border of subtrate to prevent toroidal couplings
            {
                if(querycouplings(n1, n1 + 1)==1) // horizontal couplings
                {
                    int n2 = n1 + 1;
                    float r = indiv.data(osc_index++);
                     _oscillator.phasebias_couplings[n1][n2] = r; _oscillator.phasebias_couplings[n2][n1] = -r;
                }
            }

            if(!(n1 == 0 || n1 == 4 || n1 == 8)) //exceptions at left and right border of subtrate to prevent toroidal couplings
            {
                if(querycouplings(n1, n1 - 1)==1) // horizontal couplings
                {
                    int n2 = n1 - 1;
                    float r = indiv.data(osc_index++);
                     _oscillator.phasebias_couplings[n1][n2] = r; _oscillator.phasebias_couplings[n2][n1] = -r;
                }
            }

            if(querycouplings(n1, n1 + 4)==1) // vertical couplings
            {
                int n2 = n1 + 4;
                float r = indiv.data(osc_index++);
                 _oscillator.phasebias_couplings[n1][n2] = r; _oscillator.phasebias_couplings[n2][n1] = -r;
            }
            if(querycouplings(n1, n1 - 4)==1) // vertical couplings
            {
                int n2 = n1 - 4;
                float r = indiv.data(osc_index++);
                _oscillator.phasebias_couplings[n1][n2] = r; _oscillator.phasebias_couplings[n2][n1] = -r;
            }
        }

        if (this->mode() == sferes::fit::mode::view) //is used to  view individuals in pareto front (their ids. can be got from pareto.data)
        {

        }

        //launching the simulation
#ifdef SEEDED_ENTIREPOP
        Simu simu = Simu(_oscillator, global::gen < global::dmgatgen ? global::robot : global::robot_dmg, global::gen < global::dmgatgen ? global::brokenLegs : global::brokenLegs_dmg);
#else
       Simu simu = Simu(_oscillator, global::robot, global::brokenLegs);
#endif

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
            _behavior.performance = -round(sqrt(
                                             (_behavior.position[0]-goal_pos[0])*(_behavior.position[0]-goal_pos[0]) +
                                             (_behavior.position[1]-goal_pos[1])*(_behavior.position[1]-goal_pos[1]))*100) / 100.0f; //-ve sign so as to maximize this quantity
            this->_objs[0] = _behavior.performance;

//            _behavior.arrivalangle = round(this->_arrival_angle * 100) / 100.0f; //two decimal places
//            this->_objs[1] = -fabs(_behavior.arrivalangle);
            _behavior.direction = round(this->_direction * 100) / 100.0f; //two decimal places
            this->_objs[1] = -fabs(_behavior.direction);
        }

        if(this->mode() == sferes::fit::mode::view)
        {
            std::cout << " _behavior.performance " << _behavior.performance << std::endl;
            std::cout << " _direction" << _behavior.direction << std::endl;

            static std::ofstream ofs(std::string("performance_metrics.dat").c_str());
            ofs << _behavior.performance << std::endl;
            ofs << _behavior.direction << std::endl;
            ofs << this->_covered_distance << std::endl;
            ofs.close();
        }

        //writting objectives if needed
        if (write_objs)
        {
            std::cout << std::endl << "fitness" << this->_objs[0] << std::endl << "direction of trajectory" << this->_objs[1] << std::endl  << "diversity" << this->_objs[2] << std::endl; //! _objs[0] is performance
        }
    }

    Oscillator _oscillator;
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

        for (size_t indindex = 0; indindex < ea.pop().size(); ++indindex) //!TODO: Size of population 100, but log file writes 300 ind entries
        {
            const Behavior& v = ea.pop()[indindex]->fit().behavior();

            (*this->_log_file)<<ea.gen()<< " "<<indindex<< " ";
            for (size_t i = 0; i < v.position.size(); ++i) //Final position of the robot, x and y coordinates
              (*this->_log_file)<<v.position[i]<<" ";

            (*this->_log_file)<<std::endl;
        }

         //!TODO returns pareto front of size 0, WHY
        /*if(ea.pareto_front().size())
        {
            const Behavior& v = ea.pareto_front()[0]->fit().behavior();
            (*this->_log_file)<<ea.gen()<< " "<< "P0" << " ";
            for (size_t i = 0; i < v.position.size(); ++i) //Final position of the robot, x and y coordinates
                (*this->_log_file)<<v.position[i]<<" ";
            for (size_t k = 0; k < ea.pareto_front()[0]->osc.size(); ++k) //parameters of cpg
                (*this->_log_file)<<ea.pareto_front()[0]->osc[k]<<" ";
            (*this->_log_file)<<std::endl;
        }*/
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
    //NoveltyArchive() : _rho_min(0.10f), _not_added(0) {}
    NoveltyArchive() {}
    template<typename Ea>
    void apply(Ea& ea)
    {
        std::vector<Behavior> archive;
        //for (size_t i = 0; i < ea.pop().size(); ++i)
        //    archive.push_back(ea.pop()[i]->fit().behavior());

        for (size_t i = 0; i < ea.pop().size(); ++i)
        {
            if(ea.pop()[i]->fit().dead())
            {   //setting both objectives to very low values
                ea.pop()[i]->fit().set_obj(0,-10000.0);//distance traversed
                ea.pop()[i]->fit().set_obj(1,-10000.0);//direction angle
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

                assert(behavior1.features.size() == behavior2.features.size());
                behavdiversity += hdist(behavior1.features, behavior2.features);
                //behavdiversity += dist(behavior1.dutycycles, behavior2.dutycycles);
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
    }
    global::gen++;
#endif


    }

    std::vector<boost::shared_ptr<Behavior> > & archive() { return _archive; }
    protected:
    std::vector<boost::shared_ptr<Behavior> > _archive;
};


int main(int argc, char **argv)
{
    dInitODE();

#ifndef NO_PARALLEL
    typedef eval::Parallel<Params> eval_t;
#else
    typedef eval::Eval<Params> eval_t;
#endif


    typedef FitSpace<Params> fit_t;
    typedef gen::EvoFloat<NumOsc + 17, Params> genom_t;
    typedef phen::Parameters<genom_t, fit_t, Params> phen_t;

    typedef boost::fusion::vector<
            sferes::stat::ParetoFrontConstraintSort<phen_t, Params>
            >  stat_t;


    typedef boost::fusion::vector<NoveltyArchive<Params> > modifier_t;


    if(argc >= 25)
    {
        // test the gait, parameters of which are provided on command line
//        init_simu(argc, argv,false);
//        phen_t indiv;
//        std::cout << "LOADING..." << std::endl;
//        indiv.osc.resize(24);   //36
//        for(int i=0;i < 24;++i) //36
//        {
//            /*24 parameters*/
//            indiv.osc[i] = atof(argv[i+1]);
//            //Translated Antoine's tripod gait parameter values from [0,1] to [-1,1]
//            /* 1.0   -1.0   -0.5   -0.5    1.0   0   -0.50    0.50    1.0   -1.0   -0.5   -0.5 1.0   -1.0   -0.5    0.5    1.0   0   -0.5   -0.5    1.0   -1.0   -0.5    0.5*/
//        }

//        fit_t fitness;
//        fitness.eval(indiv);

//        std::ofstream ofss("simu_fit");
//        boost::archive::text_oarchive  of(ofss);
//        of<<fitness.behavior().position;
//        std::ofstream ofs("simu_contacts");
//        boost::archive::text_oarchive  oa(ofs);
//        oa<<fitness.legs_features;

//        std::cout<<"fin eval"<<std::endl;

//        global::robot.reset();
//        global::env.reset();

//#ifdef SEEDED_ENTIREPOP
//        global::robot_dmg.reset();
//        global::env_dmg.reset();
//#endif
//        dCloseODE();

//        return 0;
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
    init_simu(argc, argv,true);

    std::cout<<"debut run"<<std::endl;

    run_ea(argc, argv, ea);
    std::cout <<"fin run"<<std::endl;

    global::robot.reset();
    global::env.reset();

#ifdef SEEDED_ENTIREPOP
        global::robot_dmg.reset();
        global::env_dmg.reset();
#endif
    dCloseODE();
    std::cout <<"fin"<<std::endl;
    return 0;
}
