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
#include "gen_hnn.hpp"
#include "phen_hnn.hpp"

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

#define MutMultiplier 1.0   // standard mutation rates

using namespace sferes;
using namespace boost::assign;
using namespace sferes::gen::dnn;
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

    //static long seed;
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

    struct cppn
    {
      // params of the CPPN
      struct sampled
      {
        SFERES_ARRAY(float, values, 0, 1, 2, 3); //sine; sigmoid; gaussian; linear;
        static constexpr float mutation_rate = 0.01f * MutMultiplier;
        static constexpr float cross_rate = 0.25f;
        static constexpr bool ordered = false;
      };
      struct evo_float
      {
        static constexpr float mutation_rate = 0.1f * MutMultiplier;//per conn mut rate
        static constexpr float eta_m = 10.0f /  MutMultiplier;  // perturbation of the order O(1/eta_m)

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
          //x1 and y1, x2 and y2, the positions of the source and target HNN nodes + bias input
          static constexpr size_t nb_inputs = 5;

          //outputs: weight of connection input to hidden layer, and from hidden to output layer of HNN
          static constexpr size_t nb_outputs = 2;
        static constexpr init_t init = ff; //feedforward nn
        static constexpr float m_rate_add_conn = 0.05f * MutMultiplier; //0.1f
        static constexpr float m_rate_del_conn = 0.1f * MutMultiplier;
        static constexpr float m_rate_change_conn = 0.0f * MutMultiplier; //move conn to new neurons
        static constexpr float m_rate_add_neuron = 0.05f * MutMultiplier; //0.2f
        static constexpr float m_rate_del_neuron = 0.01f * MutMultiplier;

        //these only count w/ random init, instead of feed forward
        static constexpr size_t min_nb_neurons = 6;
        static constexpr size_t max_nb_neurons = 20;
        static constexpr size_t min_nb_conns = 30;
        static constexpr size_t max_nb_conns = 150;
      };
    };

    struct hnn
    {
      static constexpr size_t nb_rand_mutations = 0;
      static constexpr size_t nb_layers = 3;

      static constexpr size_t nb_inputs   = 14;   //6 legs * 2 motors + 2 misc nodes
      static constexpr size_t nb_outputs  = 12; // 12 outputs controlling the hip and knee motors of each of the six legs

      typedef float weight_t;
      typedef float io_t;
      typedef nn::params::Vectorf<1> params_t1;

      /*typedef nn::Neuron<nn::PfWSum<io_t>,
                         nn::AfSigmoidBias<params_t1>, io_t > neuron_t;*/

      typedef nn::Neuron<nn::PfWSum<io_t>,
                         nn::AfTanh<params_t1>, io_t > neuron_t;
      typedef nn::Connection<weight_t, io_t> connection_t;
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
#endif
};
///---------------------------

void init_simu(int argc ,char** argv, bool master)
{
    global::brokenLegs[1] = 3;
    //global::brokenLegs[1] = 4;

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
        //three objectives: performance - displacement, direction angle (wrt y-axis) and behavior diversity.
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


    size_t cppn_nodes() const { return _cppn_nodes; }
    size_t cppn_conns() const { return _cppn_conns; }
    size_t nn_nodes() const { return _nn_nodes; }
    size_t nn_conns() const { return _nn_conns; }

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

    size_t _cppn_nodes;
    size_t _cppn_conns;
    size_t _nn_nodes;
    size_t _nn_conns;

    template<typename Indiv>
    void _eval(Indiv& indiv, bool write_objs)
    {
        indiv.nn().init();
        _cppn_nodes = indiv.gen().returncppn().get_nb_neurons();
        _cppn_conns = indiv.gen().returncppn().get_nb_connections();

        if (this->mode() == sferes::fit::mode::view) //is used to  view individuals in pareto front (their ids. can be got from pareto.data)
        {
            //indiv.nn().simplify();
            std::ofstream ofs(std::string("cppn.dot").c_str());
            indiv.gen().write_cppndot(ofs, indiv.gen().returncppn());

            std::ofstream ofs_cppn(std::string("cppn.dat").c_str());
            ofs_cppn << indiv.gen().returncppn().get_nb_neurons() << " " << indiv.gen().returncppn().get_nb_connections() <<  " " << indiv.gen().returncppn().get_nb_inputs() <<  " " << indiv.gen().returncppn().get_nb_outputs() << std::endl;

            std::ofstream ofs_hnn(std::string("hnnn.dot").c_str());
            indiv.nn().write(ofs_hnn);
        }

        //launching the simulation
        //Simu simu = Simu(indiv.nn(), global::robot, global::brokenLegs);
#ifdef SEEDED_ENTIREPOP
        Simu simu = Simu(indiv.nn(), global::gen < global::dmgatgen ? global::robot : global::robot_dmg, global::gen < global::dmgatgen ? global::brokenLegs : global::brokenLegs_dmg);
#else
       Simu simu = Simu(indiv.nn(), global::robot, global::brokenLegs);
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
            _behavior.performance = -round(sqrt(
                                             (_behavior.position[0]-goal_pos[0])*(_behavior.position[0]-goal_pos[0]) +
                                             (_behavior.position[1]-goal_pos[1])*(_behavior.position[1]-goal_pos[1]))*100.0) / 100.0f; //-ve sign so as to maximize this quantity
            this->_objs[0] = _behavior.performance;

//            _behavior.arrivalangle = round(this->_arrival_angle * 100) / 100.0f; //two decimal places
//            this->_objs[1] = -fabs(_behavior.arrivalangle);

            _behavior.direction = round(this->_direction * 100.0) / 100.0f; //two decimal places
            this->_objs[1] = -fabs(_behavior.direction);
        }

        if(this->mode() == sferes::fit::mode::view)
        {
            std::cout << " _behavior.performance " << _behavior.performance << std::endl;
            std::cout << " _direction " << _behavior.direction << std::endl;

            static std::ofstream ofs(std::string("performance_metrics.dat").c_str());
            ofs << _behavior.performance << std::endl;
            ofs << _behavior.direction << std::endl;
            ofs << this->_covered_distance << std::endl;
            ofs.close();
        }

        if (write_objs)
        {
            std::cout << std::endl << "fitness" << this->_objs[0] << std::endl << "direction" << this->_objs[1] << std::endl  << "diversity" << this->_objs[2] << std::endl; //! _objs[0] is performance
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

        for (size_t indindex = 0; indindex < ea.pop().size(); ++indindex) //!TODO: Size of population 100, but log file writes 300 ind entries
        {
            const Behavior& v = ea.pop()[indindex]->fit().behavior();

            (*this->_log_file)<<ea.gen()<< " "<<indindex<< " ";
            for (size_t i = 0; i < v.position.size(); ++i) //Final position of the robot, x and y coordinates
              (*this->_log_file)<<v.position[i]<<" ";

            (*this->_log_file)<<std::endl;
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
                ea.pop()[i]->fit().set_obj(1,-10000.0);//direction of trajectory angle
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
            }
            behavdiversity /= ea.pop().size();
            ea.pop()[i]->fit().set_obj(2,  behavdiversity);
        }

        this->_log_behaviors(ea);

#ifdef SEEDED_ENTIREPOP
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

    using namespace nn;
    typedef FitSpace<Params> fit_t;
    typedef phen::Parameters<gen::EvoFloat<1, Params::cppn>,
            fit::FitDummy<>, Params::cppn> weight_t;
    typedef gen::Hnn<weight_t, Params, Params::cppn> gen_t;
    typedef phen::Hnn<gen_t, fit_t, Params> phen_t;


    typedef boost::fusion::vector<
            sferes::stat::ParetoFrontConstraintSort<phen_t, Params>
            >  stat_t;

    typedef boost::fusion::vector<NoveltyArchive<Params> > modifier_t;


    if(argc >= 25)
    {
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

    //#endif
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
