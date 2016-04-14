#ifndef SIMU_HPP
#define SIMU_HPP

#include <boost/tuple/tuple.hpp>
#include <boost/shared_ptr.hpp>
#include <ode/box.hh>

#include "hexapod.hh"
#include "controllerPhase.hpp"
#define MAX_ANG_VEL 11.9380521
#define DYN2RAD 0.00511326929
#define SAMPLING_FREQUENCY 20

#include <fstream>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/filesystem.hpp>

#include <modules/nn2/phen_dnn.hpp>

class Simu
{
public:
    static constexpr float step = 0.015;
    typedef boost::shared_ptr<robot::Hexapod> robot_t;

    typedef float weight_t;
    typedef float io_t;
    typedef nn::params::Vectorf<1> params_t;
    typedef nn::Neuron<nn::PfWSum<io_t>,
                       nn::AfTanh<params_t>, io_t > neuron_t;
    typedef nn::Connection<weight_t, io_t> connection_t;
    typedef typename nn::NN<neuron_t, connection_t> ctrl_t;


    Simu(ctrl_t& ctrl, const robot_t& robot,std::vector<int> brokenLegs) :
        _brokenLegs(brokenLegs),
        _controller(ctrl,brokenLegs),
        _covered_distance(10.0f),
        _slam_duration(0.0f),
        _energy(0.0f),
    #ifdef GRAPHIC
       _env(new ode::Environment_hexa()),
       _visitor(renderer::OsgVisitor::FOLLOW)
   #else
       _env(new ode::Environment_hexa())
    #endif
    {
        float duration = 5.0; // seconds

        _robot = robot->clone(*_env);


        _env->set_gravity(0, 0, -9.81);
        _env->resetGRF();

        try
        {
            _make_robot_init(duration);
        }
        catch (int e)
        {
            std::cout << "An exception occurred. Exception Nr. " << e << std::endl;
            _covered_distance=-10002.0; //!Dirty way to prevent selection on this robot
        }

#ifdef GRAPHIC
        write_contact("contact_simu.txt");
        write_traj("traj_simu.txt");
#endif
    }

    ~Simu()
    {
        // we have to clean in the good order
        _robot.reset();
        _env->resetGRF();
        _env.reset();
    }

    void next_step()
    {
        _robot->next_step(step);
        _env->next_step(step);

#ifdef GRAPHIC
        _visitor.update();
        usleep(1e4);

#endif
    }

    robot_t robot()
    {
        return _robot;
    }

    float covered_distance()
    {
        return _covered_distance;
    }

  std::vector<float> get_orientation_bd(float perc_threshold)
  {
    std::vector<float> results;
    results.push_back(countofvector(pitch_vec , (perc_threshold / 100.0) * M_PI, true));
    results.push_back(countofvector(pitch_vec , (perc_threshold / 100.0) * M_PI, false));
    results.push_back(countofvector(roll_vec , (perc_threshold / 100.0) * M_PI, true));
    results.push_back(countofvector(roll_vec , (perc_threshold / 100.0) * M_PI, false));
    results.push_back(countofvector(yaw_vec , (perc_threshold / 100.0) * M_PI, true));
    results.push_back(countofvector(yaw_vec , (perc_threshold / 100.0) * M_PI, false));
    return results;
  }


    float slam_duration()
    {
        return _slam_duration;
    }

    float energy()
    {
        return _energy;
    }

    float direction()  {return _direction;}
    float arrival_angle() {return _arrival_angle;}
    std::vector<float> final_pos(){return _final_pos;}

    void write_contact(std::string const name)
    {
        std::ofstream workingFile(name.c_str());

        if (workingFile)
        {
            for (int i =0;i<_behavior_contact_0.size();i++)
            {
                workingFile<<_behavior_contact_0[i]<<" "<<_behavior_contact_1[i]<<" "<<_behavior_contact_2[i]<<" "<<_behavior_contact_3[i]<<" "<<_behavior_contact_4[i]<<" "<<_behavior_contact_5[i]<<std::endl;
            }
        }
        else
        {
            std::cout << "ERROR: Impossible to open the file." << std::endl;
        }
    }

    void write_traj(std::string const name)
    {
        std::ofstream workingFile(name.c_str());

        if (workingFile)
        {
            for (int i =0;i<_behavior_traj.size();i++)
            {
                workingFile<<_behavior_traj[i][0]<<" "<<_behavior_traj[i][1]<<" "<<_behavior_traj[i][2]<<std::endl;
            }
        }
        else
        {
            std::cout << "ERROR: Impossible to open the file." << std::endl;
        }
    }

    const std::vector<Eigen::Vector3d>& get_traj()
    {
        return _behavior_traj;
    }

    const std::vector<float>& get_contact(int i)
    {
        switch (i)
        {
        case 0:
            return _behavior_contact_0;
            break;
        case 1:
            return _behavior_contact_1;
            break;
        case 2:
            return _behavior_contact_2;
            break;
        case 3:
            return _behavior_contact_3;
            break;
        case 4:
            return _behavior_contact_4;
            break;
        case 5:
            return _behavior_contact_5;
            break;

        }
        assert(false);
        return _behavior_contact_0;
    }

    //count the number of elements above a threshold
    template<typename V1>
    float countofvector(const V1& v1, double threshold, bool dir)
    {
        typename V1::const_iterator it1 = v1.begin();
        float res = 0.0f;
        while (it1 != v1.end())
        {
            if(dir && (float)*it1 > threshold)
                res+=1.0f;

            if(!dir && (float)*it1 < -threshold)
                res+=1.0f;

            ++it1;
        }
        return res / (float) v1.size() ;
    }


    std::vector<std::vector<float> > angles_forfft;
    float servo_frequencies_max;

    // Additional behavior descriptors
    std::vector<double> pitch_vec, roll_vec, yaw_vec; // orientation
    float amp_x, amp_y, amp_z; // for the amplitude behavior descriptor
    std::vector < std::vector <float> > grf_x, grf_y, grf_z, grf_angle_xz, grf_angle_yz, grf_angle_xy;   // GRF along each component for legs 0 .. 5

protected:
    bool stabilize_robot()
    {
        robot_t rob = this->robot();

        // low gravity to slow things down (eq. smaller timestep?)
        _env->set_gravity(0, 0, -9.81);
        bool stabilized = false;
        int stab = 0;

        for (size_t s = 0; s < 1000 && !stabilized; ++s)
        {
            Eigen::Vector3d prev_pos = rob->pos();
            next_step();
            if ((rob->pos() - prev_pos).norm() < 1e-4)
                stab++;
            else
                stab = 0;
            if (stab > 30)
                stabilized = true;
        }
        _env->set_gravity(0, 0, -9.81);
        return(stabilized);
    }

    float servo_frequency(float t, float duration)
    {
        float frequency;
        if (angles_forfft.size() >= 1) // just in case a run is stopped short
        {
            std::vector<std::vector<float> > phasechanges_forfft;
            std::vector<float> tmp_phasechanges(18,0.0);
            for (int o =1; o < angles_forfft.size(); o++)
            {
                for (int p =0; p < angles_forfft[o].size(); p++)
                {

                    if(angles_forfft[o][p] == angles_forfft[o-1][p]) // && _prev_angles_actual[leg][0] == 0
                        tmp_phasechanges[p] = 0.0;
                    else if(angles_forfft[o][p] < angles_forfft[o-1][p])
                        tmp_phasechanges[p] = -1.0;
                    else if(angles_forfft[o][p] > angles_forfft[o-1][p])
                        tmp_phasechanges[p] = 1.0;
                }
                phasechanges_forfft.push_back(tmp_phasechanges);
            }

            std::vector<float> servo_frequencies(18,0.0);
            for (int o =1; o < phasechanges_forfft.size(); o++)
                for (int p =0; p < phasechanges_forfft[o].size(); p++)
                {
                    if ((phasechanges_forfft[o][p] == -1.0 && phasechanges_forfft[o-1][p] == 0.0) ||
                            (phasechanges_forfft[o][p] == -1.0 && phasechanges_forfft[o-1][p] == 1.0))
                        servo_frequencies[p] = servo_frequencies[p] + 1.0;
                }

            frequency = -1.0; // max servo frequency
            for (int p =0; p < servo_frequencies.size(); p++)
                if (servo_frequencies[p] > frequency)
                    frequency = servo_frequencies[p];

            frequency = frequency / std::min(t, duration); // in case a run is cut short, and does not last till 'duration'
        }
        else
            frequency = 0.0;

        return frequency;
    }

    void _make_robot_init(float duration);

    std::vector<int> _brokenLegs;
    std::vector<Eigen::Vector3d> _behavior_traj;
    std::vector<float> _behavior_contact_0;
    std::vector<float> _behavior_contact_1;
    std::vector<float> _behavior_contact_2;
    std::vector<float> _behavior_contact_3;
    std::vector<float> _behavior_contact_4;
    std::vector<float> _behavior_contact_5;
    ControllerPhase _controller;
    robot_t _robot;
    std::vector<float> _final_pos;
    float _direction;
    float _arrival_angle;
    float _covered_distance;
    float _slam_duration;
    float _energy;
    boost::shared_ptr<ode::Environment_hexa> _env;
#ifdef GRAPHIC
    renderer::OsgVisitor _visitor;
#endif
};

#endif
