#ifndef SIMU_HPP
#define SIMU_HPP


//#define GRND_OBSTACLES
//#define ROB_DAMAGES
//#define GRND_SLOPES
#define GRND_OBSTACLES_EVO

#ifdef GRAPHIC
#include "renderer/osg_visitor.hh"
#endif


#include <boost/tuple/tuple.hpp>
#include <boost/shared_ptr.hpp>
//#include <robot/quadruped.hh>
#include <ode/box.hh>



//#include "actuator.hpp"
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

//#include "robotHexa.hpp"

class Simu
{
public:
    static constexpr float step = 0.015; float end_time;
    typedef boost::shared_ptr<robot::Hexapod> robot_t;
    typedef std::vector<float> ctrl_t;


#if defined GRND_OBSTACLES || defined GRND_OBSTACLES_EVO
    Simu(const ctrl_t& ctrl, const robot_t& robot,std::vector<int> brokenLegs, float duration, std::vector < std::vector<float> > obstacle_pos_rad, float floorangle) :
        _brokenLegs(brokenLegs),
        _controller(ctrl,brokenLegs),
        _covered_distance(10.0f),
        _slam_duration(0.0f),
        _energy(0.0f),
        _env(new ode::Environment_hexa(floorangle, obstacle_pos_rad))
#else
    Simu(const ctrl_t& ctrl, const robot_t& robot,std::vector<int> brokenLegs, float duration, float floorangle) :
        _brokenLegs(brokenLegs),
        _controller(ctrl,brokenLegs),
        _covered_distance(10.0f),
        _slam_duration(0.0f),
        _energy(0.0f),
        _env(new ode::Environment_hexa(floorangle))
#endif
    {
        _robot = robot->clone(*_env);

#ifdef GRAPHIC
        _robot->accept(_visitor);
        /*std::string prefix = "frame";
        _visitor.enable_dump(prefix);*/
#endif

        _env->set_gravity(0, 0, -9.81);

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
