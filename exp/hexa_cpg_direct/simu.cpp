#ifdef GRAPHIC
#include "renderer/osg_visitor.hh"
#endif

#include <sferes/misc.hpp>
#include <numeric>
#include "simu.hpp"


void Simu ::_make_robot_init(float duration)
{
    robot_t rob = this->robot();
    Eigen::Vector3d rot=rob->rot();
    _arrival_angle= atan2( cos(rot[2])* sin(rot[1])* sin(rot[0]) + sin(rot[2])* cos(rot[0]), cos(rot[2])* cos(rot[1]))*180/M_PI;

    _controller.moveRobot(rob,0);

    float t=0;
    int index = 0;
#ifdef GRAPHIC
    while (t < duration && !_visitor.done()) //!visitor, term for graphic interface (esc aborts simulation)
#else
    while (t < duration)
#endif
    {
        _controller.moveRobot(rob,t);

        if(_robot->bodies()[0]->get_in_contact() || _env->get_colision_between_legs())
        {//Death if robot body touches ground or if legs collide
#ifdef GRAPHIC
            std::cout<<"mort subite"<<std::endl;
#endif
            _covered_distance=-10002.0;
            return;
        }
        int nbCassee=0;
        //Log contact points every iteration to better measure behav diversity for high frequency gaits
        //if (index%2==0) //Log leg contact points at every other iteration
            for (unsigned i = 0; i < 6; ++i)
            {
                switch (i)
                {
                case 0:
                    if (_controller.isBroken(i))
                    {
                        _behavior_contact_0.push_back(0);
                        nbCassee++;
                    }
                    else
                    {
                        _behavior_contact_0.push_back( _robot->bodies()[(i-nbCassee) * 3 + 3]->get_in_contact() );
                    }
                    break;
                case 1:
                    if (_controller.isBroken(i))
                    {
                        _behavior_contact_1.push_back(0);
                        nbCassee++;
                    }
                    else
                    {
                        _behavior_contact_1.push_back( _robot->bodies()[(i-nbCassee) * 3 + 3]->get_in_contact() );
                    }
                    break;
                case 2:
                    if (_controller.isBroken(i))
                    {
                        _behavior_contact_2.push_back(0);
                        nbCassee++;
                    }
                    else
                    {
                        _behavior_contact_2.push_back( _robot->bodies()[(i-nbCassee) * 3 + 3]->get_in_contact() );
                    }
                    break;
                case 3:
                    if (_controller.isBroken(i))
                    {
                        _behavior_contact_3.push_back(0);
                        nbCassee++;
                    }
                    else
                    {
                        _behavior_contact_3.push_back( _robot->bodies()[(i-nbCassee) * 3 + 3]->get_in_contact() );
                    }
                    break;
                case 4:
                    if (_controller.isBroken(i))
                    {
                        _behavior_contact_4.push_back(0);
                        nbCassee++;
                    }
                    else
                    {
                        _behavior_contact_4.push_back( _robot->bodies()[(i-nbCassee) * 3 + 3]->get_in_contact() );
                    }
                    break;
                case 5:
                    if (_controller.isBroken(i))
                    {
                        _behavior_contact_5.push_back(0);
                        nbCassee++;
                    }
                    else
                    {
                        _behavior_contact_5.push_back( _robot->bodies()[(i-nbCassee) * 3 + 3]->get_in_contact() );
                    }
                    break;
                }
            }
        _behavior_traj.push_back(rob->pos());


#if defined GRND_OBSTACLES || defined ROB_DAMAGES || defined GRND_SLOPES
            if(rob->pos()[0] > 0.7 || rob->pos()[0] < -0.7) // if the robot steps out of the corridor (of width 40cm), we terminate the run
                break;
#endif

        t += step;
        next_step();

        ++index;
    }

    end_time = t;

    Eigen::Vector3d prev_pos = rob->pos();

#if !(defined GRND_OBSTACLES || defined ROB_DAMAGES || defined GRND_SLOPES)
        stabilize_robot(); // post-stabilization now removed in BOMEAN
#endif
    Eigen::Vector3d next_pos = rob->pos();

    if((fabs(prev_pos[0]-next_pos[0]) > 0.4) || (fabs(prev_pos[1]-next_pos[1]) > 0.4) || (fabs(prev_pos[2]-next_pos[2]) > 0.4))
    {
        //Death if robot transitions more than 400cm during the stablization phase
        #ifdef GRAPHIC
            std::cout<<"mort subite"<<std::endl;
        #endif
        _covered_distance=-10002.0;
        return;
    }

    _final_pos.resize(2);
    _final_pos[0]=next_pos[0];
    _final_pos[1]=next_pos[1];

    _covered_distance = round(sqrt(next_pos[0]*next_pos[0]+next_pos[1]*next_pos[1]/*+next_pos[2]*next_pos[2]*/)*100) / 100.0f;
    _direction=atan2(-next_pos[0],next_pos[1])*180/M_PI;
    rot=rob->rot();
    _arrival_angle= atan2( cos(rot[2])* sin(rot[1])* sin(rot[0])
            + sin(rot[2])* cos(rot[0]), cos(rot[2])* cos(rot[1]))*180/M_PI;
    while(_arrival_angle<-180)
        _arrival_angle+=360;
    while(_arrival_angle>180)
        _arrival_angle-=360;
}
