#ifndef CONTROLLERPHASE_HPP
#define CONTROLLERPHASE_HPP

#include <vector>
#include <boost/shared_ptr.hpp>
#include "robot/hexapod.hh"
#include<fstream>

class ControllerPhase
{
protected :
    std::vector< std::vector<float> > _legsParams;
    std::vector< std::vector<float> > _prevAmp;
    std::vector< std::vector<float> > _prevdAmp;
    std::vector< std::vector<float> > _prevPhase;
    std::vector<int> _brokenLegs;

    std::vector< std::vector<float> > _angles;

    std::vector<bool> _contact;
    std::vector< std::vector<float> > _prev_angles_actual;
    std::vector< std::vector<float> > _angles_actual;

private:
    float delayedPhase(float t, float phi);
    float cpg(size_t leg, size_t servo, float IntrinsicAmp, float InitPhase, float t);

public :
    typedef boost::shared_ptr<robot::Hexapod> robot_t;


    bool isBroken(int leg)
    {
        for (int j=0;j<_brokenLegs.size();j++)
        {
            if (leg==_brokenLegs[j])
            {
                return true;
            }
        }
        return false;
    }


    ControllerPhase(const std::vector<float>& ctrl,std::vector<int> brokenLegs):_brokenLegs(brokenLegs)
    {
        assert(ctrl.size()==3*6*2);
        for (int leg=0;leg<6;leg++)
        {
            std::vector<float> param;

            param.push_back(ctrl[leg*6]);   //amplitude - servo 0 (intrinsic amplitude)
            param.push_back(ctrl[leg*6+1]); //phase  - servo 0    (initial phase)
            param.push_back(ctrl[leg*6+2]); //amplitude  - servo 1
            param.push_back(ctrl[leg*6+3]); //phase - servo 1
            param.push_back(ctrl[leg*6+4]); //amplitude  - servo 2
            param.push_back(ctrl[leg*6+5]); //phase  - servo 2

            _legsParams.push_back(param);

            std::vector<float> tmp(3, 0.0); //for the three oscillators on each leg
            _prevAmp.push_back(tmp);
            _prevdAmp.push_back(tmp);
            _prevPhase.push_back(tmp);

            _angles.push_back(tmp);

            _contact.push_back(true);
            _angles_actual.push_back(tmp);
            _prev_angles_actual.push_back(tmp);
        }
    }

    void moveRobot(robot_t& robot, float t);
    std::vector<int> get_pos_dyna(float t);
    std::vector<int> get_speeds_dyna( );
    std::vector<bool> get_directions_dyna( );
};

#endif
