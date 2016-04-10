#ifndef CONTROLLERPHASE_HPP
#define CONTROLLERPHASE_HPP

#include <vector>
#include <boost/shared_ptr.hpp>
#include "robot/hexapod.hh"

#include <modules/nn2/phen_dnn.hpp>


class ControllerPhase
{
    typedef float weight_t;
    typedef float io_t;
    typedef nn::params::Vectorf<1> params_t;
    typedef nn::Neuron<nn::PfWSum<io_t>,
                       nn::AfTanh<params_t>, io_t > neuron_t;
    typedef nn::Connection<weight_t, io_t> connection_t;
    typedef typename nn::NN<neuron_t, connection_t> nn_t;


protected :
    std::vector< std::vector<float> > _prevDesiredAngle;
    std::vector< std::vector<float> > _outputsample;
    std::vector<int> _brokenLegs;

    nn_t& _hnnctrl;
    std::vector<float> _input;

    std::vector<bool> _contact;
    std::vector< std::vector<float> > _prev_angles_actual;
    std::vector< std::vector<float> > _angles_actual;

private:

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


    ControllerPhase(nn_t& ctrl, std::vector<int> brokenLegs):_brokenLegs(brokenLegs), _hnnctrl(ctrl)
    {
        for (int leg=0;leg<6;leg++)
        {
            std::vector<float> param;
            param.push_back(0.0); // the initial desired angles of the joints is set to 0
            param.push_back(0.0);
            _prevDesiredAngle.push_back(param);  // previous reauested joint angle
            _outputsample.push_back(param);      // the outputs are accumulated here before subsampling

            _contact.push_back(true);
            std::vector<float> tmp(3, 0.0); //for the three oscillators on each leg
            _angles_actual.push_back(tmp);
            _prev_angles_actual.push_back(tmp);
        }
    }

    std::vector<float> moveRobot(robot_t& robot, float t, float step);
    std::vector<int> get_speeds_dyna( );
    std::vector<bool> get_directions_dyna( );
};

#endif
