#ifndef CONTROLLERPHASE_HPP
#define CONTROLLERPHASE_HPP

#include <vector>
#include <boost/shared_ptr.hpp>
#include "robot/hexapod.hh"

#include <sferes/misc.hpp>

#include <fstream>

#include "oscillator.hpp"

class ControllerPhase
{

protected :
    std::vector< std::vector<float> > _prevAmp;
    std::vector< std::vector<float> > _prevdAmp;
    std::vector< std::vector<float> > _prevPhase;

    std::vector< std::vector<float> > _Amp;
    std::vector< std::vector<float> > _dAmp;
    std::vector< std::vector<float> > _Phase;

    std::vector< std::vector<float> > _angles;

    std::vector<int> _brokenLegs;
    Oscillator& _oscparams;

    std::vector<bool> _contact;
    std::vector<bool> _prev_contact;

    std::vector<std::vector<float> > _angles_actual;
    std::vector<std::vector<float> > _prev_angles_actual;

private:
    float cpg(size_t leg, size_t servo, float t);

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


    ControllerPhase(Oscillator& params, std::vector<int> brokenLegs):_brokenLegs(brokenLegs), _oscparams(params)
    {
        assert(_oscparams.amp_osc.size()==6*2);
        //assert(_oscparams.initphase_osc.size()==6*2);
        //assert(_oscparams.wts_couplings.size()==12*12);
        assert(_oscparams.phasebias_couplings.size()==12);
        assert(_oscparams.phasebias_couplings[0].size()==12);
        assert(_oscparams.adjacency.size()==12);
        assert(_oscparams.adjacency[0].size()==12);

        for (size_t i=0; i<_oscparams.amp_osc.size(); ++i)
        {
            size_t leg=10, servo=10;
            osc2legservo(i, &leg, &servo);
            assert(leg < 6); assert(servo < 2);
            assert(legservo2osc(leg,servo)==i);

            if(servo == 0)
                _oscparams.amp_osc[i] = (_oscparams.amp_osc[i] + 1.0) * M_PI / 16.0; // IntrinsicAmp in range [-1,+1] -> [0,pi/8]
            else //servo == 1
                _oscparams.amp_osc[i] = (_oscparams.amp_osc[i] + 1.0) * M_PI / 8.0; // IntrinsicAmp in range [-1,+1] -> [0,pi/4]

            //oscparams.initphase_osc[i] = (oscparams.initphase_osc[i] + 1.0) * M_PI; // InitPhase in range [-1,+1] -> [0, 2 pi]

            _oscparams.dutyfactor_osc[i] = (_oscparams.dutyfactor_osc[i] + 1.0f) / 2.0f; // in range [-1, +1] -> [0, 1]
            if(_oscparams.dutyfactor_osc[i] > 0.99 || _oscparams.dutyfactor_osc[i] < 0.01)
                _oscparams.dutyfactor_osc[i] = 0.5f;
        }

        for (size_t i=0; i<_oscparams.adjacency.size(); ++i)
            for (size_t j=0; j<_oscparams.adjacency[i].size() && _oscparams.adjacency[i][j]!=-1; ++j)
            {
                int k = _oscparams.adjacency[i][j];

                //assert(oscparams.wts_couplings[i][k] >= -1 && oscparams.wts_couplings[i][k] <= 1); // if not connected, init to -10000 (defined DEFAULTVALUE)
                assert(_oscparams.phasebias_couplings[i][k] >= -1 && _oscparams.phasebias_couplings[i][k] <= 1); // if not connected, init to -10000 (defined DEFAULTVALUE)

                //oscparams.wts_couplings[i][k] = (oscparams.wts_couplings[i][k] + 1.0) * 5.0; // weights in range [-1,+1] -> [0, 10]
                _oscparams.phasebias_couplings[i][k] = (_oscparams.phasebias_couplings[i][k] + 1.0) * M_PI; // phase bias in range [-1,+1] -> [0, 2 pi]
            }


        //recompute specific phase biases so that the sum of phase biases in closed loop inter-oscllator couplings is a multiple of 2 pi
        _oscparams.phasebias_couplings[4][0] = 2.0*M_PI -
                (_oscparams.phasebias_couplings[0][1]+_oscparams.phasebias_couplings[1][5]
                +_oscparams.phasebias_couplings[5][4]);
        _oscparams.phasebias_couplings[0][4] = -_oscparams.phasebias_couplings[4][0];

        _oscparams.phasebias_couplings[4][8] = 2.0*M_PI -
                (_oscparams.phasebias_couplings[8][9]+_oscparams.phasebias_couplings[9][5]+
                _oscparams.phasebias_couplings[5][4]);
        _oscparams.phasebias_couplings[8][4] = -_oscparams.phasebias_couplings[4][8];

        _oscparams.phasebias_couplings[2][1] = 2.0*M_PI -
                (_oscparams.phasebias_couplings[1][5]+_oscparams.phasebias_couplings[5][6]
                +_oscparams.phasebias_couplings[6][2]);
        _oscparams.phasebias_couplings[1][2] = -_oscparams.phasebias_couplings[2][1];

        _oscparams.phasebias_couplings[10][9] = 2.0*M_PI -
                (_oscparams.phasebias_couplings[9][5]+_oscparams.phasebias_couplings[5][6]
                +_oscparams.phasebias_couplings[6][10]);
        _oscparams.phasebias_couplings[9][10] = -_oscparams.phasebias_couplings[10][9];

        _oscparams.phasebias_couplings[3][7] = 2.0*M_PI -
                (_oscparams.phasebias_couplings[7][6]+_oscparams.phasebias_couplings[6][2]
                +_oscparams.phasebias_couplings[2][3]);
        _oscparams.phasebias_couplings[7][3] = -_oscparams.phasebias_couplings[3][7];

        _oscparams.phasebias_couplings[11][7] = 2.0*M_PI -
                (_oscparams.phasebias_couplings[7][6]+_oscparams.phasebias_couplings[6][10]
                +_oscparams.phasebias_couplings[10][11]);
        _oscparams.phasebias_couplings[7][11] = -_oscparams.phasebias_couplings[11][7];

        for (size_t leg=0;leg<6;leg++)
        {
            std::vector<float> tmp(2, 0.0); //for the two oscillators on each leg
            _prevAmp.push_back(tmp);
            _prevdAmp.push_back(tmp);
            _prevPhase.push_back(tmp);

            _Amp.push_back(tmp);
            _dAmp.push_back(tmp);
            _Phase.push_back(tmp);

            std::vector<float> tmp1(3, 0.0);
            _angles.push_back(tmp1);

            _prev_contact.push_back(true);
            _contact.push_back(true);

            std::vector<float> param1(3, 0.0);
            _angles_actual.push_back(param1);
            _prev_angles_actual.push_back(param1);
        }
    }

    size_t legservo2osc(size_t leg, size_t servo);
    void   osc2legservo(size_t osc, size_t* leg, size_t* servo);

    void moveRobot(robot_t& robot, float t);
    std::vector<int> get_pos_dyna(float t);
    std::vector<int> get_speeds_dyna( );
    std::vector<bool> get_directions_dyna( );
};

#endif
