#include "controllerPhase.hpp"
#define RAD2DYNMX28 651.42
#define RAD2DYN 195.57
#define EULERSTEPSIZE 0.02

#define INTRFREQ M_PI * 2.0
//#define INTRFREQ M_PI * 2.0 * 4.0 //intrinsic frequency  //4 Hz

void ControllerPhase::moveRobot(robot_t& robot, float t)
{
    /******************* Logs BEGIN *************************************/
    size_t tmp_leg = 0;
    for(size_t funclastlegsegment = 3; funclastlegsegment < robot->bodies().size(); funclastlegsegment+=3)
    {
        for (int j=0;j<_brokenLegs.size();j++)
        {
            if (tmp_leg==_brokenLegs[j])
            {
                _contact[tmp_leg] = false;
                tmp_leg++;
                if (_brokenLegs.size()>j+1 && _brokenLegs[j+1]!=tmp_leg)
                    break;
            }
        }
        _contact[tmp_leg] = robot->bodies()[funclastlegsegment]->get_in_contact();
        ++tmp_leg;
    }
    static std::ofstream ofs1(std::string("output_actualjointangles.dat").c_str());
    ofs1 << t << " ";
    size_t leg1 = 0;
    for (size_t i = 0; i < robot->servos().size(); i+=3)
    {
        for (int j=0;j<_brokenLegs.size();j++)
        {
            if (leg1==_brokenLegs[j])
            {
                leg1++;
                if (_brokenLegs.size()>j+1 && _brokenLegs[j+1]!=leg1)
                    break;
            }
        }

        ofs1 << " " << _contact[leg1] << " " << robot->servos()[i]->get_angle(0) << " " << _contact[leg1] << " " << robot->servos()[i+1]->get_angle(0);
        _prev_angles_actual[leg1][0] = _angles_actual[leg1][0];
        _prev_angles_actual[leg1][1] = _angles_actual[leg1][1];
        _prev_angles_actual[leg1][2] = _angles_actual[leg1][2];
        _angles_actual[leg1][0] = robot->servos()[i]->get_angle(0);
        _angles_actual[leg1][1] = robot->servos()[i+1]->get_angle(0);
        _angles_actual[leg1][2] = robot->servos()[i+2]->get_angle(0);
        ++leg1;
    }
    ofs1 << std::endl;
    static std::ofstream leglogsfs(std::string("leglogs.dat").c_str());
    leglogsfs << t << " ";
    for(size_t leg = 0; leg <= 2; ++leg) /* Warning: Code will break when legs are removed */
    {
        if(_prev_angles_actual[leg][0] == _angles_actual[leg][0]) //&& _prev_angles_actual[leg][0] == 0
            leglogsfs << "0"  << " " << _contact[leg] << " ";  //std::cout << "NS";
        else if(_angles_actual[leg][0] < _prev_angles_actual[leg][0])
            leglogsfs << "-1"  << " " << _contact[leg] << " ";  //std::cout << "Swing phase" << std::endl;
        else if(_angles_actual[leg][0] > _prev_angles_actual[leg][0])
            leglogsfs << "+1"  << " " << _contact[leg] << " ";  //std::cout << "Stance phase" << std::endl;
    }
    for(size_t leg = 3; leg <= 5; ++leg)
    {
        if(_prev_angles_actual[leg][0] == _angles_actual[leg][0]) // && _prev_angles_actual[leg][0] == 0
            leglogsfs << "0"  << " " << _contact[leg] << " ";   //std::cout << "NS";
        else if(_angles_actual[leg][0] > _prev_angles_actual[leg][0])
            leglogsfs << "-1"  << " " << _contact[leg] << " ";  //std::cout << "Swing phase" << std::endl;
        else if(_angles_actual[leg][0] < _prev_angles_actual[leg][0])
            leglogsfs << "+1"  << " " << _contact[leg] << " ";  //std::cout << "Stance phase" << std::endl;
    }
    leglogsfs << std::endl;
    /******************* Logs END *************************************/


    for(size_t leg=0; leg < 6; ++leg)
    {
        _angles[leg][0] = cpg(leg, 0, _legsParams[leg][0], _legsParams[leg][1], t);
        _angles[leg][1] = cpg(leg, 1, _legsParams[leg][2], _legsParams[leg][3], t);
        _angles[leg][2] = cpg(leg, 2, _legsParams[leg][4], _legsParams[leg][5], t);
    }

    size_t leg = 0;
    for (size_t i = 0; i < robot->servos().size(); i+=3)
    {
        for (int j=0;j<_brokenLegs.size();j++)
        {
            if (leg==_brokenLegs[j])
            {
                leg++;
                if (_brokenLegs.size()>j+1 && _brokenLegs[j+1]!=leg)
                    break;
            }
        }

//        robot->servos()[i]->set_angle(0, cpg(leg, 0, _legsParams[leg][0], _legsParams[leg][1], t));
//        robot->servos()[i+1]->set_angle(0, cpg(leg, 1, _legsParams[leg][2], _legsParams[leg][3], t));
//        robot->servos()[i+2]->set_angle(0, cpg(leg, 2, _legsParams[leg][4], _legsParams[leg][5], t));

        robot->servos()[i]->set_angle(0, _angles[leg][0]);
        robot->servos()[i+1]->set_angle(0, _angles[leg][1]);
        robot->servos()[i+2]->set_angle(0, _angles[leg][2]);

        ++leg;
    }
}


//!Constants convert leg desired position from degrees to encoder positions for RAD2DYN (ax18 motors 0-1024) and RAD2DYNMX28 (mx28 motors 0-4096)
std::vector<int> ControllerPhase::get_pos_dyna(float t)
{
    std::vector<int> pos;

    size_t leg = 0;
    for (size_t i = 0; i < 24; i+=4)
    {
        for (int j=0;j<_brokenLegs.size();j++)
        {
            if (leg==_brokenLegs[j])
            {
                leg++;
                if (_brokenLegs.size()>j+1 && _brokenLegs[j+1]!=leg)
                    break;
            }
        }
        //servo 0
        float theta0=_legsParams[leg][0]*M_PI/8+ _legsParams[leg][1]*M_PI/8*delayedPhase(t,_legsParams[leg][2]);

        if(leg==0 ||leg ==3) //Offsets, eg. 64 are because of misalignment of encoder center, and is only for the last two motors (servos 2 and 3). The first motor is directly screwed to the robot (no gear placement needed), and therfore doesnot need calibration
            pos.push_back(512-64-RAD2DYN*(theta0));
        else if(leg == 2 || leg==5)
            pos.push_back(512+64-RAD2DYN*(theta0));
        else
            pos.push_back(512-RAD2DYN*(theta0));

        //servo 1
        //setting an offset for mx28s which have bad zero
        float theta1=_legsParams[leg][3]*M_PI/4+_legsParams[leg][4]*M_PI/4*delayedPhase(t,_legsParams[leg][5]);

        if (leg==2)
            pos.push_back(2048+50+RAD2DYNMX28*(theta1));
        else if (leg==3)
            pos.push_back(2048-150+RAD2DYNMX28*(theta1));
        else if (leg == 1)
            pos.push_back(2048-200+RAD2DYNMX28*(theta1));
        else
            pos.push_back(2048+RAD2DYNMX28*(theta1));


        //dont need to change sign of signal between second and third motor, because of the inverted motor positions

        //servo 2
        float theta2=-_legsParams[leg][3]*M_PI/4-_legsParams[leg][4]*M_PI/4*delayedPhase(t,_legsParams[leg][6]);

        if(leg==0)
            pos.push_back(2048-300-RAD2DYNMX28*(theta2));
        else
            pos.push_back(2048-RAD2DYNMX28*(theta2));

        ++leg;
    }
    return pos;
}


// used to compute square wave with phase delay phi
float ControllerPhase::delayedPhase(float t, float phi)
{
    //tanh applied to phase shifted sinusoidal wave. Tanh applied to give a square wave, with smooth transitions
    return tanh(sin(2*M_PI*t+phi*2*M_PI)*4);
}


// used to compute the output of the CPG signal for each oscillator
float ControllerPhase::cpg(size_t leg, size_t servo, float IntrinsicAmp, float InitPhase, float t)
{
    float Offset = 0.0;
    float alpha  = 10.0; // convergence rate to intrinsic amplitude

    float normIntrinsicAmp;

    if(servo==0)
        normIntrinsicAmp = (IntrinsicAmp + 1.0) * M_PI / 16.0; // IntrinsicAmp in range [-1,+1] -> [0,pi/8]
    else //servo 1 and 2
        normIntrinsicAmp = (IntrinsicAmp + 1.0) * M_PI / 8.0; // IntrinsicAmp in range [-1,+1] -> [0,pi/4]

    float normInitPhase    = (InitPhase + 1.0) * M_PI; // InitPhase in range [-1,+1] -> [0, 2 pi]

    assert(leg < 6); assert(servo < 3);
    assert(normIntrinsicAmp >= 0.0 && normIntrinsicAmp <= M_PI/2.0 + 1.0e+6);
    assert(normInitPhase >= 0.0 && normInitPhase <= M_PI * 2.0 + 1.0e+6);

    if(t==0.0)
    {
        _prevPhase[leg][servo] = normInitPhase;
        _prevAmp[leg][servo]   = normIntrinsicAmp; //start at the IntrinsicAmp, or do we really want to see convergence
        _prevdAmp[leg][servo]  = 0.0;
    }

    float Phase = _prevPhase[leg][servo] + EULERSTEPSIZE * INTRFREQ;
    float Amp   = _prevAmp[leg][servo]   + EULERSTEPSIZE * _prevdAmp[leg][servo];
    float dAmp  = _prevdAmp[leg][servo]  + EULERSTEPSIZE * alpha*(alpha/4.0*(normIntrinsicAmp - _prevAmp[leg][servo]) - _prevdAmp[leg][servo]);

    _prevPhase[leg][servo] = Phase;
    _prevAmp[leg][servo]   = Amp;
    _prevdAmp[leg][servo]  = dAmp;

    if(servo == 2) //the third servo has signal in antiphase with second servo, and so minus sign to its amplitude
        //return -Offset - Amp * tanh(sin(Phase)*4);
        return -Offset - Amp * cos(Phase);
    else
        //return Offset + Amp * tanh(sin(Phase)*4);
         return Offset + Amp * cos(Phase);

    //return Offset + Amp * tanh(cos(Phase)*4);
    //tanh(sin(2*M_PI*t+phi*2*M_PI)*4)
}
