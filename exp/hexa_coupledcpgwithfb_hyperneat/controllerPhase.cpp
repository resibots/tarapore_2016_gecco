#include "controllerPhase.hpp"
#define RAD2DYNMX28 651.42
#define RAD2DYN 195.57
#define EULERSTEPSIZE 0.02

#define INTRFREQ M_PI * 2.0
//#define INTRFREQ M_PI * 2.0 * 4.0 // 4 Hz

//#define DUTYRATIO 0.5 - its now queried from the cppn

#define GAIN Inf //instantaneous change in phase

void ControllerPhase::moveRobot(robot_t& robot, float t)
{    
    if(t==0)
    {   // since oscillators are coupled, we set the intial conditions for all of them, before starting any integration
        for(size_t leg=0;leg<6;++leg)
            for(size_t servo=0;servo<2;++servo)
            {
                _prevPhase[leg][servo] = sferes::misc::rand<float>() / 1000000.0; // [0, 0.001) //initial phase
                _prevAmp[leg][servo]   = _oscparams.amp_osc[legservo2osc(leg,servo)]; //start at the IntrinsicAmp, as we want amplitude to converge quickly
                _prevdAmp[leg][servo]  = 0.0;
            }
    }

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
    /*static std::ofstream actualjointanglesfs(std::string("output_actualjointangles.dat").c_str());
    actualjointanglesfs << t << " ";
    tmp_leg = 0;
    for (size_t i = 0; i < robot->servos().size(); i+=3)
    {
        for (int j=0;j<_brokenLegs.size();j++)
        {
            if (tmp_leg==_brokenLegs[j])
            {
                tmp_leg++;
                if (_brokenLegs.size()>j+1 && _brokenLegs[j+1]!=tmp_leg)
                    break;
            }
        }
        actualjointanglesfs << " " << _contact[tmp_leg] << " " << robot->servos()[i]->get_angle(0) << " " << _contact[tmp_leg] << " " << robot->servos()[i+1]->get_angle(0);
        _prev_angles_actual[tmp_leg][0] = _angles_actual[tmp_leg][0];
        _prev_angles_actual[tmp_leg][1] = _angles_actual[tmp_leg][1];
        _prev_angles_actual[tmp_leg][2] = _angles_actual[tmp_leg][2];
        _angles_actual[tmp_leg][0] = robot->servos()[i]->get_angle(0);
        _angles_actual[tmp_leg][1] = robot->servos()[i+1]->get_angle(0);
        _angles_actual[tmp_leg][2] = robot->servos()[i+2]->get_angle(0);
        ++tmp_leg;
    }
    actualjointanglesfs << std::endl;
    static std::ofstream leglogsfs(std::string("leglogs.dat").c_str());
    leglogsfs << t << " ";
    for(size_t leg = 0; leg <= 2; ++leg)
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
    leglogsfs << std::endl;*/


    for(size_t leg = 0; leg < 6; ++leg)
    {
        _angles[leg][0] = cpg(leg, 0, t);
        _angles[leg][1] = cpg(leg, 1, t);
        _angles[leg][2] = cpg(leg, 2, t);
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

//        robot->servos()[i]->set_angle(0, cpg(leg, 0, t));
//        robot->servos()[i+1]->set_angle(0, cpg(leg, 1, t));
//        robot->servos()[i+2]->set_angle(0, cpg(leg, 2, t));

        robot->servos()[i]->set_angle(0, _angles[leg][0]);
        robot->servos()[i+1]->set_angle(0, _angles[leg][1]);
        robot->servos()[i+2]->set_angle(0, _angles[leg][2]);

        ++leg;
    }

    for(size_t leg=0;leg<6;++leg)
    {
        for(size_t servo=0;servo<2;++servo)
        {
            _prevPhase[leg][servo] = _Phase[leg][servo];//redundant
            _prevPhase[leg][servo] = fmod(_Phase[leg][servo], 2.0*M_PI);
            _prevAmp[leg][servo]   = _Amp[leg][servo];
            _prevdAmp[leg][servo]  = _dAmp[leg][servo];
        }
        _prev_contact[leg] = _contact[leg];
    }

    //Printing the output of the cpg for plotting
    /*static std::ofstream ofs(std::string("output_coupledcpg.dat").c_str());
    ofs << t << " ";
    for (size_t leg = 0; leg < 6; ++leg)
        for(size_t servo = 0; servo < 2; ++servo)
            ofs << " " << 0.0 + _Amp[leg][servo] * cos(_Phase[leg][servo]);
    ofs << std::endl;*/
}

// used to compute the output of the CPG signal for each oscillator
//float ControllerPhase::cpg(size_t leg, size_t servo, float IntrinsicAmp, float InitPhase, float t)
float ControllerPhase::cpg(size_t leg, size_t servo, float t)
{
    assert(leg < 6); assert(servo < 3);

    float Offset = 0.0;

    if(servo == 2)  //the third servo has signal in antiphase with second servo, and so minus sign to its amplitude
         return -Offset - _Amp[leg][servo-1] * cos(_Phase[leg][servo-1]);


    float DUTYRATIO = _oscparams.dutyfactor_osc[legservo2osc(leg,servo)];



    size_t oscindex = legservo2osc(leg,servo);
    float normIntrinsicAmp = _oscparams.amp_osc[oscindex];
    float alpha  = 10.0; // convergence rate to intrinsic amplitude

    _Amp[leg][servo]   = _prevAmp[leg][servo]   + EULERSTEPSIZE * _prevdAmp[leg][servo];
    _dAmp[leg][servo]  = _prevdAmp[leg][servo]  + EULERSTEPSIZE * alpha*(alpha/4.0*(normIntrinsicAmp - _prevAmp[leg][servo]) - _prevdAmp[leg][servo]);


    float w = 20.0; //coupling weights
    float totalbias = 0.0;
    for(int i = 0; i < _oscparams.adjacency[oscindex].size() && _oscparams.adjacency[oscindex][i] != -1; ++i) //max num of neighbouring coupled oscillators is 4
    {
        int nbroscindex = _oscparams.adjacency[oscindex][i];
        size_t l,s; osc2legservo((size_t)nbroscindex,&l,&s); assert(l < 6); assert(s < 2);

        totalbias += _prevAmp[l][s] * w * sin(_prevPhase[l][s] - _prevPhase[leg][servo] - _oscparams.phasebias_couplings[oscindex][nbroscindex]);
    }

    _Phase[leg][servo] = _prevPhase[leg][servo] + EULERSTEPSIZE * (INTRFREQ + totalbias);


    if(_prev_contact[leg] == false && _contact[leg] == true) // foot has just touched the ground
    {
        if((leg == 0 || leg == 1 || leg == 2) && servo == 0)
        {
            float _Phase_AEP = 2.0*M_PI*(1.0 - DUTYRATIO);
            float _Phase_PEP = 0.0;
            //_Phase[leg][servo] = _prevPhase[leg][servo] + EULERSTEPSIZE * (INTRFREQ + totalbias + (_Phase_AEP - _prevPhase[leg][servo]));
            _Phase[leg][servo] = _Phase_AEP;
        }

        if((leg == 3 || leg == 4 || leg == 5) && servo == 0)
        {
            float _Phase_AEP = 0.0;
            float _Phase_PEP = 2.0*M_PI*(1.0 - DUTYRATIO);
            //_Phase[leg][servo] = _prevPhase[leg][servo] + EULERSTEPSIZE * (INTRFREQ + totalbias + (_Phase_AEP - _prevPhase[leg][servo]));
            _Phase[leg][servo] = _Phase_AEP;
        }

        if (servo == 1)
        {
            /*enforce the evolved phase difference*/
            float phasediff = _oscparams.phasebias_couplings[legservo2osc(leg,servo-1)][oscindex];
            float _Phase_AEP = fmod(2.0*M_PI*(1.0 - DUTYRATIO) + phasediff, 2.0 * M_PI); //M_PI/2.0;
            float _Phase_PEP = fmod(0.0 + phasediff, 2.0 * M_PI);// M_PI/2.0;

            /*float _Phase_AEP = fmod(2.0*M_PI*(1.0 - DUTYRATIO) + M_PI/2.0, 2.0 * M_PI); //M_PI/2.0;
            float _Phase_PEP = fmod(0.0 + M_PI/2.0, 2.0 * M_PI);// M_PI/2.0;*/
            //_Phase[leg][servo] = _prevPhase[leg][servo] + EULERSTEPSIZE * (INTRFREQ + totalbias + GAIN*(_Phase_AEP - _prevPhase[leg][servo]));
            //_Phase[leg][servo] = _Phase_AEP;
        }
    }

    if(_prev_contact[leg] == true && _contact[leg] == false) // foot has just lifted off the ground
    {
        if((leg == 0 || leg == 1 || leg == 2) && servo == 0)
        {
            float _Phase_AEP = 2.0*M_PI*(1.0 - DUTYRATIO);
            float _Phase_PEP = 0.0;
            //_Phase[leg][servo] = _prevPhase[leg][servo] + EULERSTEPSIZE * (INTRFREQ + totalbias + GAIN*(_Phase_PEP - _prevPhase[leg][servo]));
            //_Phase[leg][servo] = _Phase_PEP; /* lets start by resetting only at AEP*/
        }

        if((leg == 3 || leg == 4 || leg == 5) && servo == 0)
        {
            float _Phase_AEP = 0.0;
            float _Phase_PEP = 2.0*M_PI*(1.0 - DUTYRATIO);
            //_Phase[leg][servo] = _prevPhase[leg][servo] + EULERSTEPSIZE * (INTRFREQ + totalbias + GAIN*(_Phase_PEP - _prevPhase[leg][servo]));
            //_Phase[leg][servo] = _Phase_PEP;  /* lets start by resetting only at AEP*/
        }

        if (servo == 1)
        {
            /*enforce the evolved phase difference*/
            float phasediff = _oscparams.phasebias_couplings[legservo2osc(leg,servo-1)][oscindex];
            float _Phase_AEP = fmod(2.0*M_PI*(1.0 - DUTYRATIO) + phasediff, 2.0 * M_PI); //M_PI/2.0;
            float _Phase_PEP = fmod(0.0 + phasediff, 2.0 * M_PI);// M_PI/2.0;

            /*float _Phase_AEP = fmod(2.0 * M_PI * (1.0 - DUTYRATIO) + M_PI / 2.0, 2.0 * M_PI); //M_PI/2.0;
            float _Phase_PEP = fmod(0.0 + M_PI / 2.0, 2.0 * M_PI);// M_PI/2.0;*/
            //_Phase[leg][servo] = _prevPhase[leg][servo] + EULERSTEPSIZE * (INTRFREQ + totalbias + GAIN*(_Phase_PEP - _prevPhase[leg][servo]));
            //_Phase[leg][servo] = _Phase_PEP;
        }
    }

    //_Phase[leg][servo] = fmod(_Phase[leg][servo], 2.0*M_PI);

    /*if(leg == 0 && servo == 0)
        std::cout << _Phase[leg][servo] << " " << _Phase[leg][servo+1] << " " << _contact[leg] << std::endl;*/


    /*static std::ofstream ofs(std::string("instfreq_coupledcpg.dat").c_str());
    if(leg==0 && servo==0)
    {
        ofs << std::endl;
        ofs << t << " ";
    }
    ofs << (INTRFREQ + totalbias)/(2.0 * M_PI) << " ";*/

    return Offset + _Amp[leg][servo] * cos(_Phase[leg][servo]);
}

// Returns the index of the oscillator on the substrate for a give leg and servo
size_t ControllerPhase::legservo2osc(size_t leg, size_t servo)
{
    assert(leg < 6); assert(servo < 2);

    if(leg==0 && servo==0)
        return 2;
    else if(leg==0 && servo==1)
        return 3;
    else if(leg==1 && servo==0)
        return 6;
    else if(leg==1 && servo==1)
        return 7;
    else if(leg==2 && servo==0)
        return 10;
    else if(leg==2 && servo==1)
        return 11;
    else if(leg==3 && servo==0)
        return 9;
    else if(leg==3 && servo==1)
        return 8;
    else if(leg==4 && servo==0)
        return 5;
    else if(leg==4 && servo==1)
        return 4;
    else if(leg==5 && servo==0)
        return 1;
    else if(leg==5 && servo==1)
        return 0;
    else
    {
        std::cout << std::endl << "leg " << leg << " servo " << servo << " not in range" << std::endl;
        return -1;
    }
}

// Returns the index of the leg and servo for the oscillator on the substrate
void ControllerPhase::osc2legservo(size_t osc, size_t* leg, size_t* servo)
{
    switch (osc)
    {
    case 0:
        (*leg) = 5; (*servo) = 1; return;
    case 1:
        (*leg) = 5; (*servo) = 0; return;
    case 2:
        (*leg) = 0; (*servo) = 0; return;
    case 3:
        (*leg) = 0; (*servo) = 1; return;
    case 4:
        (*leg) = 4; (*servo) = 1; return;
    case 5:
        (*leg) = 4; (*servo) = 0; return;
    case 6:
        (*leg) = 1; (*servo) = 0; return;
    case 7:
        (*leg) = 1; (*servo) = 1; return;
    case 8:
        (*leg) = 3; (*servo) = 1; return;
    case 9:
        (*leg) = 3; (*servo) = 0; return;
    case 10:
        (*leg) = 2; (*servo) = 0; return;
    case 11:
        (*leg) = 2; (*servo) = 1; return;
    default:
        (*leg) = 25; (*servo) = 25;
    }
}


