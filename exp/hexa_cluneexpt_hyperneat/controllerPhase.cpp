#include "controllerPhase.hpp"
#define RAD2DYNMX28 651.42
#define RAD2DYN 195.57

std::vector<float>  ControllerPhase::moveRobot(robot_t& robot, float t, float step)
{
    // recording angles output by cppn separately for the FFT
    std::vector<float> param1(18, 0.0);

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
        if(_prev_angles_actual[leg][0] == _angles_actual[leg][0]) // && _prev_angles_actual[leg][0] == 0
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


    /*for (size_t leg = 0; leg < 6; ++leg)
    {
        for(size_t sensor = 0; sensor < 3; ++sensor) //angles of three servo joints
            _input.push_back(robot->servos()[leg*3 + sensor]->get_angle(0));
        _input.push_back(robot->bodies()[leg*3 + 3]->get_in_contact());

        if(leg == 0)
            _input.push_back(robot->rot()[0]); //pitch or roll?
        else if(leg == 1)
            _input.push_back(robot->rot()[1]); //roll or pitch?
        else if(leg == 2)
            _input.push_back(robot->rot()[2]); //heading
        else if(leg == 3)
            _input.push_back(M_PI/2 * sin(2*M_PI*t)); //sine wave with amplitude M_PI/2 (joint angles are renormalized in output based on position on segment). Frequency of 1 Hz
    }*/

    for (size_t leg = 0; leg < 6; ++leg)
        for(size_t sensor = 0; sensor < 2; ++sensor)
            _outputsample[leg][sensor] = 0.0;

    float numsamples = 4.0; //1.0;
    for(float sample=0; sample < numsamples; sample+=1.0)
    {
        _input.clear();
        for (size_t leg = 0; leg < 6; ++leg)
        {
            for(size_t sensor = 0; sensor < 2; ++sensor) //previously desired angles of first two servo joints
                _input.push_back(_prevDesiredAngle[leg][sensor]);

            if(leg == 0)
                _input.push_back(sin(2*M_PI*(t+sample*step/numsamples))); //sine: with amplitude +1. Frequency of 1 Hz
            else if(leg == 1)
                _input.push_back(cos(2*M_PI*(t+sample*step/numsamples))); //cos: with amplitude +1. Frequency of 1 Hz
        }
        assert(_input.size() == _hnnctrl.get_nb_inputs());

        for (size_t i = 0; i < 2; ++i)
          _hnnctrl.step(_input);

        size_t outputindex = 0;
        for (size_t leg = 0; leg < 6; ++leg)
            for(size_t sensor = 0; sensor < 2; ++sensor) //current desired angles of first two servo joints of each leg, stored for next control cycle
            {
               _prevDesiredAngle[leg][sensor] = _hnnctrl.get_outf(outputindex++);
               _outputsample[leg][sensor]    += _prevDesiredAngle[leg][sensor];
            }
    }

    for (size_t leg = 0; leg < 6; ++leg)
        for(size_t sensor = 0; sensor < 2; ++sensor)
            _outputsample[leg][sensor] /= numsamples;


    //Printing the output of the neural network for plotting
    /*static std::ofstream ofs(std::string("output_hnn.dat").c_str());
    ofs << t << " ";
    for (size_t leg = 0; leg < 6; ++leg)
        for(size_t sensor = 0; sensor < 2; ++sensor)
            ofs << " " << _outputsample[leg][sensor];
    ofs << std::endl;*/


    size_t leg = 0;
    //size_t outputindex = 0;
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

//        robot->servos()[i]->set_angle(0,   M_PI/8 * _outputsample[(int)((float)i/3.0)][0]); // nn output in range -1 to 1, normalized to [-pi/8,pi/8];
//        robot->servos()[i+1]->set_angle(0, M_PI/4 * _outputsample[(int)((float)i/3.0)][1]); //  nn output in range -1 to 1, normalized to [-pi/4,pi/4];
//        //The third servo moves at an opposite but equal angle to the second servo, to allow for improved stability in stance
//        robot->servos()[i+2]->set_angle(0,-M_PI/4 * _outputsample[(int)((float)i/3.0)][1]); // antiphase to servo 2 signal

        robot->servos()[i]->set_angle(0,   M_PI/8 * _outputsample[leg][0]); // nn output in range -1 to 1, normalized to [-pi/8,pi/8];
        robot->servos()[i+1]->set_angle(0, M_PI/4 * _outputsample[leg][1]); //  nn output in range -1 to 1, normalized to [-pi/4,pi/4];
        //The third servo moves at an opposite but equal angle to the second servo, to allow for improved stability in stance
        robot->servos()[i+2]->set_angle(0,-M_PI/4 * _outputsample[leg][1]); // antiphase to servo 2 signal

        unsigned servo = 0;
        param1[leg * 3 + servo] =  M_PI/8 * _outputsample[leg][0];
        servo = 1;
        param1[leg * 3 + servo] =  M_PI/4 * _outputsample[leg][1];

        ++leg;
    }

    return param1;
}
