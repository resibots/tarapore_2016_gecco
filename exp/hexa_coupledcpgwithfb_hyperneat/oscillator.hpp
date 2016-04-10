#ifndef OSCILLATOR_HPP
#define OSCILLATOR_HPP

struct Oscillator
{
    std::vector<float>  amp_osc;
    std::vector<float>  dutyfactor_osc;

    //std::vector<float>  &initphase_osc;
    //std::vector<std::vector<float> >  &wts_couplings;

    std::vector<std::vector<float> >  phasebias_couplings;

    std::vector<std::vector<int> >  adjacency;

//    template<class Archive>
//    void serialize(Archive & ar, const unsigned int version)
//    {
//        dbg::trace trace("behavior", DBG_HERE);
//        //ar & BOOST_SERIALIZATION_NVP(this->_value);
//        //ar & BOOST_SERIALIZATION_NVP(this->_objs);


//        ar & BOOST_SERIALIZATION_NVP(position);
//        ar & BOOST_SERIALIZATION_NVP(drift);
//        ar & BOOST_SERIALIZATION_NVP(transferability);
//        ar & BOOST_SERIALIZATION_NVP(performance);
//        ar & BOOST_SERIALIZATION_NVP(arrivalangle);
//        ar & BOOST_SERIALIZATION_NVP(controller);
//        ar & BOOST_SERIALIZATION_NVP(features);
//        ar & BOOST_SERIALIZATION_NVP(dutycycles);
//    }
};

#endif // OSCILLATOR_HPP
