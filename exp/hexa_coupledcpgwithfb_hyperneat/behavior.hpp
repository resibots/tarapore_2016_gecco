#ifndef BEHAVIOR_HPP
#define BEHAVIOR_HPP


struct Behavior
{
    std::vector<float> position;
    float drift;
    float transferability;
    float performance;
    float direction;
    float arrivalangle;

    std::vector<float>  features;
    std::vector<float>  dutycycles;

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
#endif
