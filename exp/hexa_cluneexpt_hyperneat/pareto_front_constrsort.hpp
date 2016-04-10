#ifndef PARETO_FRONT_HPP_
#define PARETO_FRONT_HPP_

#include <boost/serialization/shared_ptr.hpp>
#include <boost/serialization/nvp.hpp>
#include <sferes/stc.hpp>
#include <sferes/parallel.hpp>
#include <sferes/fit/fitness.hpp>
#include <sferes/stat/stat.hpp>

#define ANGLE_THRESHOLD 1.0 //DEGREES

namespace sferes
{
namespace stat
{
SFERES_STAT(ParetoFrontConstraintSort, Stat)
{
    public:
    typedef std::vector<boost::shared_ptr<Phen> > pareto_t;
    // asume a ea.pareto_front() method
    template<typename E>
    void refresh(const E& ea)
    {
        _pareto_front = ea.pareto_front();

        parallel::sort(_pareto_front.begin(), _pareto_front.end(),
                       fit::compare_obj(1)); //sort based on arrival angle

        int index = _pareto_front.size();
        for (size_t i =0; i < _pareto_front.size(); ++i)
            if(_pareto_front[i]->fit().obj(1) < -ANGLE_THRESHOLD) //solutions with arrival angle exceeding 5 degrees are not in the subset
                {index = i; break;}

        //if(index != -1)
            parallel::sort(_pareto_front.begin(), _pareto_front.begin()+index,
                           fit::compare_objs_lex()); //sort the resultant subset on forward displacement
        /*else
            parallel::sort(_pareto_front.begin(), _pareto_front.end(),
                           fit::compare_objs_lex());*/


        this->_create_log_file(ea, "pareto.dat");
        if (ea.dump_enabled())
            show_all(*(this->_log_file), ea.gen());
        //this->_log_file->close();
    }
    void show(std::ostream& os, size_t k) const
    {
        os<<"log format : gen id obj_1 ... obj_n"<<std::endl;
        show_all(os, 0);

        //_pareto_front[k]->mutate(); //Mutation to assess evolvability

        _pareto_front[k]->develop();
        _pareto_front[k]->show(os);
        _pareto_front[k]->fit().set_mode(fit::mode::view);
        _pareto_front[k]->fit().eval(*_pareto_front[k]);
        os << "=> displaying individual " << k << std::endl;
        os << "fit:";
        for (size_t i =0; i < _pareto_front[k]->fit().objs().size(); ++i)
            os << _pareto_front[k]->fit().obj(i) << " ";
        os << std::endl;
        assert(k < _pareto_front.size());

    }
    const pareto_t& pareto_front() const { return _pareto_front; }
    template<class Archive>
    void serialize(Archive & ar, const unsigned int version)
    {
        ar & BOOST_SERIALIZATION_NVP(_pareto_front);
    }
    void show_all(std::ostream& os, size_t gen = 0) const
    {
        for (unsigned i = 0; i < _pareto_front.size(); ++i)
        {
            os << gen << " " << i << " ";
            for (unsigned j = 0; j < _pareto_front[i]->fit().objs().size(); ++j)
                os << _pareto_front[i]->fit().obj(j) << " ";
            os << std::endl;;
        }
    }

    protected:
    pareto_t _pareto_front;
};
}
}
#endif
