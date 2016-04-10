//| This file is a part of the sferes2 framework.
//| Copyright 2009, ISIR / Universite Pierre et Marie Curie (UPMC)
//| Main contributor(s): Jean-Baptiste Mouret, mouret@isir.fr
//|
//| This software is a computer program whose purpose is to facilitate
//| experiments in evolutionary computation and evolutionary robotics.
//|
//| This software is governed by the CeCILL license under French law
//| and abiding by the rules of distribution of free software.  You
//| can use, modify and/ or redistribute the software under the terms
//| of the CeCILL license as circulated by CEA, CNRS and INRIA at the
//| following URL "http://www.cecill.info".
//|
//| As a counterpart to the access to the source code and rights to
//| copy, modify and redistribute granted by the license, users are
//| provided only with a limited warranty and the software's author,
//| the holder of the economic rights, and the successive licensors
//| have only limited liability.
//|
//| In this respect, the user's attention is drawn to the risks
//| associated with loading, using, modifying and/or developing or
//| reproducing the software by the user in light of its specific
//| status of free software, that may mean that it is complicated to
//| manipulate, and that also therefore means that it is reserved for
//| developers and experienced professionals having in-depth computer
//| knowledge. Users are therefore encouraged to load and test the
//| software's suitability as regards their requirements in conditions
//| enabling the security of their systems and/or data to be ensured
//| and, more generally, to use and operate it in the same conditions
//| as regards security.
//|
//| The fact that you are presently reading this means that you have
//| had knowledge of the CeCILL license and that you accept its terms.


#ifndef GEN_HNN_HPP_
#define GEN_HNN_HPP_

#include <boost/multi_array.hpp>
#include <boost/tr1/tuple.hpp>
#include <modules/nn2/neuron.hpp>
#include <modules/nn2/pf.hpp>
#include <modules/nn2/gen_dnn_ff.hpp>
#include <modules/nn2/trait.hpp>

#include "af_cppn.hpp"

#define InputRescaleFactor 3.0 // rescaling inputs to the network so that after applying the linear activation, they more fully utilize range of [-1, +1] and bias input is at 1

namespace sferes
{
namespace gen
{
template<typename W,
         typename Params,
         typename ParamsConns>
class Hnn
{
public:
    typedef DnnFF<nn::Neuron<nn::PfWSum<W>,
    nn::AfCppn<nn::cppn::AfParams<typename Params::cppn> > >,
    nn::Connection<W>,
    ParamsConns> cppn_t;
    typedef boost::tuple<float, float> point_t;
    typedef std::vector<point_t> layer_t;
    typedef std::vector<layer_t> substrate_t;

//#ifndef ORIENTFB
    typedef boost::tuple<float, float, float> temporalpoint_t;
/*#else
    typedef boost::tuple<float, float, float, float> temporalpoint_t;
#endif*/


    // we need this because we don't want to copy the substrate
    Hnn& operator = (const Hnn &o)
    {
        _cppn = o._cppn;
        return *this;
    }

    void init()
    {
        // force linear inputs. other activation functions applied directly on the inputs can result in loss of information
        typedef nn::cppn::AfParams<typename Params::cppn> afp_t;
        afp_t afparams;
        afparams.set(nn::cppn::linear, 0);
        for(size_t i = 0; i < _cppn.get_inputs().size(); ++i)
            _cppn.get_graph()[_cppn.get_inputs()[i]].set_afparams(afparams);

#ifdef CONSTRAINED_SUPG
        afp_t afparams1;
        afparams1.set(nn::cppn::copy, 0);
        _cppn.get_graph()[_cppn.get_outputs()[0]].set_afparams(afparams1); // the first output neuron is fixed at copy function. tanh applied in simu
#endif

        _develop(_cppn); // develop the CPPN (not the HNN)
        _cppn.init();

#ifndef ORIENTFB
        assert(_cppn.get_nb_inputs() == 4);  //x1 and y1, the position of the supg; time since last trigger event + bias input
#else
        assert(_cppn.get_nb_inputs() == 5); // all the previous four inputs plus the orientation heading f/b
#endif
        _create_substrate();
    }

    void random()
    {
        typename cppn_t::weight_t w;
        w.gen().data(0, 0.75f);//corresponds to 1 in range of weights from -2 to 2, because [0, 1] -> [-2, 2]

#ifndef ORIENTFB
        assert(ParamsConns::dnn::nb_inputs == 4);
#else
        assert(ParamsConns::dnn::nb_inputs == 5);
#endif

        assert(ParamsConns::dnn::nb_outputs == 3);
        _cppn.set_nb_inputs(ParamsConns::dnn::nb_inputs);
        _cppn.set_nb_outputs(ParamsConns::dnn::nb_outputs);

        // bias
        //w.gen().data(0, 0.5f);// correponds to 0 in range of weights from -2 to 2, because [0, 1] -> [-2, 2]
        w.random();
        _cppn.add_connection(_cppn.get_input(_cppn.get_nb_inputs() - 1), _cppn.get_output(0), w);
        w.random();
        _cppn.add_connection(_cppn.get_input(_cppn.get_nb_inputs() - 1), _cppn.get_output(1), w);
        w.random();
        _cppn.add_connection(_cppn.get_input(_cppn.get_nb_inputs() - 1), _cppn.get_output(2), w);


#ifdef CONSTRAINED_SUPG
        // all inputs are connected to all outputs
        // but the time input has four connections with preset weights to the output-0 governing servo angles.
        for (size_t i = 0; i < _cppn.get_nb_inputs() - 1; ++i)
            for (size_t j = 0; j < _cppn.get_nb_outputs(); ++j)
            {
                if((i == 2) && (j == 0)) //_cppn.get_nb_inputs() - 2
                {
                    w.gen().data(0, (M_PI/2.0 + 2.0)/4.0); // corresponds to pi/2 in range [-2, +2]
                    // add four connections each of weights pi/2 from the time input
                    _cppn.add_connection(_cppn.get_input(i), _cppn.get_output(j), w);
                    _cppn.add_connection(_cppn.get_input(i), _cppn.get_output(j), w);
                    _cppn.add_connection(_cppn.get_input(i), _cppn.get_output(j), w);
                    _cppn.add_connection(_cppn.get_input(i), _cppn.get_output(j), w);
                    continue;
                }

                w.random();
                //w.gen().data(0, 0.5f);// correponds to 0
                _cppn.add_connection(_cppn.get_input(i),_cppn.get_output(j), w);
            }
#else
        // connect the input and output neuros together
//        for (size_t i = 0; i < _cppn.get_nb_inputs() - 1; ++i) // all inputs except bias
//            for (size_t j = 0; j < _cppn.get_nb_outputs(); ++j)
//            {
//                w.random();
//                //w.gen().data(0, 0.5f);// correponds to 0
//                _cppn.add_connection(_cppn.get_input(i),_cppn.get_output(j), w);
//            }

        // all inputs are connected to all outputs
        // but the time input has four connections with preset weights to the output-0 governing servo angles.seeding the population like this may help evolution of gaits.
        for (size_t i = 0; i < _cppn.get_nb_inputs() - 1; ++i)
            for (size_t j = 0; j < _cppn.get_nb_outputs(); ++j)
            {
                if((i == 2) && (j == 0)) //_cppn.get_nb_inputs() - 2
                {
                    w.gen().data(0, (M_PI/2.0 + 2.0)/4.0); // corresponds to pi/2 in range [-2, +2]
                    // add four connections each of weights pi/2 from the time input
                    _cppn.add_connection(_cppn.get_input(i), _cppn.get_output(j), w);
                    _cppn.add_connection(_cppn.get_input(i), _cppn.get_output(j), w);
                    _cppn.add_connection(_cppn.get_input(i), _cppn.get_output(j), w);
                    _cppn.add_connection(_cppn.get_input(i), _cppn.get_output(j), w);
                    continue;
                }

                w.random();
                //w.gen().data(0, 0.5f);// correponds to 0
                _cppn.add_connection(_cppn.get_input(i),_cppn.get_output(j), w);
            }
#endif

#ifdef CONSTRAINED_SUPG
        for (size_t j = 1; j < _cppn.get_nb_outputs(); ++j) // the first output neuron is fixed at copy function. the remaining output functions are initialised at random
            _cppn.get_graph()[_cppn.get_outputs()[j]].get_afparams().random();
#else
        for (size_t j = 0; j < _cppn.get_nb_outputs(); ++j) // all the output functions are initialised at random
            _cppn.get_graph()[_cppn.get_outputs()[j]].get_afparams().random();
#endif
    }

    void mutate()
    {
        _cppn.mutate();
    }

    void cross(const Hnn& o, Hnn& c1, Hnn& c2)
    {
        if (misc::flip_coin())
        {
            c1 = *this;
            c2 = o;
        }
        else
        {
            c2 = *this;
            c1 = o;
        }
    }

    std::vector<float> query(const temporalpoint_t& p1, float normalized_heading = 0.0)
    {
#ifndef ORIENTFB
        std::vector<float> in(4);
        // rescaling inputs to the network so that after applying the linear activation, they more fully utilize range of [-1, +1] and bias input is at 1
        in[0] = p1.get<0>()  * InputRescaleFactor;
        in[1] = p1.get<1>()  * InputRescaleFactor;
        in[2] = p1.get<2>(); // Multiplying by InputRescaleFactor will be responsible for the freqiency at 3 Hz instead of 1 Hz.

        in[3] = 1.0 * InputRescaleFactor; //bias input
#else
        std::vector<float> in(5);
        // rescaling inputs to the network so that after applying the linear activation, they more fully utilize range of [-1, +1] and bias input is at 1
        in[0] = p1.get<0>()  * InputRescaleFactor;
        in[1] = p1.get<1>()  * InputRescaleFactor;

        in[2] = p1.get<2>();  // Multiplying by InputRescaleFactor will be responsible for the freqiency at 3 Hz instead of 1 Hz.

        in[3] = normalized_heading; //p1.get<3>()  * InputRescaleFactor;

        in[4] = 1.0 * InputRescaleFactor; //bias input
#endif


        for (size_t k = 0; k < _cppn.get_depth(); ++k)
            _cppn.step(in);

        return _cppn.get_outf();
    }

    template<class Archive>
    void serialize(Archive& ar, const unsigned int version)
    {
        ar& BOOST_SERIALIZATION_NVP(_cppn);
    }
    const cppn_t& cppn() const
    {
        return _cppn;
    }
    cppn_t& cppn()
    {
        return _cppn;
    }

    const cppn_t& returncppn() const
    {
        return _cppn;
    }

    template<typename NN>
    void write_cppndot(std::ostream& ofs, const NN& nn)
    {
        ofs << "digraph G {" << std::endl;
        BGL_FORALL_VERTICES_T(v, nn.get_graph(), typename NN::graph_t)
        {
            ofs << nn.get_graph()[v].get_id();
            ofs << " [label=\""<<nn.get_graph()[v].get_id() << "|" <<nn.get_graph()[v].get_afparams().type()<<"\"";

            if (nn.is_input(v) || nn.is_output(v))
                ofs<<" shape=doublecircle";

            ofs <<"]"<< std::endl;
        }
        BGL_FORALL_EDGES_T(e, nn.get_graph(), typename NN::graph_t)
        {
            ofs << nn.get_graph()[source(e, nn.get_graph())].get_id()
                << " -> " << nn.get_graph()[target(e, nn.get_graph())].get_id()
                << "[label=\"" << nn.get_graph()[e].get_weight() << "\"]" << std::endl;
        }
        ofs << "}" << std::endl;
    }


    const substrate_t& substrate() const { return _substrate; }
protected:
    cppn_t _cppn;
    substrate_t _substrate;

    void _create_substrate()
    {
        _substrate.clear();
        _substrate.resize(Params::hnn::nb_layers);

        assert(Params::hnn::nb_layers == 1);

        ////=============================================================================================================
        //Enter coordinates of the two SUPGs at each of the six legs in range [-1, +1]
        //Angle for third motor in antiphase of second one

        //         // Ken Stanley inspired geometry. Each leg on different Y coordinate, allowing the same offset value to be set for both motors on the leg
        //        //leg 0:
        //        _substrate[0].push_back(boost::make_tuple(-1, 0.03)); //servo 0
        //        _substrate[0].push_back(boost::make_tuple(+1, 0.03)); //servo 1

        //        //leg 1;
        //        _substrate[0].push_back(boost::make_tuple(-1, 0.02)); //servo 0
        //        _substrate[0].push_back(boost::make_tuple(+1, 0.02)); //servo 1

        //        //leg 2;
        //        _substrate[0].push_back(boost::make_tuple(-1, 0.01)); //servo 0
        //        _substrate[0].push_back(boost::make_tuple(+1, 0.01)); //servo 1

        //        //leg 3;
        //        _substrate[0].push_back(boost::make_tuple(-1, -0.01)); //servo 0
        //        _substrate[0].push_back(boost::make_tuple(+1, -0.01)); //servo 1

        //        //leg 4;
        //        _substrate[0].push_back(boost::make_tuple(-1, -0.02)); //servo 0
        //        _substrate[0].push_back(boost::make_tuple(+1, -0.02)); //servo 1

        //        //leg 5;
        //        _substrate[0].push_back(boost::make_tuple(-1, -0.03)); //servo 0
        //        _substrate[0].push_back(boost::make_tuple(+1, -0.03)); //servo 1

        ////=============================================================================================================
        //        // More hexapod like geometry. Offset only calculated for first servo of each leg (using both x and y values), and then applied to the second servo on the leg
        //leg 0;
        _substrate[0].push_back(boost::make_tuple(0.7*2.0-1.0,0.75*2.0-1.0)); //servo 0
        _substrate[0].push_back(boost::make_tuple(0.8*2.0-1.0,0.75*2.0-1.0)); //servo 1
        //_substrate[0].push_back(boost::make_tuple(0.9*2.0-1.0,0.75*2.0-1.0)); //servo 2

        //leg 1;
        _substrate[0].push_back(boost::make_tuple(0.7*2.0-1.0,0.5*2.0-1.0)); //servo 0
        _substrate[0].push_back(boost::make_tuple(0.8*2.0-1.0,0.5*2.0-1.0)); //servo 1
        //_substrate[0].push_back(boost::make_tuple(0.9*2.0-1.0,0.5*2.0-1.0)); //servo 2

        //leg 2;
        _substrate[0].push_back(boost::make_tuple(0.7*2.0-1.0,0.25*2.0-1.0)); //servo 0
        _substrate[0].push_back(boost::make_tuple(0.8*2.0-1.0,0.25*2.0-1.0)); //servo 1
        //_substrate[0].push_back(boost::make_tuple(0.9*2.0-1.0,0.25*2.0-1.0)); //servo 2

        //leg 3;
        _substrate[0].push_back(boost::make_tuple(0.3*2.0-1.0,0.25*2.0-1.0)); //servo 0
        _substrate[0].push_back(boost::make_tuple(0.2*2.0-1.0,0.25*2.0-1.0)); //servo 1
        //_substrate[0].push_back(boost::make_tuple(0.1*2.0-1.0,0.25*2.0-1.0)); //servo 2

        //leg 4;
        _substrate[0].push_back(boost::make_tuple(0.3*2.0-1.0,0.5*2.0-1.0)); //servo 0
        _substrate[0].push_back(boost::make_tuple(0.2*2.0-1.0,0.5*2.0-1.0)); //servo 1
        //_substrate[0].push_back(boost::make_tuple(0.1*2.0-1.0,0.5*2.0-1.0)); //servo 2

        //leg 5;
        _substrate[0].push_back(boost::make_tuple(0.3*2.0-1.0,0.75*2.0-1.0)); //servo 0
        _substrate[0].push_back(boost::make_tuple(0.2*2.0-1.0,0.75*2.0-1.0)); //servo 1
        //_substrate[0].push_back(boost::make_tuple(0.1*2.0-1.0,0.75*2.0-1.0)); //servo 2

        ////=============================================================================================================
    }

    template<typename NN>
    void _develop(NN& nn)
    {
        BGL_FORALL_EDGES_T(e, nn.get_graph(),
                           typename NN::graph_t)
                nn.get_graph()[e].get_weight().develop();
        BGL_FORALL_VERTICES_T(v, nn.get_graph(),
                              typename NN::graph_t)
        {
            nn.get_graph()[v].get_pfparams().develop();
            nn.get_graph()[v].get_afparams().develop();

        }
    }
};
}
}

#endif
