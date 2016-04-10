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

        _develop(_cppn); // develop the CPPN (not the HNN)
        _cppn.init();
        assert(_cppn.get_nb_inputs() == 5); //x1 and y1, x2 and y2, the positions of the source and target HNN nodes + bias input
        _create_substrate();
    }

    void random()
    {
        typedef nn::cppn::AfParams<typename Params::cppn> afp_t;
        afp_t afparams;
        afparams.set(nn::cppn::gaussian, 0);

        typename cppn_t::weight_t w;
        w.gen().data(0, 0.75f);//corresponds to 1

        assert(ParamsConns::dnn::nb_inputs == 5);
        assert(ParamsConns::dnn::nb_outputs == 2);
        _cppn.set_nb_inputs(ParamsConns::dnn::nb_inputs);
        _cppn.set_nb_outputs(ParamsConns::dnn::nb_outputs);

        // bias
        //w.gen().data(0, 0.5f);// correponds to 0
        w.random();_cppn.add_connection(_cppn.get_input(4), _cppn.get_output(0), w);
        w.random();_cppn.add_connection(_cppn.get_input(4), _cppn.get_output(1), w);

        // connect the other I/O together, that are not connected through the hidden neurons above
        for (size_t i = 0; i < _cppn.get_nb_inputs() - 1; ++i) // all inputs except bias
            for (size_t j = 0; j < _cppn.get_nb_outputs(); ++j)
            {
                w.random(); _cppn.add_connection(_cppn.get_input(i), _cppn.get_output(j), w);
            }

        for (size_t j = 0; j < _cppn.get_nb_outputs(); ++j)
            _cppn.get_graph()[_cppn.get_outputs()[j]].get_afparams().random();
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

    std::vector<float> query(const point_t& p1, const point_t& p2)
    {
        std::vector<float> in(5);
        // rescaling inputs to the network so that after applying the linear activation, they more fully utilize range of [-1, +1] and bias input is at 1
        in[0] = p1.get<0>() * InputRescaleFactor;
        in[1] = p1.get<1>() * InputRescaleFactor;

        in[2] = p2.get<0>() * InputRescaleFactor;
        in[3] = p2.get<1>() * InputRescaleFactor;

        in[4] = 1.0 * InputRescaleFactor;; //bias input

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

    cppn_t& returncppn()
    {
        return _cppn;
    }


    template<typename NN>
    void write_cppndot(std::ostream& ofs, NN& nn)
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
        assert(Params::hnn::nb_layers == 3);

        ////=============================================================================================================
        // INPUT LAYER
        // Enter coordinates of the follwing input nodes

//        // Clune inspired sustrate
//        //leg 0;
//        _substrate[0].push_back(boost::make_tuple(-1.0,1.0));   //servo 0 - preivously desired joint angle
//        //_substrate[0].push_back(boost::make_tuple(-0.5,1.0));   //servo 1 - preivously desired joint angle
//        _substrate[0].push_back(boost::make_tuple( 0.0,1.0));   //servo 1 - joint angle
//        //_substrate[0].push_back(boost::make_tuple( 0.5,1.0));   //touch sensor
//        _substrate[0].push_back(boost::make_tuple( 1.0,1.0));   //misc. - sine signal value

//        //leg 1;
//        _substrate[0].push_back(boost::make_tuple(-1.0,0.6));   //servo 0 - previously desired joint angle
//        //_substrate[0].push_back(boost::make_tuple(-0.5,0.6));   //servo 1 - joint angle
//        _substrate[0].push_back(boost::make_tuple( 0.0,0.6));   //servo 1 - previously desired joint angle
//        //_substrate[0].push_back(boost::make_tuple( 0.5,0.6));   //touch sensor
//        _substrate[0].push_back(boost::make_tuple( 1.0,0.6));   //misc. - cos signal value

//        //leg 2;
//        _substrate[0].push_back(boost::make_tuple(-1.0,0.2));   //servo 0 - previously desired joint angle
//        //_substrate[0].push_back(boost::make_tuple(-0.5,0.2));   //servo 1 - joint angle
//        _substrate[0].push_back(boost::make_tuple( 0.0,0.2));   //servo 1 - previously desired joint angle
//        //_substrate[0].push_back(boost::make_tuple( 0.5,0.2));   //touch sensor
//        //_substrate[0].push_back(boost::make_tuple( 1.0,0.2));   //misc. - torso yaw

//        //leg 3;
//        _substrate[0].push_back(boost::make_tuple(-1.0,-0.2));   //servo 0 - previously desired joint angle
//        //_substrate[0].push_back(boost::make_tuple(-0.5,-0.2));   //servo 1 - joint angle
//        _substrate[0].push_back(boost::make_tuple( 0.0,-0.2));   //servo 1 - previously desired joint angle
//        //_substrate[0].push_back(boost::make_tuple( 0.5,-0.2));   //touch sensor
//        //_substrate[0].push_back(boost::make_tuple( 1.0,-0.2));   //misc. -  sine input

//        //leg 4;
//        _substrate[0].push_back(boost::make_tuple(-1.0,-0.6));   //servo 0 - previously desired joint angle
//        //_substrate[0].push_back(boost::make_tuple(-0.5,-0.6));   //servo 1 - joint angle
//        _substrate[0].push_back(boost::make_tuple( 0.0,-0.6));   //servo 1 - previously desired joint angle
//        //_substrate[0].push_back(boost::make_tuple( 0.5,-0.6));   //touch sensor
//        //_substrate[0].push_back(boost::make_tuple( , ));   //misc. - not used

//        //leg 5;
//        _substrate[0].push_back(boost::make_tuple(-1.0,-1.0));   //servo 0 - previously desired joint angle
//        //_substrate[0].push_back(boost::make_tuple(-0.5,-1.0)); //servo 1 - joint angle
//        _substrate[0].push_back(boost::make_tuple( 0.0,-1.0));   //servo 1 - previously desired joint angle
//        //_substrate[0].push_back(boost::make_tuple( 0.5,-1.0));   //touch sensor
//        //_substrate[0].push_back(boost::make_tuple( , ));   //misc. - not used
        ////=============================================================================================================

        // More hexapod like geometry.
        //leg 0;
        _substrate[0].push_back(boost::make_tuple(0.7*2.0-1.0,0.75*2.0-1.0)); //servo 0
        _substrate[0].push_back(boost::make_tuple(0.8*2.0-1.0,0.75*2.0-1.0)); //servo 1
        //_substrate[0].push_back(boost::make_tuple(0.8,0.5));   //misc. - sine signal value
        _substrate[0].push_back(boost::make_tuple(0.0,0.25));   //misc. - sine signal value

        //leg 1;
        _substrate[0].push_back(boost::make_tuple(0.7*2.0-1.0,0.5*2.0-1.0)); //servo 0
        _substrate[0].push_back(boost::make_tuple(0.8*2.0-1.0,0.5*2.0-1.0)); //servo 1
        //_substrate[0].push_back(boost::make_tuple(0.8,0.0));   //misc. - cosine signal value
        _substrate[0].push_back(boost::make_tuple(0.0,-0.25));   //misc. - cosine signal value

        //leg 2;
        _substrate[0].push_back(boost::make_tuple(0.7*2.0-1.0,0.25*2.0-1.0)); //servo 0
        _substrate[0].push_back(boost::make_tuple(0.8*2.0-1.0,0.25*2.0-1.0)); //servo 1

        //leg 3;
        _substrate[0].push_back(boost::make_tuple(0.3*2.0-1.0,0.25*2.0-1.0)); //servo 0
        _substrate[0].push_back(boost::make_tuple(0.2*2.0-1.0,0.25*2.0-1.0)); //servo 1

        //leg 4;
        _substrate[0].push_back(boost::make_tuple(0.3*2.0-1.0,0.5*2.0-1.0)); //servo 0
        _substrate[0].push_back(boost::make_tuple(0.2*2.0-1.0,0.5*2.0-1.0)); //servo 1

        //leg 5;
        _substrate[0].push_back(boost::make_tuple(0.3*2.0-1.0,0.75*2.0-1.0)); //servo 0
        _substrate[0].push_back(boost::make_tuple(0.2*2.0-1.0,0.75*2.0-1.0)); //servo 1

        ////=============================================================================================================
        // HIDDEN LAYER

//         // Clune inspired sustrate
//        //hidden nodes in row 0-5 and col 0-2 --> 18 nodes
//        for(float x = -1.0; x <= 1.0; x+=1.0)
//            for(float y = -1.0; y <= 1.0; y+=0.4)
//                _substrate[1].push_back(boost::make_tuple(x,y));

       // More hexapod like geometry.
       //hidden nodes in row 0-2 and col 0-4 --> 15 nodes
        _substrate[1].push_back(boost::make_tuple(0.7*2.0-1.0,0.75*2.0-1.0)); //servo 0
        _substrate[1].push_back(boost::make_tuple(0.8*2.0-1.0,0.75*2.0-1.0)); //servo 1
        //_substrate[1].push_back(boost::make_tuple(0.8,0.5));   //misc. - sine signal value
        _substrate[1].push_back(boost::make_tuple(0.0,0.25));   //misc. - sine signal value

        _substrate[1].push_back(boost::make_tuple(0.7*2.0-1.0,0.5*2.0-1.0)); //servo 0
        _substrate[1].push_back(boost::make_tuple(0.8*2.0-1.0,0.5*2.0-1.0)); //servo 1
        //_substrate[1].push_back(boost::make_tuple(0.8,0.0));   //misc. - cosine signal value
        _substrate[1].push_back(boost::make_tuple(0.0,-0.25));   //misc. - sine signal value

        _substrate[1].push_back(boost::make_tuple(0.7*2.0-1.0,0.25*2.0-1.0)); //servo 0
        _substrate[1].push_back(boost::make_tuple(0.8*2.0-1.0,0.25*2.0-1.0)); //servo 1
        //_substrate[1].push_back(boost::make_tuple(0.8,-0.5));   //misc. - no correspondence to node in input layer at same coordinate
        //_substrate[1].push_back(boost::make_tuple(0.8,-0.5));   //misc. - no correspondence to node in input layer at same coordinate

        _substrate[1].push_back(boost::make_tuple(0.3*2.0-1.0,0.25*2.0-1.0)); //servo 0
        _substrate[1].push_back(boost::make_tuple(0.2*2.0-1.0,0.25*2.0-1.0)); //servo 1

        _substrate[1].push_back(boost::make_tuple(0.3*2.0-1.0,0.5*2.0-1.0)); //servo 0
        _substrate[1].push_back(boost::make_tuple(0.2*2.0-1.0,0.5*2.0-1.0)); //servo 1

        _substrate[1].push_back(boost::make_tuple(0.3*2.0-1.0,0.75*2.0-1.0)); //servo 0
        _substrate[1].push_back(boost::make_tuple(0.2*2.0-1.0,0.75*2.0-1.0)); //servo 1
        ////=============================================================================================================
        // OUTPUT LAYER
        // Output nodes: For each of six legs, enter coordinates for output joint angle of servo 0 and of servo 1
        // For starters we simplify by considering that servo 2 joint angle signal is in antiphase with signal from servo 1

//        // Clune inspired sustrate
//        //leg 0;
//        _substrate[2].push_back(boost::make_tuple(-1.0,1.0));   //servo 0 - joint angle
//        _substrate[2].push_back(boost::make_tuple( 0.0,1.0));   //servo 1 - joint angle;

//        //leg 1;
//        _substrate[2].push_back(boost::make_tuple(-1.0,0.6));   //servo 0 - joint angle
//        _substrate[2].push_back(boost::make_tuple( 0.0,0.6));   //servo 1 - joint angle;

//        //leg 2;
//        _substrate[2].push_back(boost::make_tuple(-1.0,0.2));   //servo 0 - joint angle
//        _substrate[2].push_back(boost::make_tuple( 0.0,0.2));   //servo 1 - joint angle;

//        //leg 3;
//        _substrate[2].push_back(boost::make_tuple(-1.0,-0.2));   //servo 0 - joint angle
//        _substrate[2].push_back(boost::make_tuple( 0.0,-0.2));   //servo 1 - joint angle;

//        //leg 4;
//        _substrate[2].push_back(boost::make_tuple(-1.0,-0.6));   //servo 0 - joint angle
//        _substrate[2].push_back(boost::make_tuple( 0.0,-0.6));   //servo 1 - joint angle;

//        //leg 5;
//        _substrate[2].push_back(boost::make_tuple(-1.0,-1.0));   //servo 0 - joint angle
//        _substrate[2].push_back(boost::make_tuple( 0.0,-1.0));   //servo 1 - joint angle;


        // More hexapod like geometry.
        //leg 0;
        _substrate[2].push_back(boost::make_tuple(0.7*2.0-1.0,0.75*2.0-1.0)); //servo 0
        _substrate[2].push_back(boost::make_tuple(0.8*2.0-1.0,0.75*2.0-1.0)); //servo 1

        //leg 1;
        _substrate[2].push_back(boost::make_tuple(0.7*2.0-1.0,0.5*2.0-1.0)); //servo 0
        _substrate[2].push_back(boost::make_tuple(0.8*2.0-1.0,0.5*2.0-1.0)); //servo 1

        //leg 2;
        _substrate[2].push_back(boost::make_tuple(0.7*2.0-1.0,0.25*2.0-1.0)); //servo 0
        _substrate[2].push_back(boost::make_tuple(0.8*2.0-1.0,0.25*2.0-1.0)); //servo 1

        //leg 3;
        _substrate[2].push_back(boost::make_tuple(0.3*2.0-1.0,0.25*2.0-1.0)); //servo 0
        _substrate[2].push_back(boost::make_tuple(0.2*2.0-1.0,0.25*2.0-1.0)); //servo 1

        //leg 4;
        _substrate[2].push_back(boost::make_tuple(0.3*2.0-1.0,0.5*2.0-1.0)); //servo 0
        _substrate[2].push_back(boost::make_tuple(0.2*2.0-1.0,0.5*2.0-1.0)); //servo 1

        //leg 5;
        _substrate[2].push_back(boost::make_tuple(0.3*2.0-1.0,0.75*2.0-1.0)); //servo 0
        _substrate[2].push_back(boost::make_tuple(0.2*2.0-1.0,0.75*2.0-1.0)); //servo 1
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
