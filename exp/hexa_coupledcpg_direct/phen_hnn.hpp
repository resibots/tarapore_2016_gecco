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

#ifndef PHEN_HNN_HPP
#define PHEN_HNN_HPP

#include <map>
#include <queue>
#include <sferes/phen/indiv.hpp>
#include <modules/nn2/nn.hpp>

#include <modules/nn2/params.hpp>
#include "gen_hnn.hpp"

#define DEFAULTVALUE -10000.0 // default value for phase bias couplings

namespace sferes
{
namespace phen
{
// hyperneat-inspired phenotype, based on a cppn
SFERES_INDIV(Hnn, Indiv)
{
    public:
    typedef Gen gen_t;
    typedef typename Params::hnn::neuron_t neuron_t;
    typedef typename Params::hnn::connection_t connection_t;
    typedef typename nn::NN<neuron_t, connection_t> nn_t;
    typedef typename nn_t::vertex_desc_t v_d_t;
    typedef typename std::vector<v_d_t> layer_t;
    Hnn() : _developed(false) {}

    void develop()
    {
        if (_developed)
            return;
        _developed = true;
        this->gen().init();


        static const size_t nb_inputs  = Params::hnn::nb_inputs;
        static const size_t last_layer = Params::hnn::nb_layers - 1;
        static const size_t nb_outputs = Params::hnn::nb_outputs;

        //The hnn is not used in this experiment. Simply query the parameters of each oscillator from the CPNN
        //Store the parameters in the below vectors
        assert(this->gen().substrate().size() == 1);

        amp_osc.clear();
        for (size_t n = 0; n < this->gen().substrate()[0].size(); ++n)
        {
            std::vector<float> r = this->gen().query(this->gen().substrate()[0][n]);
            assert(r.size()==2);

            amp_osc.push_back(r[0]); // intrinsic amplitude of oscillator
        }         
        assert(amp_osc.size()==6*2); // two oscillators on each of six legs

        //inter-oscillator couplings
        for (size_t n = 0; n < this->gen().substrate()[0].size(); ++n)
        {
            std::vector<float> params(this->gen().substrate()[0].size(), DEFAULTVALUE);
            phasebias_coupling.push_back(params);
            //wts_coupling.push_back(params);

            std::vector<int> params1(this->gen().substrate()[0].size(), -1);
            adjacency.push_back(params1);
        }

         assert(phasebias_coupling.size()==12);
         assert(phasebias_coupling[0].size()==12);
         assert(adjacency.size()==12);
         assert(adjacency[0].size()==12);

        for (int n1 = 0; n1 < (int)this->gen().substrate()[0].size(); ++n1)
        {
            if(!(n1 == 3 || n1 == 7 || n1 == 11)) //exceptions at left and right border of subtrate to prevent toroidal couplings
                querycouplings(n1, n1 + 1); // horizontal couplings

            if(!(n1 == 0 || n1 == 4 || n1 == 8)) //exceptions at left and right border of subtrate to prevent toroidal couplings
                querycouplings(n1, n1 - 1); // horizontal couplings

            querycouplings(n1, n1 + 4); // vertical couplings
            querycouplings(n1, n1 - 4); // vertical couplings
        }
    }

    void querycouplings(int n1, int n2)
    {
        if(n2 < 0 || n2 >= (int)this->gen().substrate()[0].size())
            return;

        for(size_t i=0; i<this->gen().substrate()[0].size();++i)
            if(adjacency[n1][i] == -1)
            {
                adjacency[n1][i] = n2;
                break;
            }

        if(phasebias_coupling[n1][n2] == DEFAULTVALUE)
        {
            std::vector<float> r = this->gen().query(this->gen().substrate()[0][n1], this->gen().substrate()[0][n2]);
            assert(r.size()==2);
            //wts_coupling[n1][n2] = r[2]; wts_coupling[n2][n1] = r[2];
            phasebias_coupling[n1][n2] = r[1]; phasebias_coupling[n2][n1] = -r[1];
        }
    }

    nn_t& nn() { return _nn; }
    const nn_t& nn() const { return _nn; }

    std::vector<float> amp_osc; // individual oscillator parameters are queried from the cppn and pushed here
    //std::vector<float> initphase_osc; // individual oscillator parameters are queried from the cppn and pushed here

    //std::vector<std::vector<float> > wts_coupling; // inter-oscillator weight parameters are queried from the cppn and pushed here
    std::vector<std::vector<float> > phasebias_coupling; // inter-oscillator phase bias parameters are queried from the cppn and pushed here
    std::vector<std::vector<int> > adjacency; // index of connected oscillators

    protected:
    bool _developed;

    nn_t _nn;
    std::vector<layer_t> _layers;
};
}
}
#endif
