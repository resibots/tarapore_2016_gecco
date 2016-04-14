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


#ifndef   	CROWD_H_
# define   	CROWD_H_

#include <sferes/parallel.hpp>

namespace sferes
{
  namespace ea
  {
    namespace crowd
    {
      static const float inf = 1.0e14;

      template<typename Indiv>
      class assign_crowd
      {
      public:
	std::vector<std::vector<Indiv> >& _fronts;

	~assign_crowd() { }
	assign_crowd(std::vector<std::vector<Indiv> >& fronts) :
	  _fronts(fronts) {}
	assign_crowd(const assign_crowd& ev) : _fronts(ev._fronts) {}
	void operator() (const parallel::range_t& r) const
	{
 	  for (size_t i = r.begin(); i != r.end(); ++i)
 	    _assign_crowd(_fronts[i]);
	}
      protected:
	typedef typename std::vector<Indiv>::iterator it_t;
	typedef typename std::vector<Indiv>::const_iterator cit_t;

	void _fmin_max(const std::vector<Indiv>& f,
		       std::vector<float>& fmin,
		       std::vector<float>& fmax) const
	{
	  assert(f.size());
	  size_t nb_objs = f[0]->fit().objs().size();
	  assert(nb_objs);
	  fmin.resize(nb_objs);
	  fmax.resize(nb_objs);
	  for (unsigned i = 0; i < nb_objs; ++i)
	    {
	      float mi = std::numeric_limits<float>::max();
	      float ma = -std::numeric_limits<float>::max();
	      for (cit_t it = f.begin(); it != f.end(); ++it)
		{
		  float o = (*it)->fit().obj(i);
		  assert(!std::isnan(o));
		  assert(!std::isnan(i));
		  assert(!std::isinf(o));
		  assert(!std::isinf(i));

		  if (o < mi)
		    mi = o;
		  if (o > ma)
		    ma = o;
		}
	      fmin[i] = mi;
	      fmax[i] = ma;
	      assert(fmin[i] <= fmax[i]);
	    }
	}

	/// Deb, p248
	/// /!\ end not included (like any stl algo)
	void _assign_crowd(std::vector<Indiv>& f) const
	{

#ifndef NDEBUG
	  BOOST_FOREACH(Indiv& ind, f)
	    for (size_t i = 0; i < ind->fit().objs().size(); ++i)
	      { assert(!std::isnan(ind->fit().obj(i))); }
#endif


	  if (f.size() == 1)
	    {
	      f[0]->set_crowd(crowd::inf);
	      return;
	    }
	  if (f.size() == 2)
	  {
	    f[0]->set_crowd(crowd::inf);
	    f[1]->set_crowd(crowd::inf);
	  }


	  for (it_t it = f.begin() ; it != f.end() ; ++it)
	    {
	      float d = 0.0f;
	      for (it_t it2 = f.begin() ; it2 != f.end() ; ++it2)
		{

		  float dist = 0;

		  assert((*it)->data().size() == (*it2)->data().size());
		  for (size_t i = 0; i < (*it)->data().size(); i++)
		    dist += (float) fabs((*it)->data()[i] - (*it2)->data()[i]);
		 d+=dist / (float)(*it)->data().size();

		}
	      d /= (float)f.size();
	      (*it)->set_crowd(d);
	      
	    }
	
	}


	
      };



      struct compare_crowd
      {
	template<typename I>
	bool operator()(const boost::shared_ptr<I> i1, const boost::shared_ptr<I> i2) const
	{
	  return i1->crowd() > i2->crowd();
	}
      };

      struct compare_ranks
      {
	template<typename I>
	bool operator()(const boost::shared_ptr<I> i1, const boost::shared_ptr<I> i2) const
	{
	  return i1->rank() < i2->rank();
	}
      };


      // a special indiv to add rank & crowd
      template<typename Phen>
      class Indiv : public Phen
      {
      public:
	int rank() const { return _rank; }
	float crowd() const { return _crowd; }
	void set_rank(int r) { _rank = r; }
	void set_crowd(float d) { _crowd = d; }
	Indiv(const Phen& p) : Phen(p) {}
	Indiv() {}
	// overriding ! (not a redefinition of a virtual)
	void cross(const boost::shared_ptr<Indiv>& i2,
                   boost::shared_ptr<Indiv>& o1,
                   boost::shared_ptr<Indiv>& o2)
        {
	  assert(i2);
	  if (!o1)
	    o1 = boost::shared_ptr<Indiv>(new Indiv());
	  if (!o2)
	    o2 = boost::shared_ptr<Indiv>(new Indiv());
	  this->_gen.cross(i2->gen(), o1->gen(), o2->gen());
#ifdef TRACK_FIT
#warning track fit is enabled
           o1->fit() = this->fit();
           o2->fit() = i2->fit();
#endif
        }
      protected:
	// rank
	int _rank;
	// crowding distance
	float _crowd;
      };

    }
  }
}

#endif 	    /* !CROWD_H_ */
