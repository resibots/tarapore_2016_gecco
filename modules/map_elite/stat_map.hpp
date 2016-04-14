#ifndef STAT_MAP_HPP_
#define STAT_MAP_HPP_

#include <numeric>
#include <boost/multi_array.hpp>
#include <sferes/stat/stat.hpp>

namespace sferes
{
namespace stat
{
SFERES_STAT(Map, Stat)
{
    public:
    typedef boost::shared_ptr<Phen> phen_t;
    typedef boost::multi_array<phen_t, Params::ea::behav_dim> array_t;
    typedef boost::array<float, Params::ea::behav_dim> point_t;
    typedef boost::array<typename array_t::index, Params::ea::behav_dim> behav_index_t;

    size_t behav_dim;
    behav_index_t behav_shape;
    behav_index_t behav_strides;
    behav_index_t behav_indexbase;

    //Map() : _xs(0), _ys(0) {}
    Map() : behav_dim(Params::ea::behav_dim)
    {
        for(size_t i = 0; i < Params::ea::behav_shape_size(); ++i)
            behav_shape[i] = Params::ea::behav_shape(i);
    }

    template<typename E>
    void refresh(const E& ea)
    {
        _archive.clear();

        for(size_t i = 0; i < behav_dim; ++i) //Params::ea::behav_shape_size()
        {
            assert(ea.archive().shape()[i] == behav_shape[i]);  //ea.behav_shape[i]);
            behav_strides[i] = ea.archive().strides()[i];
            behav_indexbase[i] = ea.archive().index_bases()[i];
        }

        /*_xs = ea.archive().shape()[0];
        _ys = ea.archive().shape()[1];
        assert(_xs == Params::ea::res_x);
        assert(_ys == Params::ea::res_y);*/

        //for(auto phen_t* i = ea.archive().data(); i < (ea.archive().data() + ea.archive().num_elements()); ++i)
        for(const phen_t* i = ea.archive().data(); i < (ea.archive().data() + ea.archive().num_elements()); ++i)
        {
            phen_t p = *i;
            _archive.push_back(p);
        }


        /*for (size_t i = 0; i < _xs; ++i)
            for (size_t j = 0; j < _ys; ++j)
            {
                phen_t p = ea.archive()[i][j];
                _archive.push_back(p);
            }*/

        if (ea.gen() % Params::pop::dump_period_logs == 0)
        {
            //_write_archive_antoinestyle(ea.archive(), std::string("archive_"), ea);
            _write_archive(ea.archive(), std::string("archive_"), ea);
            _write_progress(ea.archive(),std::string("progress_archive"), ea);
            _write_progress_performance(ea.archive(),std::string("progress_archive_actualperformance"), ea);
#ifdef MAP_WRITE_PARENTS
            _write_parents(ea.archive(), ea.parents(), std::string("parents_"), ea);
#endif
            _write_fitness_diversity_stats(std::string("diff_fitness_diversity_"), ea); 
        }
    }


    void show(std::ostream& os, size_t k)
    {
        //std::cerr << "loading "<< k / _ys << "," << k % _ys << std::endl;

        std::cerr << "loading " ;
        for(size_t i = 0; i < behav_dim; ++i)
            std::cerr << (k / behav_strides[i] % behav_shape[i] +  behav_indexbase[i]) << ",";
        std::cerr << std::endl;


        assert(_archive[k]);

        if (_archive[k])
        {
            _archive[k]->develop();
            _archive[k]->show(os);
            _archive[k]->fit().set_mode(fit::mode::view);
            _archive[k]->fit().eval(*_archive[k]);
        }
        else
            std::cerr << "Warning, no point here" << std::endl;


	/*std::cerr << "loading non-discretized behavior descriptors" ;
	for(size_t bd_index=0;bd_index<Params::ea::behav_dim;++bd_index)
	    std::cerr << _archive[k]->fit().desc()[bd_index] << " ";
	    std::cerr << std::endl;*/
    }

    template<class Archive>
    void serialize(Archive& ar, const unsigned int version)
    {
        ar & BOOST_SERIALIZATION_NVP(_archive);
        ar & BOOST_SERIALIZATION_NVP(behav_dim);
        ar & BOOST_SERIALIZATION_NVP(behav_shape);
        ar & BOOST_SERIALIZATION_NVP(behav_strides);
        ar & BOOST_SERIALIZATION_NVP(behav_indexbase);

        //ar & boost::serialization::make_nvp("behav_indexbase", behav_indexbase);

        /*ar & BOOST_SERIALIZATION_NVP(_xs);
        ar & BOOST_SERIALIZATION_NVP(_ys);*/
    }

    std::vector<phen_t> & archive()
    {
        return _archive;
    }


    protected:
    std::vector<phen_t> _archive;
    //int _xs, _ys;

    template<typename EA>
    void _write_parents(const array_t& array,
                        const array_t& p_array,
                        const std::string& prefix,
                        const EA& ea) const
    {
        std::cout << "writing..." << prefix << ea.gen() << std::endl;
        std::string fname =  ea.res_dir() + "/"
                + prefix
                + boost::lexical_cast<
                std::string>(ea.gen())
                + std::string(".dat");
        std::ofstream ofs(fname.c_str());

        //for(auto i = array.data(); i < (array.data() + array.num_elements()); ++i)
        for(const phen_t* i = array.data(); i < (array.data() + array.num_elements()); ++i)
        {
            //boost::array<typename array_t::index, behav_dim> idx = posinarray;
            if (*i)
            {
                behav_index_t idx = ea.getindexarray(array, i);
                assert(array(idx)->fit().value() == (*i)->fit().value());
                if(p_array(idx))
                {
                    for(size_t dim = 0; dim < behav_dim; ++dim)
                        ofs << idx[dim] / (float) behav_shape[dim] << " ";
                    ofs << " " << p_array(idx)->fit().value() << " " ;

                    point_t p = ea.get_point(p_array(idx));
                    behav_index_t posinparent;
                    for(size_t dim = 0; dim < behav_dim; ++dim)
                    {
                        posinparent[dim] = round(p[dim] * behav_shape[dim]); //SHOULD IT BE THE BELOW
                        ofs << posinparent[dim] / (float) behav_shape[dim] << " "; //SHOULD IT BE THE BELOW
                        //posinparent[dim] = std::min(round(p[dim] * (float)(behav_shape[dim]-1)), (float)(behav_shape[dim]-1));
                        //ofs << posinparent[dim] / (float) (behav_shape[dim]-1) << " ";
                    }
                    ofs << " " << array(idx)->fit().value() << std::endl;
                }
            }
        }

        /*for (size_t i = 0; i < _xs; ++i)
            for (size_t j = 0; j < _ys; ++j)
                if (array[i][j] && p_array[i][j])
                {
                    point_t p = _get_point(p_array[i][j]);
                    size_t x = round(p[0] * _xs);
                    size_t y = round(p[1] * _ys);
                    ofs << i / (float) _xs
                        << " " << j / (float) _ys
                        << " " << p_array[i][j]->fit().value()
                        << " " << x / (float) _xs
                        << " " << y / (float) _ys
                        << " " << array[i][j]->fit().value()
                        << std::endl;
                }*/
    }

    template<typename EA>
    void _write_fitness_diversity_stats(const std::string& prefix, const EA& ea) const
    {
        std::cout << "writing..." << prefix << ea.gen() << std::endl;
        std::string fname = ea.res_dir() + "/"
                + prefix
                + boost::lexical_cast<
                std::string>(ea.gen())
                + std::string(".dat");

        std::ofstream ofs(fname.c_str());

        for (size_t i = 0; i< ea.generated_fitness().size(); ++i)
            ofs << ea.generated_fitness()[i] << "  " << ea.generated_diversity()[i] << std::endl;

        std::string fname1 = ea.res_dir() + "/"
                + std::string("lethalmutantscount.dat");
        std::ofstream ofs1(fname1.c_str(), std::ofstream::out | std::ofstream::app);
        ofs1 << ea.gen() << " " << ea.lethalmutantcount() << std::endl;
    }


    template<typename EA>
    void _write_archive(const array_t& array,
                        const std::string& prefix,
                        const EA& ea) const
    {
        std::cout << "writing..." << prefix << ea.gen() << std::endl;
        std::string fname = ea.res_dir() + "/"
                + prefix
                + boost::lexical_cast<
                std::string>(ea.gen())
                + std::string(".dat");

        std::ofstream ofs(fname.c_str());

        size_t offset = 0;
        for(const phen_t* i = array.data(); i < (array.data() + array.num_elements()); ++i)
        {

            //boost::array<typename array_t::index, behav_dim> idx = posinarray;

            if (*i)
            {
                behav_index_t posinarray = ea.getindexarray(array, i);
                assert(array(posinarray)->fit().value() == (*i)->fit().value());

                //boost::array<typename array_t::index, behav_dim> idx = posinarray; //: error: ‘this’ is not a constant expression
                //../modules/map_elite/stat_map.hpp:186:66: note: in template argument for type ‘long unsigned int’ ???


                /*for(size_t dim = 0; dim < behav_dim; ++dim)
                    ofs << posinarray[dim] / (float) behav_shape[dim] << " ";*/


                phen_t indiv = *i;
                for(size_t bd_index=0;bd_index<Params::ea::behav_dim;++bd_index)
                    ofs << indiv->fit().desc()[bd_index] << " ";

                ofs << "   " << array(posinarray)->fit().value() << std::endl;

                ofs << "   " << offset;

                ofs << std::endl;


            }
            ++offset;
        }

        /*for (size_t i = 0; i < _xs; ++i)
            for (size_t j = 0; j < _ys; ++j)
                if (array[i][j])
                {
                    ofs << i / (float) _xs
                        << " " << j / (float) _ys
                        << " " << array[i][j]->fit().value()
                        << std::endl;
                }*/
    }


    template<typename EA>
    void _write_archive_antoinestyle(const array_t& array,
                                     const std::string& prefix,
                                     const EA& ea) const
    {
        std::cout << "writing ..." << prefix << ea.gen() << std::endl;
        std::string fname = ea.res_dir() + "/"
                + prefix
                + boost::lexical_cast<
                std::string>(ea.gen())
                + std::string(".dat");

        std::ofstream ofs(fname.c_str());

        for(const phen_t* i = array.data(); i < (array.data() + array.num_elements()); ++i)
        {
            if (*i)
            {
                phen_t indiv = *i;
		
		for(size_t bd_index=0;bd_index<Params::ea::behav_dim;++bd_index)
                	ofs << indiv->fit().desc()[bd_index] << " ";

                ofs << "    " << indiv->fit().value() << "    ";

                //ofs << indiv->fit().desc()[0] << " " << indiv->fit().desc()[1] << " " << indiv->fit().desc()[2] << " " << indiv->fit().desc()[3] << " " << indiv->fit().desc()[4] << " " << indiv->fit().desc()[5] << "    " << indiv->fit().value() << "    ";

                for(int j=0;j<indiv->gen().size();j++)
                    ofs << indiv->gen().data(j) << " ";
                ofs<<std::endl;
            }
        }
    }


    template<typename EA>
    void _write_progress(const array_t& array,
                         const std::string& prefix,
                         const EA& ea) const
    {
        std::cout << "writing..." << prefix << std::endl;
        std::string fname = ea.res_dir() + "/"
                + prefix
                + std::string(".dat");

        std::ofstream ofs(fname.c_str(), std::ofstream::out | std::ofstream::app);

        size_t archive_size = 0;
        float archive_mean = 0.0f;
        float archive_max = 0.0f;
        float mean_dist_center = 0.0f;
	float servo_frequencies_max = 0.0f;

        for(const phen_t* i = array.data(); i < (array.data() + array.num_elements()); ++i)
        {
            if(*i)
            {
                phen_t p = *i;
                archive_size++;
                archive_mean += p->fit().value();

                if(archive_max < p->fit().value())
		{   
                    archive_max = p->fit().value();
		    servo_frequencies_max = p->fit().servo_frequencies_max;
		}    

                float dist = 0.0f;
                for(size_t i = 0; i < Params::ea::behav_shape_size(); ++i)
                {
                    assert(p->fit().desc()[i] >= 0.0f && p->fit().desc()[i] <= 1.0f);

                    float diff = p->fit().desc()[i] -
                            (float)round(p->fit().desc()[i] * (float)(Params::ea::behav_shape(i)-1)) / (float)(Params::ea::behav_shape(i) - 1);

                    dist += diff * diff;
                }
                mean_dist_center+=sqrtf(dist);
            }
        }

        archive_mean /= archive_size;
        mean_dist_center /= archive_size;

        ofs << ea.gen() << " " << archive_size << " " << archive_mean << " " << archive_max << " " << mean_dist_center << " " << servo_frequencies_max << std::endl;
    }



    template<typename EA>
    void _write_progress_performance(const array_t& array,
                         const std::string& prefix,
                         const EA& ea) const
    {
        std::cout << "writing..." << prefix << std::endl;
        std::string fname = ea.res_dir() + "/"
                + prefix
                + std::string(".dat");

        std::ofstream ofs(fname.c_str(), std::ofstream::out | std::ofstream::app);

        size_t archive_size = 0;
        float archive_mean = 0.0f;
        float archive_max = 0.0f;

        for(const phen_t* i = array.data(); i < (array.data() + array.num_elements()); ++i)
        {
            if(*i)
            {
                phen_t p = *i;
                archive_size++;
                archive_mean += p->fit().dist;

                if(archive_max < p->fit().dist)
		{   
                    archive_max = p->fit().dist;
		}    
            }
        }

        archive_mean /= archive_size;

        ofs << ea.gen() << " " << archive_size << " " << archive_mean << " " << archive_max << std::endl;
    }

};
}
}

#endif
