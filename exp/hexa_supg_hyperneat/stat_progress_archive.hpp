#ifndef STAT_BEST_HPP
#define STAT_BEST_HPP

#include <sferes/stat/stat.hpp>

namespace sferes
{
namespace stat
{

SFERES_STAT(ProgressArchive, Stat)
{
    public:
    typedef boost::shared_ptr<Phen> phen_t;

    template<typename E>
    void refresh(const E& ea)
    {
        if(ea.gen() % Params::pop::dump_period != 0)
            return;

        this->_create_log_file(ea, "progress_archive.dat");

        size_t archive_size = 0;
        float archive_mean = 0.0f;
        float archive_max = 0.0f;
        float mean_dist_center = 0.0f;


        for(const phen_t* i = ea.archive().data(); i < (ea.archive().data() + ea.archive().num_elements()); ++i)
        {
            phen_t p = *i;
            archive_size++;
            archive_mean += p->fit().value();

            if(archive_max < p->fit().value())
                archive_max = p->fit().value();

            float dist = 0.0f;
            for(size_t i = 0; i < Params::ea::behav_shape_size(); ++i)
            {
                //assert(p->fit().desc()[i] >= 0.0f && p->fit().desc()[i] <= 1.0f);

                float diff = p->fit().desc()[i] -
                        (float)round(p->fit().desc()[i] * (float)(Params::ea::behav_shape(i)-1)) / (float)(Params::ea::behav_shape(i) - 1);

                dist += diff * diff;
            }
            mean_dist_center+=sqrtf(dist);
        }

        archive_mean /= archive_size;
        mean_dist_center /= archive_size;

        (*this->_log_file) << ea.gen() << " " << archive_size << " " << archive_mean << " " << archive_max << " " << mean_dist_center << std::endl;
    }
};
}
}
#endif
