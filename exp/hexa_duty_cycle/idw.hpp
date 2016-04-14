#ifndef IDW_HPP
#define IDW_HPP

#include <math.h>

//implementation of Inverse Distance Weighting interpolator
class IDW
{
public:
    //no learning step with IDW interpolation
    template<typename Center>
    void learn(std::vector<Center> centers)
    {
    }



    //params: kappa (smoothness)
    template<typename Center,typename TypeInputs>
    void predict(std::vector<Center> centers, std::vector<TypeInputs> inputs, float& outputs) const
    {
        float numer = 0.0f;
        float denom = 0.0f;
        for (size_t i=0;i < centers.size();++i)
        {
            if (centers[i].fit().legs_features.size() != inputs.size())
            {
                printf("Problem with dimensions in IDW model while predicting!\n");
                printf("%d %d\n", centers[i].fit().legs_features.size(), inputs.size());
                return;
            }

            float dist = 0.0f;
            /*for(size_t j=0;j < inputs.size();++j)
              {
                float diff = (centers[i]->fit().features[j] - inputs[j]);
                dist += pow(diff, params[0]);
              }*/
            dist=centers[i].fit().dist_contact(inputs);

                if (dist < 1e-8)
            {
                outputs = centers[i].fit().disparity[0];
                return;
            }

            numer += centers[i].fit().disparity[0] / dist;
            denom += 1.0f / dist;
        }

        outputs = numer / denom;
    }
};

#endif
