#ifndef IDW_HPP
#define IDW_HPP

#include "svm.h"//"/home/koos/Desktop/libsvm-3.11/svm.h"

//implementation of SVM
class SVM
{
public:
    SVM()
    {
        model = 0;

        param.svm_type = NU_SVR;
        param.kernel_type = LINEAR;
        param.degree = 3;
        param.gamma = 1 / 12.0f;
        param.coef0 = 0;
        param.nu = 0.5;
        param.cache_size = 100;
        param.C = 1;
        param.eps = 1e-3;
        param.p = 0.1;
        param.shrinking = 1;
        param.probability = 0;
        param.nr_weight = 0;
        param.weight_label = NULL;
        param.weight = NULL;
    }

    //re-building problem from scratch
    //+ learning
    template<typename Center>
    void learn(std::vector<Center> centers)
    {

        if (model)
            svm_free_and_destroy_model(&model);

        struct svm_problem prob;

        prob.l = centers.size();
        prob.y = (double *) malloc(prob.l * sizeof(double));
        prob.x = (struct svm_node **) malloc(prob.l * sizeof(struct svm_node*));

        for (int i=0;i < prob.l;++i)
        {
            prob.y[i] = centers[i].fit().disparity[0];

            prob.x[i] = (struct svm_node*) malloc((centers[i].fit().features.size() + 1) * sizeof(struct svm_node));

            for (int j=0;j < centers[i].fit().features.size();++j)
            {
                prob.x[i][j].index = j+1;
                prob.x[i][j].value = centers[i].fit().features[j];
            }

            prob.x[i][centers[i].fit().features.size()].index = -1;
            prob.x[i][centers[i].fit().features.size()].value = 0;
        }

        model = svm_train(&prob, &param);
        //svm_save_model("test.dat", model);
    }

    //params: none
    //prediction with SVM
    template<typename Center>
    void predict(std::vector<float> params, std::vector<Center> centers, std::vector<float> inputs, float& outputs) const
    {
        struct svm_node *x_pred = (struct svm_node*) malloc(inputs.size() * sizeof(struct svm_node));

        for (int i=0;i < inputs.size();++i)
        {
            x_pred[i].index = i+1;
            x_pred[i].value = inputs[i];
        }

        outputs = svm_predict(model, x_pred);
    }

private:
    struct svm_parameter param;
    struct svm_model *model;
};

#endif
