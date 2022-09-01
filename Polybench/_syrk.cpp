/* DaCe AUTO-GENERATED FILE. DO NOT MODIFY */
#include <dace/dace.h>
#include "../../include/hash.h"

struct _syrk_t {
    double * __restrict__ __0_dace_C_0;
    double * __restrict__ __0_dace_A_0;
    dace::perf::Report report;
};


#include <chrono>

#include <chrono>

#include <chrono>

#include <chrono>
#include <chrono>
#include <chrono>
#include <chrono>
#include <chrono>
inline void nested__state54_55_2_1_6(_syrk_t *__state, double* dace_A_4, const double& dace_alpha_2, double* dace_C_4, int i, long long j_0) {
    long long k;




    for (k = 0; (k <= 999); k = (k + 1)) {
        {

            {
                double A_1_1 = dace_A_4[((1000 * i) + k)];
                double A_2_1 = dace_A_4[((1000 * j_0) + k)];
                double C_1_1 = dace_C_4[0];
                double alpha = dace_alpha_2;
                double C_out_1_1;

                ///////////////////
                C_out_1_1=C_1_1+alpha*A_1_1*A_2_1;
                ///////////////////

                dace_C_4[0] = C_out_1_1;
            }

        }

    }

    
}

inline void loop_body_1_1_0_0(_syrk_t *__state, double* dace_A_4, const double& dace_alpha_2, const double& dace_beta_2, double* dace_C_4, int i) {

    {

        {
            for (auto j = 0; j < (i + 1); j += 1) {
                {
                    double C_1_1 = dace_C_4[j];
                    double beta = dace_beta_2;
                    double C_out_1_1;

                    ///////////////////
                    C_out_1_1=C_1_1*beta;
                    ///////////////////

                    dace_C_4[j] = C_out_1_1;
                }
            }
        }

    }
    {

        {
            for (auto j_0 = 0; j_0 < (i + 1); j_0 += 1) {
                nested__state54_55_2_1_6(__state, &dace_A_4[0], dace_alpha_2, &dace_C_4[j_0], i, j_0);
            }
        }

    }

    
}

inline void kernel_syrk_0_1_0(_syrk_t *__state, double* dace_A_4, const double& dace_alpha_2, const double& dace_beta_2, double* dace_C_4, int& dace_print_6) {

    {

        {
            for (auto i = 0; i < 1200; i += 1) {
                loop_body_1_1_0_0(__state, &dace_A_4[0], dace_alpha_2, dace_beta_2, &dace_C_4[(1200 * i)], i);
            }
        }

    }
    
}

void __program__syrk_internal(_syrk_t *__state, char * __restrict__ argv_loc, int _argcount, int argc_loc)
{
    auto __dace_tbegin_0 = std::chrono::high_resolution_clock::now();
    int print;
    double dace_alpha_0;
    double dace_beta_0;
    bool dace_tmp_if_1_0;
    long long tmp_for_1;
    long long tmp_for_2;

    {

        {
            double alpha_out;

            ///////////////////
            alpha_out=1.5;
            ///////////////////

            dace_alpha_0 = alpha_out;
        }
        {
            double beta_out;

            ///////////////////
            beta_out=1.2;
            ///////////////////

            dace_beta_0 = beta_out;
        }
        {
            for (auto i = 0; i < 1200; i += 1) {
                for (auto j_0 = 0; j_0 < 1000; j_0 += 1) {
                    {
                        double A_out_1_1;

                        ///////////////////
                        auto dace_n_0 = 1200;
                        auto j = j_0;
                        A_out_1_1=(double)(((i*j+1)%dace_n_0))/dace_n_0;
                        ///////////////////

                        __state->__0_dace_A_0[((1000 * i) + j_0)] = A_out_1_1;
                    }
                }
            }
        }
        {
            for (auto i = 0; i < 1200; i += 1) {
                for (auto j = 0; j < 1200; j += 1) {
                    {
                        double C_out_1_1;

                        ///////////////////
                        auto dace_m_0 = 1000;
                        C_out_1_1=(double)(((i*j+2)%dace_m_0))/dace_m_0;
                        ///////////////////

                        __state->__0_dace_C_0[((1200 * i) + j)] = C_out_1_1;
                    }
                }
            }
        }

    }
    dace_tmp_if_1_0 = (argc_loc > 42);
    {

        kernel_syrk_0_1_0(__state, &__state->__0_dace_A_0[0], dace_alpha_0, dace_beta_0, &__state->__0_dace_C_0[0], print);

    }
    if ((dace_tmp_if_1_0 != 0)) {
        {

            {
                int dace_print_11_task = print;
                int dace_print_11_task_out;

                ///////////////////
                fprintf((stderr),"==BEGIN DUMP_ARRAYS==\n");
                ///////////////////

                print = dace_print_11_task_out;
            }
            {
                int dace_print_11_task = print;
                int dace_print_11_task_out;

                ///////////////////
                fprintf((stderr),"begin dump: %s","C");
                ///////////////////

                print = dace_print_11_task_out;
            }

        }


        for (tmp_for_1 = 0; (tmp_for_1 < 1200); tmp_for_1 = (tmp_for_1 + 1)) {



            for (tmp_for_2 = 0; (tmp_for_2 < 1200); tmp_for_2 = (tmp_for_2 + 1)) {
                {

                    {
                        int dace_print_13_task = print;
                        int dace_print_13_task_out;

                        ///////////////////
                        fprintf((stderr),"\n");
                        ///////////////////

                        print = dace_print_13_task_out;
                    }
                    {
                        double C_1_1 = __state->__0_dace_C_0[((1200 * tmp_for_1) + tmp_for_2)];
                        int dace_print_13_task = print;
                        int dace_print_13_task_out;

                        ///////////////////
                        fprintf((stderr),"%0.2lf ",C_1_1);
                        ///////////////////

                        print = dace_print_13_task_out;
                    }

                }

            }


        }

    }

    auto __dace_tend_0 = std::chrono::high_resolution_clock::now();
    unsigned long int __dace_ts_start_0 = std::chrono::duration_cast<std::chrono::microseconds>(__dace_tbegin_0.time_since_epoch()).count();
    unsigned long int __dace_ts_end_0 = std::chrono::duration_cast<std::chrono::microseconds>(__dace_tend_0.time_since_epoch()).count();
    __state->report.add_completion("SDFG _syrk", "Timer", __dace_ts_start_0, __dace_ts_end_0, 0, -1, -1);
}

DACE_EXPORTED void __program__syrk(_syrk_t *__state, char * __restrict__ argv_loc, int _argcount, int argc_loc)
{
    __program__syrk_internal(__state, argv_loc, _argcount, argc_loc);
}

DACE_EXPORTED _syrk_t *__dace_init__syrk(int _argcount)
{
    int __result = 0;
    _syrk_t *__state = new _syrk_t;


    __state->__0_dace_C_0 = new double DACE_ALIGN(64)[1440000];
    __state->__0_dace_A_0 = new double DACE_ALIGN(64)[1200000];

    if (__result) {
        delete __state;
        return nullptr;
    }
    return __state;
}

DACE_EXPORTED void __dace_exit__syrk(_syrk_t *__state)
{
    __state->report.save(".dacecache/_syrk/perf", __HASH__syrk);
    delete[] __state->__0_dace_C_0;
    delete[] __state->__0_dace_A_0;
    delete __state;
}

