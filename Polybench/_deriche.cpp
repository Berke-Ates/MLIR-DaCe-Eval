/* DaCe AUTO-GENERATED FILE. DO NOT MODIFY */
#include <dace/dace.h>
#include "../../include/hash.h"

struct _deriche_t {
    float * __restrict__ __0_dace_imgIn_0;
    float * __restrict__ __0_dace_imgOut_0;
    float * __restrict__ __0_dace_y1_0;
    float * __restrict__ __0_dace_y2_0;
    dace::perf::Report report;
};


#include <chrono>

#include <chrono>
#include <chrono>
#include <chrono>
inline void kernel_deriche_0_1_0(_deriche_t *__state, const float& dace_alpha_2, float* dace_imgIn_4, float* dace_imgOut_1, int& dace_print_4, float* dace_y1_1, float* dace_y2_1) {
    float dace_xm1_0;
    float dace_tm1_0;
    float dace_ym1_0;
    float dace_ym2_0;
    float dace_xp1_0;
    float dace_xp2_0;
    float dace_tp1_0;
    float dace_tp2_0;
    float dace_yp1_0;
    float dace_yp2_0;
    float dace_a1_0;
    float dace_a2_0;
    float dace_a3_0;
    float dace_a4_0;
    float dace_a5_0;
    float dace_a6_0;
    float dace_a7_0;
    float dace_a8_0;
    float dace_b1_0;
    float dace_b2_0;
    float dace_c1_0;
    float dace_c2_0;
    int i;
    long long j;
    int ii;
    long long jj;
    int j5;
    long long i5;
    int j6;
    long long i6;

    {
        float dace_k_0;

        {
            float alpha = dace_alpha_2;
            float k_out;

            ///////////////////
            k_out=(1.0f-expf(-alpha))*(1.0f-expf(-alpha))/(1.0f+2.0f*alpha*expf(-alpha)-expf(2.0f*alpha));
            ///////////////////

            dace_k_0 = k_out;
        }
        {
            float k = dace_k_0;
            float a1_out;
            float a5_out;

            ///////////////////
            a1_out=a5_out=k;
            ///////////////////

            dace_a1_0 = a1_out;
            dace_a5_0 = a5_out;
        }
        {
            float alpha = dace_alpha_2;
            float k = dace_k_0;
            float a2_out;
            float a6_out;

            ///////////////////
            a2_out=a6_out=k*expf(-alpha)*(alpha-1.0f);
            ///////////////////

            dace_a2_0 = a2_out;
            dace_a6_0 = a6_out;
        }
        {
            float alpha = dace_alpha_2;
            float k = dace_k_0;
            float a3_out;
            float a7_out;

            ///////////////////
            a3_out=a7_out=k*expf(-alpha)*(alpha+1.0f);
            ///////////////////

            dace_a3_0 = a3_out;
            dace_a7_0 = a7_out;
        }
        {
            float alpha = dace_alpha_2;
            float k = dace_k_0;
            float a4_out;
            float a8_out;

            ///////////////////
            a4_out=a8_out=-k*expf(-2.0f*alpha);
            ///////////////////

            dace_a4_0 = a4_out;
            dace_a8_0 = a8_out;
        }
        {
            float alpha = dace_alpha_2;
            float b1_out;

            ///////////////////
            b1_out=powf(2.0f,-alpha);
            ///////////////////

            dace_b1_0 = b1_out;
        }
        {
            float alpha = dace_alpha_2;
            float b2_out;

            ///////////////////
            b2_out=-expf(-2.0f*alpha);
            ///////////////////

            dace_b2_0 = b2_out;
        }
        {
            float c1_out;
            float c2_out;

            ///////////////////
            c1_out=c2_out=1;
            ///////////////////

            dace_c1_0 = c1_out;
            dace_c2_0 = c2_out;
        }

    }


    for (i = 0; (i < 4096); i = (i + 1)) {
        {

            {
                float ym1_out;

                ///////////////////
                ym1_out=0.0f;
                ///////////////////

                dace_ym1_0 = ym1_out;
            }
            {
                float ym2_out;

                ///////////////////
                ym2_out=0.0f;
                ///////////////////

                dace_ym2_0 = ym2_out;
            }
            {
                float xm1_out;

                ///////////////////
                xm1_out=0.0f;
                ///////////////////

                dace_xm1_0 = xm1_out;
            }

        }


        for (j = 0; (j < 2160); j = (j + 1)) {
            {

                {
                    float a1 = dace_a1_0;
                    float a2 = dace_a2_0;
                    float b1 = dace_b1_0;
                    float b2 = dace_b2_0;
                    float imgIn_1_1 = dace_imgIn_4[((2160 * i) + j)];
                    float xm1 = dace_xm1_0;
                    float ym1 = dace_ym1_0;
                    float ym2 = dace_ym2_0;
                    float y1_out_1_1;

                    ///////////////////
                    y1_out_1_1=a1*imgIn_1_1+a2*xm1+b1*ym1+b2*ym2;
                    ///////////////////

                    dace_y1_1[((2160 * i) + j)] = y1_out_1_1;
                }

            }
            {

                {
                    float imgIn_1_1 = dace_imgIn_4[((2160 * i) + j)];
                    float xm1_out;

                    ///////////////////
                    xm1_out=imgIn_1_1;
                    ///////////////////

                    dace_xm1_0 = xm1_out;
                }
                {
                    float ym1 = dace_ym1_0;
                    float ym2_out;

                    ///////////////////
                    ym2_out=ym1;
                    ///////////////////

                    dace_ym2_0 = ym2_out;
                }

            }
            {

                {
                    float y1_1_1 = dace_y1_1[((2160 * i) + j)];
                    float ym1_out;

                    ///////////////////
                    ym1_out=y1_1_1;
                    ///////////////////

                    dace_ym1_0 = ym1_out;
                }

            }

        }


    }



    for (ii = 0; (ii < 4096); ii = (ii + 1)) {
        {

            {
                float yp1_out;

                ///////////////////
                yp1_out=0.0f;
                ///////////////////

                dace_yp1_0 = yp1_out;
            }
            {
                float yp2_out;

                ///////////////////
                yp2_out=0.0f;
                ///////////////////

                dace_yp2_0 = yp2_out;
            }
            {
                float xp1_out;

                ///////////////////
                xp1_out=0.0f;
                ///////////////////

                dace_xp1_0 = xp1_out;
            }
            {
                float xp2_out;

                ///////////////////
                xp2_out=0.0f;
                ///////////////////

                dace_xp2_0 = xp2_out;
            }

        }


        for (jj = (2160 - 1); (jj >= 0); jj = (jj - 1)) {
            {

                {
                    float a3 = dace_a3_0;
                    float a4 = dace_a4_0;
                    float b1 = dace_b1_0;
                    float b2 = dace_b2_0;
                    float xp1 = dace_xp1_0;
                    float xp2 = dace_xp2_0;
                    float yp1 = dace_yp1_0;
                    float yp2 = dace_yp2_0;
                    float y2_out_1_1;

                    ///////////////////
                    y2_out_1_1=a3*xp1+a4*xp2+b1*yp1+b2*yp2;
                    ///////////////////

                    dace_y2_1[((2160 * ii) + jj)] = y2_out_1_1;
                }

            }
            {

                {
                    float xp1 = dace_xp1_0;
                    float xp2_out;

                    ///////////////////
                    xp2_out=xp1;
                    ///////////////////

                    dace_xp2_0 = xp2_out;
                }

            }
            {

                {
                    float imgIn_1_1 = dace_imgIn_4[((2160 * ii) + jj)];
                    float xp1_out;

                    ///////////////////
                    xp1_out=imgIn_1_1;
                    ///////////////////

                    dace_xp1_0 = xp1_out;
                }
                {
                    float yp1 = dace_yp1_0;
                    float yp2_out;

                    ///////////////////
                    yp2_out=yp1;
                    ///////////////////

                    dace_yp2_0 = yp2_out;
                }

            }
            {

                {
                    float y2_1_1 = dace_y2_1[((2160 * ii) + jj)];
                    float yp1_out;

                    ///////////////////
                    yp1_out=y2_1_1;
                    ///////////////////

                    dace_yp1_0 = yp1_out;
                }

            }

        }


    }
    {

        {
            for (auto iiii = 0; iiii < 4096; iiii += 1) {
                for (auto jjjj = 0; jjjj < 2160; jjjj += 1) {
                    {
                        float y1_1_1 = dace_y1_1[((2160 * iiii) + jjjj)];
                        float c1 = dace_c1_0;
                        float y2_1_1 = dace_y2_1[((2160 * iiii) + jjjj)];
                        float imgOut_out_1_1;

                        ///////////////////
                        imgOut_out_1_1=c1*(y1_1_1+y2_1_1);
                        ///////////////////

                        dace_imgOut_1[((2160 * iiii) + jjjj)] = imgOut_out_1_1;
                    }
                }
            }
        }

    }


    for (j5 = 0; (j5 < 2160); j5 = (j5 + 1)) {
        {

            {
                float tm1_out;

                ///////////////////
                tm1_out=0.0f;
                ///////////////////

                dace_tm1_0 = tm1_out;
            }
            {
                float ym1_out;

                ///////////////////
                ym1_out=0.0f;
                ///////////////////

                dace_ym1_0 = ym1_out;
            }
            {
                float ym2_out;

                ///////////////////
                ym2_out=0.0f;
                ///////////////////

                dace_ym2_0 = ym2_out;
            }

        }


        for (i5 = 0; (i5 < 4096); i5 = (i5 + 1)) {
            {

                {
                    float a5 = dace_a5_0;
                    float a6 = dace_a6_0;
                    float b1 = dace_b1_0;
                    float b2 = dace_b2_0;
                    float imgOut_1_1 = dace_imgOut_1[((2160 * i5) + j5)];
                    float tm1 = dace_tm1_0;
                    float ym1 = dace_ym1_0;
                    float ym2 = dace_ym2_0;
                    float y1_out_1_1;

                    ///////////////////
                    y1_out_1_1=a5*imgOut_1_1+a6*tm1+b1*ym1+b2*ym2;
                    ///////////////////

                    dace_y1_1[((2160 * i5) + j5)] = y1_out_1_1;
                }

            }
            {

                {
                    float imgOut_1_1 = dace_imgOut_1[((2160 * i5) + j5)];
                    float tm1_out;

                    ///////////////////
                    tm1_out=imgOut_1_1;
                    ///////////////////

                    dace_tm1_0 = tm1_out;
                }
                {
                    float ym1 = dace_ym1_0;
                    float ym2_out;

                    ///////////////////
                    ym2_out=ym1;
                    ///////////////////

                    dace_ym2_0 = ym2_out;
                }

            }
            {

                {
                    float y1_1_1 = dace_y1_1[((2160 * i5) + j5)];
                    float ym1_out;

                    ///////////////////
                    ym1_out=y1_1_1;
                    ///////////////////

                    dace_ym1_0 = ym1_out;
                }

            }

        }


    }



    for (j6 = 0; (j6 < 2160); j6 = (j6 + 1)) {
        {

            {
                float tp1_out;

                ///////////////////
                tp1_out=0.0f;
                ///////////////////

                dace_tp1_0 = tp1_out;
            }
            {
                float tp2_out;

                ///////////////////
                tp2_out=0.0f;
                ///////////////////

                dace_tp2_0 = tp2_out;
            }
            {
                float yp1_out;

                ///////////////////
                yp1_out=0.0f;
                ///////////////////

                dace_yp1_0 = yp1_out;
            }
            {
                float yp2_out;

                ///////////////////
                yp2_out=0.0f;
                ///////////////////

                dace_yp2_0 = yp2_out;
            }

        }


        for (i6 = (4096 - 1); (i6 >= 0); i6 = (i6 - 1)) {
            {

                {
                    float a7 = dace_a7_0;
                    float a8 = dace_a8_0;
                    float b1 = dace_b1_0;
                    float b2 = dace_b2_0;
                    float tp1 = dace_tp1_0;
                    float tp2 = dace_tp2_0;
                    float yp1 = dace_yp1_0;
                    float yp2 = dace_yp2_0;
                    float y2_out_1_1;

                    ///////////////////
                    y2_out_1_1=a7*tp1+a8*tp2+b1*yp1+b2*yp2;
                    ///////////////////

                    dace_y2_1[((2160 * i6) + j6)] = y2_out_1_1;
                }

            }
            {

                {
                    float tp1 = dace_tp1_0;
                    float tp2_out;

                    ///////////////////
                    tp2_out=tp1;
                    ///////////////////

                    dace_tp2_0 = tp2_out;
                }

            }
            {

                {
                    float imgOut_1_1 = dace_imgOut_1[((2160 * i6) + j6)];
                    float tp1_out;

                    ///////////////////
                    tp1_out=imgOut_1_1;
                    ///////////////////

                    dace_tp1_0 = tp1_out;
                }
                {
                    float yp1 = dace_yp1_0;
                    float yp2_out;

                    ///////////////////
                    yp2_out=yp1;
                    ///////////////////

                    dace_yp2_0 = yp2_out;
                }

            }
            {

                {
                    float y2_1_1 = dace_y2_1[((2160 * i6) + j6)];
                    float yp1_out;

                    ///////////////////
                    yp1_out=y2_1_1;
                    ///////////////////

                    dace_yp1_0 = yp1_out;
                }

            }

        }


    }
    {

        {
            for (auto i7 = 0; i7 < 4096; i7 += 1) {
                for (auto j7 = 0; j7 < 2160; j7 += 1) {
                    {
                        float c2 = dace_c2_0;
                        float y1_1_1 = dace_y1_1[((2160 * i7) + j7)];
                        float y2_1_1 = dace_y2_1[((2160 * i7) + j7)];
                        float imgOut_out_1_1;

                        ///////////////////
                        imgOut_out_1_1=c2*(y1_1_1+y2_1_1);
                        ///////////////////

                        dace_imgOut_1[((2160 * i7) + j7)] = imgOut_out_1_1;
                    }
                }
            }
        }

    }
    
}

void __program__deriche_internal(_deriche_t *__state, char * __restrict__ argv_loc, int _argcount, int argc_loc)
{
    auto __dace_tbegin_0 = std::chrono::high_resolution_clock::now();
    int print;
    float dace_alpha_0;
    bool dace_tmp_if_1_0;
    long long tmp_for_1;
    long long tmp_for_2;

    {

        {
            float alpha_out;

            ///////////////////
            alpha_out=0.25;
            ///////////////////

            dace_alpha_0 = alpha_out;
        }
        {
            for (auto i = 0; i < 4096; i += 1) {
                for (auto j = 0; j < 2160; j += 1) {
                    {
                        float imgIn_out_1_1;

                        ///////////////////
                        imgIn_out_1_1=(float)(((313*i+991*j)%65536))/65535.0f;
                        ///////////////////

                        __state->__0_dace_imgIn_0[((2160 * i) + j)] = imgIn_out_1_1;
                    }
                }
            }
        }

    }
    dace_tmp_if_1_0 = (argc_loc > 42);
    {

        kernel_deriche_0_1_0(__state, dace_alpha_0, &__state->__0_dace_imgIn_0[0], &__state->__0_dace_imgOut_0[0], print, &__state->__0_dace_y1_0[0], &__state->__0_dace_y2_0[0]);

    }
    if ((dace_tmp_if_1_0 != 0)) {
        {

            {
                int dace_print_17_task = print;
                int dace_print_17_task_out;

                ///////////////////
                fprintf((stderr),"==BEGIN DUMP_ARRAYS==\n");
                ///////////////////

                print = dace_print_17_task_out;
            }
            {
                int dace_print_17_task = print;
                int dace_print_17_task_out;

                ///////////////////
                fprintf((stderr),"begin dump: %s","imgOut");
                ///////////////////

                print = dace_print_17_task_out;
            }

        }


        for (tmp_for_1 = 0; (tmp_for_1 < 4096); tmp_for_1 = (tmp_for_1 + 1)) {



            for (tmp_for_2 = 0; (tmp_for_2 < 2160); tmp_for_2 = (tmp_for_2 + 1)) {
                {

                    {
                        int dace_print_19_task = print;
                        int dace_print_19_task_out;

                        ///////////////////
                        fprintf((stderr),"\n");
                        ///////////////////

                        print = dace_print_19_task_out;
                    }
                    {
                        int dace_print_19_task = print;
                        float imgOut_1_1 = __state->__0_dace_imgOut_0[((2160 * tmp_for_1) + tmp_for_2)];
                        int dace_print_19_task_out;

                        ///////////////////
                        fprintf((stderr),"%0.2f ",imgOut_1_1);
                        ///////////////////

                        print = dace_print_19_task_out;
                    }

                }

            }


        }

    }

    auto __dace_tend_0 = std::chrono::high_resolution_clock::now();
    unsigned long int __dace_ts_start_0 = std::chrono::duration_cast<std::chrono::microseconds>(__dace_tbegin_0.time_since_epoch()).count();
    unsigned long int __dace_ts_end_0 = std::chrono::duration_cast<std::chrono::microseconds>(__dace_tend_0.time_since_epoch()).count();
    __state->report.add_completion("SDFG _deriche", "Timer", __dace_ts_start_0, __dace_ts_end_0, 0, -1, -1);
}

DACE_EXPORTED void __program__deriche(_deriche_t *__state, char * __restrict__ argv_loc, int _argcount, int argc_loc)
{
    __program__deriche_internal(__state, argv_loc, _argcount, argc_loc);
}

DACE_EXPORTED _deriche_t *__dace_init__deriche(int _argcount)
{
    int __result = 0;
    _deriche_t *__state = new _deriche_t;


    __state->__0_dace_imgIn_0 = new float DACE_ALIGN(64)[8847360];
    __state->__0_dace_imgOut_0 = new float DACE_ALIGN(64)[8847360];
    __state->__0_dace_y1_0 = new float DACE_ALIGN(64)[8847360];
    __state->__0_dace_y2_0 = new float DACE_ALIGN(64)[8847360];

    if (__result) {
        delete __state;
        return nullptr;
    }
    return __state;
}

DACE_EXPORTED void __dace_exit__deriche(_deriche_t *__state)
{
    __state->report.save(".dacecache/_deriche/perf", __HASH__deriche);
    delete[] __state->__0_dace_imgIn_0;
    delete[] __state->__0_dace_imgOut_0;
    delete[] __state->__0_dace_y1_0;
    delete[] __state->__0_dace_y2_0;
    delete __state;
}

