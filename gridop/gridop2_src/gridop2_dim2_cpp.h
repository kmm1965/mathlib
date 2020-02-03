#include "average_operator.hpp"
#include "first_derivative_operator_dim2.hpp"
#include "cyclic_index.hpp"
#include "symmetric_index.hpp"

int main1(int argc, char* argv[])
{
    std::cout << std::setprecision(16) << std::scientific;
    //const int TN = 10;
    const int max_iter = 100;
    const size_t N = 1000;
    const length_type
        L = 1e-2, // [m]
        R = 0.16 * L,
        R_ = R * 1.08,
        y0 = 0.5 * L - R_ / std::sqrt(3.0),
        y1 = y0,
        y2 = y0 + R_ * std::sqrt(3.0),
        x0 = 0.5 * L - R_,
        x1 = x0 + 2. * R_,
        x2 = (x0 + x1) * 0.5,
        h = L / double(N); // [m]
    const time_type dt = 3e-12; // [s]
    const velocity_type CS = 500; // [m/s]
    const energy_type CS2 = CS * CS;
    const time_type tau = 0.5 * h / CS; //[s]
    const typename units::multiply_typeof_helper<velocity_type, length_type>::type eta = 5e-4 * 2; // [Pa*s]
    const units::quantity<units::quantity_dim<0, 4, -2, 0, 0, 0, 0> > lambda1 = 2e-4 * 2;
    const time_type M = 2e-8; //7e-8,
    const energy_type A_psi = 2e+4 * 2;
    const units::quantity<units::quantity_dim<0, -1, 0, 0, 0, 0, 0> > b = func::sqrt(2.0 * A_psi / lambda1);

    const size_t
        CNTR_N = N, // number of cell centers
        EDGE_N = CNTR_N, // number of edges
        sz_edge = EDGE_N * EDGE_N,
        sz_cntr = CNTR_N * CNTR_N;

    typedef dim2_index<cyclic_index, cyclic_index> index_type;
    typedef get_value_type_t<index_type> index_value_type;

    const cyclic_index row_index(CNTR_N), column_index(CNTR_N);
    const index_type index(row_index, column_index);

    vector_grid_function<tag_edge_x, velocity_type> mx(sz_edge), mx_new(sz_edge), my_x(sz_edge), Jx(sz_edge);
    vector_grid_function<tag_edge_x, energy_type> P_tau_xx(sz_edge), P_tau_xy(sz_edge), P_NS_xx(sz_edge), P_NS_xy(sz_edge);

    vector_grid_function<tag_edge_y, velocity_type> my(sz_edge), my_new(sz_edge), mx_y(sz_edge), Jy(sz_edge);
    vector_grid_function<tag_edge_y, energy_type> P_tau_yy(sz_edge), P_tau_yx(sz_edge), P_NS_yy(sz_edge), P_NS_yx(sz_edge);

    vector_grid_function<tag_cntr, nodim> cn(sz_cntr);
    vector_grid_function<tag_cntr, nodim> rh(sz_cntr, 0, 2.0), rh_new(sz_cntr), rh_cn_new(sz_cntr);
    vector_grid_function<tag_cntr, velocity_type> ux(sz_edge), uy(sz_edge), rh_ux_new(sz_cntr), rh_uy_new(sz_cntr);
    vector_grid_function<tag_cntr, energy_type> mu(sz_cntr), E_lmd(sz_cntr), G(sz_cntr), Psi0(sz_cntr), Psi1(sz_cntr), totEnrg(sz_cntr);

    const average_operator_x<tag_main> avr_x(index);
    const average_operator_x<tag_aux>  avrSt_x(index);
    const average_operator_y<tag_main> avr_y(index);
    const average_operator_y<tag_aux>  avrSt_y(index);

    const first_derivative_operator_x<tag_main> drv_x(index, h);
    const first_derivative_operator_x<tag_aux>  drvSt_x(index, h);
    const first_derivative_operator_y<tag_main> drv_y(index, h);
    const first_derivative_operator_y<tag_aux>  drvSt_y(index, h);

    // initial conditions
#ifdef __CUDACC__
    HOST_VECTOR<nodim> hcn(cn.size());
#else
    MATH_VECTOR_BASE_CLASS<nodim>& hcn = cn;
#endif
    boost::timer::cpu_timer t;
    for (int i = 0; i < CNTR_N; ++i) {
        for (int j = 0; j < CNTR_N; ++j) {
            const length_type
                x = (i + 0.5) * h,
                y = (j + 0.5) * h,
                r0 = func::sqrt(func::sqr(x - x0) + func::sqr(y - y0)),
                r1 = func::sqrt(func::sqr(x - x1) + func::sqr(y - y1)),
                r2 = func::sqrt(func::sqr(x - x2) + func::sqr(y - y2));

            hcn[index[index_value_type(i, j)]] = 0.5 * (1 + func::tanh(b * 0.5 * (R - r0)))
                + 0.5 * (1 + func::tanh(b * 0.5 * (R - r1)))
                + 0.5 * (1 + func::tanh(b * 0.5 * (R - r2)));
        }
    }
#ifdef __CUDACC__
    cn.assign(hcn);
#endif
    for (int iter = 0; iter < max_iter /*true*/; iter++) { // infinite cycle
        Psi1 = CS2 * log(rh);
        E_lmd = (0.5 * lambda1) * (
            avrSt_x(sqr(drv_x(cn)))
            + avrSt_y(sqr(drv_y(cn))));
        Psi0 = A_psi * (sqr((cn * (1.0 - cn)))) + Psi1;

        G = Psi0 + CS2 + E_lmd;

        mu = A_psi * 2.0 * cn * (1.0 - cn) * (1.0 - 2.0 * cn) -
            lambda1 * (
                  drvSt_x(avr_x(rh) * drv_x(cn))
         	    + drvSt_y(avr_y(rh) * drv_y(cn))
            ) / rh;

        mx = tau * (avr_x(rh) * (avr_x(ux) * drv_x(ux) + drv_x(G) - avr_x(mu) * drv_x(cn))
            + avr_x(rh * avrSt_y(avr_y(uy) * drv_y(ux))));

        my = tau * (avr_y(rh) * (avr_y(uy) * drv_y(uy) + drv_y(G) - avr_y(mu) * drv_y(cn))
            + avr_y(rh * avrSt_x(avr_x(ux) * drv_x(uy))));

        mx_y = tau * (avr_y(rh * avrSt_x(avr_x(ux) * drv_x(ux) + drv_x(G) - avr_x(mu) * drv_x(cn)))
            + avr_y(rh) * avr_y(uy) * drv_y(ux));

        my_x = tau * (avr_x(rh * avrSt_y(avr_y(uy) * drv_y(uy) + drv_y(G) - avr_y(mu) * drv_y(cn)))
            + avr_x(rh) * avr_x(ux) * drv_x(uy));

        Jx = avr_x(rh) * avr_x(ux) - mx;
        Jy = avr_y(rh) * avr_y(uy) - my;

        P_NS_xx = eta * (drv_x(ux) - (2.0 / 3.0) * avrSt_y(avr_x(drv_y(uy))));
        P_NS_yy = eta * (drv_y(uy) - (2.0 / 3.0) * avrSt_x(avr_y(drv_x(ux))));
        P_NS_xy = eta * (drv_x(uy) + avrSt_y(avr_x(drv_y(ux))));
        P_NS_yx = eta * (drv_y(ux) + avrSt_x(avr_y(drv_x(uy))));

        P_tau_xx = avr_x(ux) * mx;
        P_tau_xy = avr_x(ux) * my_x;
        P_tau_yy = avr_y(uy) * my;
        P_tau_yx = avr_y(uy) * mx_y;

        rh_new = rh - dt * (drvSt_x(Jx) + drvSt_y(Jy));

        rh_ux_new = rh * ux - dt * (
            drvSt_x(Jx * avr_x(ux)) + drvSt_y(Jy * avr_y(ux))
            + avrSt_x(avr_x(rh) * drv_x(G))
            - drvSt_x(P_NS_xx + P_tau_xx) - drvSt_y(P_NS_yx + P_tau_yx)
            - avrSt_x(avr_x(rh) * avr_x(mu) * drv_x(cn)));

        rh_uy_new = rh * uy - dt * (
            drvSt_x(Jx * avr_x(uy)) + drvSt_y(Jy * avr_y(uy))
            + avrSt_y(avr_y(rh) * drv_y(G))
            - drvSt_x(P_NS_xy + P_tau_xy) - drvSt_y(P_NS_yy + P_tau_yy)
            - avrSt_y(avr_y(rh) * avr_y(mu) * drv_y(cn)));

        rh_cn_new = rh * cn - dt * (
            drvSt_x(Jx * avr_x(cn)) + drvSt_y(Jy * avr_y(cn))
            - drvSt_x(M * drv_x(mu)) - drvSt_y(M * drv_y(mu)));

        totEnrg = rh * E_lmd + rh * Psi0 + 0.5 * rh * (ux * ux + uy * uy);

        if (iter % (max_iter / 10) == 0)
            std::cout << "Iter " << iter << " total system energy "
            << MATH_ACCUMULATE(totEnrg.begin(), totEnrg.end(), energy_type(0.0)) << std::endl;

        rh = rh_new;
        //cn = rh_cn_new / rh_new;
        cn = rh_cn_new / rh_new;
        ux = rh_ux_new / rh_new;
        uy = rh_uy_new / rh_new;

        //if (iter%100 == 0)
        //    writeVtk(iter, cn, rh, h, CNTR_N);
    }
    t.stop();
    std::cout << t.format(2, "Calc time=%w seconds") << std::endl;
    return 0;
}
