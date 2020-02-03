#include "average_operator.hpp"
#include "first_derivative_operator_dim.hpp"
#include "initializer.hpp"

int main1(int argc, char* argv[])
{
    std::ofstream os_enrg;
    os_enrg << std::setprecision(16);
    os_enrg << std::scientific;
    os_enrg.open("energy.dat", std::ios::out);

    std::cout << std::setprecision(16);
    std::cout << std::scientific;
    //const int TN = 10;
    const int max_iter = 100;
    const size_t N = 100;
    const length_type
        L = 1e-2 * si::meter, // [m]
        h = L / (double) N; // [m]
    const time_type dt = 0.25e-7 * si::second; // [s]
    const velocity_type CS = 200 * si::meters_per_second; // [m/s]
    const time_type tau = 0.5 * h / CS; //[s]
    const typename units::multiply_typeof_helper<velocity_type, length_type>::type eta = 5e-4 * (si::meters_per_second * si::meter); // [Pa*s]
    const units::quantity<units::unit<typename units::derived_dimension<units::length_base_dimension, 4, units::time_base_dimension, -2>::type, si::system> >
        lambda1 = 6e-4 * (si::meters_per_second * si::meters_per_second * si::meter * si::meter);
    //const double M = 1e-10;
    const time_type M = 1e-9 * si::second;
    const energy_type A_psi = 1e+4 * (si::meters_per_second * si::meters_per_second);
    const units::quantity<typename units::divide_typeof_helper<si::dimensionless, si::length>::type> beta = sqrt(2.0 * A_psi / lambda1);

    const size_t CELL_N = N;
    const size_t EDGE_N = CELL_N; // number of edges


    vector_grid_function<tag_aux, velocity_type> w(EDGE_N), jm(EDGE_N);
    vector_grid_function<tag_aux, energy_type> P(EDGE_N);

    vector_grid_function<tag_main, nodim> C(CELL_N), C_new(CELL_N), rho(CELL_N, 0, 1.0), rho_new(CELL_N), rho_C_new(CELL_N);
    vector_grid_function<tag_main, velocity_type> u(CELL_N), rho_u_new(CELL_N);
    vector_grid_function<tag_main, energy_type> Psi0(CELL_N), E_lmd(CELL_N), Psi1(CELL_N),
        totEnrgGF(CELL_N), lmdEnrgGF(CELL_N), kinEnrgGF(CELL_N), psiEnrgGF(CELL_N), mu(CELL_N);
    vector_grid_function<tag_main, double> Qw(CELL_N);

    vector_grid_function<tag_main, density_type> _rho(CELL_N);
    vector_grid_function<tag_main, specific_energy_type> _E(CELL_N);

    //double x = -1;
    C = initializer<tag_main, double>(CELL_N);

    const periodic_index index(CELL_N);
    const average_operator<tag_main> avr(index);   // operator maps from cell to edge
    const average_operator<tag_aux> avrSt(index); // operator maps from edge to cell
    const first_derivative_operator<tag_main> drv(index, h);
    const first_derivative_operator<tag_aux> drvSt(index, h);

    const int WIter = 1;

    for (int i = 0; i <= max_iter; i++){
        Psi1 = (CS * CS) * log(rho);

        E_lmd = (lambda1 * 0.5) * avrSt(sqr(drv(C)));
        Psi0 = A_psi * sqr(C * (1.0 - C)) + Psi1;

        lmdEnrgGF = rho * E_lmd;
        psiEnrgGF = rho * Psi0;
        kinEnrgGF = 0.5 * rho * sqr(u);
        totEnrgGF = lmdEnrgGF + psiEnrgGF + kinEnrgGF;
        const energy_type
            totEnrg = MATH_ACCUMULATE(totEnrgGF.begin(), totEnrgGF.end(), 0.0 * (si::joule / si::kilogram)),
            kinEnrg = MATH_ACCUMULATE(kinEnrgGF.begin(), kinEnrgGF.end(), 0.0 * (si::joule / si::kilogram)),
            lmdEnrg = MATH_ACCUMULATE(lmdEnrgGF.begin(), lmdEnrgGF.end(), 0.0 * (si::joule / si::kilogram)),
            psiEnrg = MATH_ACCUMULATE(psiEnrgGF.begin(), psiEnrgGF.end(), 0.0 * (si::joule / si::kilogram));

        std::cout << i << " Energy " << (totEnrg * h).value() << std::endl;
        os_enrg << ((double)i * dt).value() << ' ' << (totEnrg * h).value() << ' '
            << (lmdEnrg * h).value() << ' ' << (psiEnrg * h).value() << ' ' << (kinEnrg * h).value() << std::endl;

        ////////   NEW   ///////
        mu = A_psi * 2.0 * C * (1.0 - C) * (1.0 - 2.0 * C) - lambda1 * drvSt(avr(rho) * drv(C)) / rho;

        w = tau * (avr(u) * drv(u) + drv(Psi0) + drv(E_lmd) - avr(mu) * drv(C));
        P = (4.0 / 3.0) * eta * drv(u) + avr(rho) * avr(u) * w;
        jm = avr(rho) * (avr(u) - w);

        rho_new = rho - dt * drvSt(jm);
        rho_u_new = rho * u + dt * (
            - drvSt(jm * avr(u)) - avrSt(avr(rho) * drv(Psi0 + E_lmd))
            + drvSt(P) + mu * avrSt(avr(rho) * drv(C)));
        rho_C_new = rho * C + dt * (
            - drvSt(jm * avr(C) - 0.25 * h * h * avr(rho) * drv(C) * drv(u))
            + drvSt(M * drv(mu)));

        rho = rho_new;
        u = rho_u_new / rho_new;
        C = rho_C_new / rho_new;

        if ((nodim)rho[5] != (nodim)rho[5])
            abort();

        if (i % WIter == 0){
            std::ofstream os;
            os << std::setprecision(16);
            os << std::scientific;
            char buf[256];
            sprintf(buf, "results_%d.dat", i / WIter);
            os.open(buf, std::ios::out);

            for (int i = 0; i < N; ++i)
                os << (h * (i + 0.5)).value() << ' ' << ((nodim)C[i]).value() << ' ' << ((velocity_type)u[i]).value() << ' ' << ((nodim)rho[i]).value() << std::endl;
            os.close();
        }
    }
    return 0;
}
