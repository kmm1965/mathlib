#include "average_operator.hpp"
#include "first_derivative_operator.hpp"
#include "periodic_index.hpp"
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
    const double L = 1e-2; // [m]
	const double h = L / N; // [m]
    const double dt = 0.25e-7; // [s]
    const double CS = 200; // [m/s]
    const double tau = 0.5*h/CS; //[s]
    const double eta = 5e-4; // [Pa*s]
    const double lambda1 = 6e-4;
    //const double M = 1e-10;
    const double M = 1e-9;
    const double A_psi = 1e+4;
    const double beta = std::sqrt(2*A_psi/lambda1);

    const size_t CELL_N = N;
    const size_t EDGE_N = CELL_N; // number of edges
	vector_grid_function<tag_aux, double> w(EDGE_N), jm(EDGE_N), P(EDGE_N);
	vector_grid_function<tag_main, double> C(CELL_N), mu(CELL_N), u(CELL_N), rho(CELL_N, 0, 1.0),
                                           Psi0(CELL_N), E_lmd(CELL_N), Psi1(CELL_N), Qw(CELL_N),
                                           rho_new(CELL_N), rho_C_new(CELL_N), rho_u_new(CELL_N), C_new(CELL_N),
                                           totEnrgGF(CELL_N),
                                           lmdEnrgGF(CELL_N),
                                           kinEnrgGF(CELL_N),
                                           psiEnrgGF(CELL_N);

    //double x = -1;
    C = initializer<tag_main, double>(CELL_N);

    const periodic_index index(CELL_N);
	const average_operator<tag_main> avr(index);   // operator maps from cell to edge
	const average_operator<tag_aux> avrSt(index); // operator maps from edge to cell
	const first_derivative_operator<double, tag_main> drv(index, h);
	const first_derivative_operator<double, tag_aux> drvSt(index, h);

    const int WIter = 1;

    for (int i = 0; i <= max_iter; i++){
        Psi1  = (CS * CS) * log(rho);

        E_lmd = (lambda1 * 0.5) * avrSt(sqr(drv(C)));
        Psi0  = A_psi * sqr(C * (1.0 - C)) + Psi1;

        lmdEnrgGF = rho * E_lmd;
        psiEnrgGF = rho * Psi0;
        kinEnrgGF = 0.5 * rho * sqr(u);
        totEnrgGF = lmdEnrgGF + psiEnrgGF  + kinEnrgGF;
        const double
            totEnrg = MATH_ACCUMULATE(totEnrgGF.begin(), totEnrgGF.end(), 0.0),
            kinEnrg = MATH_ACCUMULATE(kinEnrgGF.begin(), kinEnrgGF.end(), 0.0),
            lmdEnrg = MATH_ACCUMULATE(lmdEnrgGF.begin(), lmdEnrgGF.end(), 0.0),
            psiEnrg = MATH_ACCUMULATE(psiEnrgGF.begin(), psiEnrgGF.end(), 0.0);

        std::cout << i << " Energy " << totEnrg * h << std::endl;
        os_enrg << i * dt << ' ' <<  totEnrg * h << ' ' 
                << lmdEnrg * h << ' ' << psiEnrg * h << ' ' << kinEnrg * h <<  std::endl;

////////   NEW   ///////
        mu = A_psi * 2.0 * C * (1.0 - C) * (1.0 - 2.0 * C) - lambda1 * drvSt(avr(rho) * drv(C)) / rho;

        w  = tau * (avr(u) * drv(u) + drv(Psi0) + drv(E_lmd) - avr(mu) * drv(C));
        P  = (4.0 / 3.0) * eta * drv(u) + avr(rho) * avr(u) * w;
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

        if (rho[5] != rho[5])
            abort();

        if (i % WIter == 0){
            std::ofstream os;
            os << std::setprecision(16);
            os << std::scientific;
            char buf[256];
            sprintf(buf, "results_%d.dat", i / WIter);
            os.open(buf, std::ios::out);

            for (int i = 0; i < N; ++i)
                os << h * (i + 0.5) << ' ' <<  C[i] << ' ' << u[i] << ' ' << rho[i]  << std::endl;
            os.close();
        }
    }
    return 0;
}
