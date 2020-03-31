#include "pch.h"
#include "average_operator.hpp"
#include "first_derivative_operator.hpp"
#include "periodic_index.hpp"
#include "symmetric_index.hpp"
#include <vtkRectilinearGridWriter.h>
#include <vtkRectilinearGrid.h>
#include <vtkStructuredPoints.h>
#include <vtkStructuredPointsWriter.h>
#include <vtkSmartPointer.h>
#include <vtkCellData.h>
#include <vtkDoubleArray.h> 
#include <vtkIntArray.h>
#include <string>


void writeVtk(int iter, 
              const vector_grid_function<tag_cntr, double> &gf, 
              const vector_grid_function<tag_cntr, double> &rh, 
              double sp_step, size_t size)
{
        std::string strVtkFileName(std::string("output_") 
                                 + std::to_string(iter) 
                                 + std::string(".vtk"));
        vtkSmartPointer<vtkDoubleArray> pXCoords = vtkDoubleArray::New(); 
        vtkSmartPointer<vtkDoubleArray> pYCoords = vtkDoubleArray::New();
        vtkSmartPointer<vtkDoubleArray> pZCoords = vtkDoubleArray::New();

        vtkSmartPointer<vtkRectilinearGridWriter> pVtkWriter = vtkRectilinearGridWriter::New();
        vtkSmartPointer<vtkRectilinearGrid> pVtkGrid = vtkRectilinearGrid::New();
#if (VTK_MAJOR_VERSION < 6)
        pVtkWriter->SetInput(pVtkGrid);
#else
        pVtkWriter->SetInputData(pVtkGrid);
#endif
        pVtkWriter->SetFileName(strVtkFileName.c_str());
        pVtkWriter->SetFileType(VTK_BINARY);

        vtkSmartPointer<vtkDoubleArray> pConcArray  = vtkDoubleArray::New();
        vtkSmartPointer<vtkDoubleArray> pDensArray  = vtkDoubleArray::New();
        pConcArray->SetNumberOfValues(size*size);
        pDensArray->SetNumberOfValues(size*size);
        pConcArray->SetName("C");
        pDensArray->SetName("Rho");

        const int size_x = size;
        const int size_y = size;

        for(int i = 0; i < size_x + 1; i++)
            pXCoords->InsertNextValue(i*sp_step);

        for(int j = 0; j < size_y + 1; j++)
            pYCoords->InsertNextValue(j*sp_step);

        pZCoords->InsertNextValue(0.0); // due to 2D

        pVtkGrid->SetDimensions(size_x + 1,
                                size_y + 1,
                                1);

        pVtkGrid->SetXCoordinates(pXCoords);
        pVtkGrid->SetYCoordinates(pYCoords);
        pVtkGrid->SetZCoordinates(pZCoords);

        pVtkGrid->GetCellData()->AddArray(pConcArray);
        pVtkGrid->GetCellData()->AddArray(pDensArray);

        typedef dim2_index<periodic_index, periodic_index> index_type;
        typedef typename index_type::value_type index_value_type;
        const periodic_index row_index(size_x), column_index(size_y);
        index_type index(row_index, column_index);


        int ijk[3];
        for (int i = 0; i < size_x; ++i) 
        {
            for (int j = 0; j < size_y; ++j) 
            {
                ijk[0] = i; ijk[1] = j; ijk[2] = 0;
                const size_t idx = pVtkGrid->ComputeCellId(ijk);
                const double val = gf[index[index_value_type(i, j)]];
                const double rho = rh[index[index_value_type(i, j)]];
                pConcArray->SetValue(idx, val);
                pDensArray->SetValue(idx, rho);
            }
        }

        pVtkWriter->Write();
}

inline double sqr(double x){
	return x * x;
}

int main(int argc, char* argv[])
{   
    std::ofstream os_enrg;
    os_enrg << std::setprecision(16);
    os_enrg << std::scientific;
    os_enrg.open("energy.dat", std::ios::out);

    std::cout << std::setprecision(16);
    std::cout << std::scientific;
    const int TN = 10;
    const int max_iter = 100;
	const size_t N = 100;
    const double L = 1e-2; // [m]
    const double R = 0.16*L;
    const double R_ = R*1.08;
    const double y0 = 0.5*L - R_/std::sqrt(3.0);
    const double y1 = y0;
    const double y2 = y0+R_*std::sqrt(3.0);
    const double x0 = 0.5*L - R_;
    const double x1 = x0+2*R_;
    const double x2 = (x0+x1)*0.5;
	const double h = L/static_cast<double>(N); // [m]
    const double eps = 0.0;
    const double dt = 3.0e-8; // [s]
    const double CS = 500; // [m/s]
    const double CS2 = CS*CS; 
    const double tau = 0.5*h/CS; //[s]
    const double eta = 5e-4*2; // [Pa*s]
    const double lambda1 = 2e-4*2;
    const double A_psi = 2e+4*2;
    const double M = 2e-8; 

typedef dim2_index<periodic_index, periodic_index> index_type;
typedef typename index_type::value_type index_value_type;

    const int CNTR_N = N; // number of cell centers
    const int EDGE_N = CNTR_N; // number of edges
    const periodic_index row_index(CNTR_N), column_index(CNTR_N);
    index_type index(row_index, column_index);

	const size_t sz_edge = EDGE_N * EDGE_N, 
                 sz_cntr = CNTR_N * CNTR_N;
	vector_grid_function<tag_edge_x, double> mx(sz_edge), mx_new(sz_edge), Jx(sz_edge), my_x(sz_edge),
                                             P_NS_xx(sz_edge),  P_NS_xy(sz_edge),
                                             P_tau_xx(sz_edge), P_tau_xy(sz_edge);

	vector_grid_function<tag_edge_y, double> my(sz_edge), my_new(sz_edge), Jy(sz_edge), mx_y(sz_edge),
                                             P_NS_yy(sz_edge),  P_NS_yx(sz_edge),
                                             P_tau_yy(sz_edge), P_tau_yx(sz_edge);

	vector_grid_function<tag_cntr,   double> rh(sz_cntr), cn(sz_cntr), mu(sz_cntr), ux(sz_edge),
                                             rh_ux_new(sz_cntr), rh_uy_new(sz_cntr), totEnrg(sz_cntr),
                                             uy(sz_edge), rh_new(sz_cntr), rh_cn_new(sz_cntr), 
                                             G(sz_cntr) , E_lmd(sz_cntr), Psi0(sz_cntr), Psi1(sz_cntr);

	const average_operator_x<tag_main> avr_x  (index);  
	const average_operator_x<tag_aux>  avrSt_x(index); 
	const average_operator_y<tag_main> avr_y  (index);  
	const average_operator_y<tag_aux>  avrSt_y(index);

	const first_derivative_operator_x<double, tag_main> drv_x  (index, h);  
	const first_derivative_operator_x<double, tag_aux>  drvSt_x(index, h); 
	const first_derivative_operator_y<double, tag_main> drv_y  (index, h);  
	const first_derivative_operator_y<double, tag_aux>  drvSt_y(index, h); 


const double b = std::sqrt(2*A_psi/lambda1);
// initial conditions
    for (int i = 0; i < CNTR_N; ++i) {
		for (int j = 0; j < CNTR_N; ++j) {
			ux[index[index_value_type(i, j)]] = 0.0;
			uy[index[index_value_type(i, j)]] = 0.0;

            const double x = (i+0.5)*h;
            const double y = (j+0.5)*h;
            const double r0 = std::sqrt((x-x0)*(x-x0) + (y-y0)*(y-y0));
            const double r1 = std::sqrt((x-x1)*(x-x1) + (y-y1)*(y-y1));
            const double r2 = std::sqrt((x-x2)*(x-x2) + (y-y2)*(y-y2));

            rh[index[index_value_type(i, j)]] = 2.0;
            cn[index[index_value_type(i, j)]] = 0.5*(1 + std::tanh(b*0.5*(R - r0)))
                                              + 0.5*(1 + std::tanh(b*0.5*(R - r1)))
                                              + 0.5*(1 + std::tanh(b*0.5*(R - r2)));

		}
	}

    

    for (int iter = 0; true; iter++) // infinite cycle
    {   
        Psi1  = CS2*log(rh);
        E_lmd = 0.5*lambda1*(   avrSt_x(drv_x(cn)*drv_x(cn)) 
                              + avrSt_y(drv_y(cn)*drv_y(cn)));
        Psi0  = A_psi*cn*cn*(1.0-cn)*(1.0-cn) + Psi1;

        G  = Psi0 + CS2 + E_lmd;

        mu = A_psi*2.0*cn*(1.0-cn)*(1.0-2.0*cn) 
           - lambda1*(  drvSt_x(avr_x(rh)*drv_x(cn)) 
                      + drvSt_y(avr_y(rh)*drv_y(cn)))/rh;

        mx =   tau*avr_x(rh)*(avr_x(ux)*drv_x(ux) + drv_x(G) - avr_x(mu)*drv_x(cn))
             + tau*avr_x(rh*avrSt_y(avr_y(uy)*drv_y(ux)));

        my =   tau*avr_y(rh)*(avr_y(uy)*drv_y(uy) + drv_y(G) - avr_y(mu)*drv_y(cn))
             + tau*avr_y(rh*avrSt_x(avr_x(ux)*drv_x(uy)));

        mx_y =   tau*avr_y(rh*avrSt_x(avr_x(ux)*drv_x(ux) + drv_x(G) - avr_x(mu)*drv_x(cn)))
               + tau*avr_y(rh)*avr_y(uy)*drv_y(ux);

        my_x =   tau*avr_x(rh*avrSt_y(avr_y(uy)*drv_y(uy) + drv_y(G) - avr_y(mu)*drv_y(cn)))
               + tau*avr_x(rh)*avr_x(ux)*drv_x(uy);

        Jx = avr_x(rh)*avr_x(ux) - mx; 
        Jy = avr_y(rh)*avr_y(uy) - my;

        P_NS_xx = eta*drv_x(ux) - (2.0/3.0)*eta*avrSt_y(avr_x(drv_y(uy)));
        P_NS_yy = eta*drv_y(uy) - (2.0/3.0)*eta*avrSt_x(avr_y(drv_x(ux)));
        P_NS_xy = eta*drv_x(uy) + eta*avrSt_y(avr_x(drv_y(ux)));
        P_NS_yx = eta*drv_y(ux) + eta*avrSt_x(avr_y(drv_x(uy)));

        P_tau_xx = avr_x(ux)*mx;
        P_tau_xy = avr_x(ux)*my_x;
        P_tau_yy = avr_y(uy)*my;
        P_tau_yx = avr_y(uy)*mx_y;
        
        std::cout << "iter " << iter << std::endl;


        rh_new    = rh - dt*(drvSt_x(Jx) + drvSt_y(Jy));

        rh_ux_new = rh*ux - dt*(
                                 drvSt_x(Jx*avr_x(ux)) + drvSt_y(Jy*avr_y(ux))
                               + avrSt_x(avr_x(rh)*drv_x(G))
                               - drvSt_x(P_NS_xx + P_tau_xx) - drvSt_y(P_NS_yx + P_tau_yx)
                               - avrSt_x(avr_x(rh)*avr_x(mu)*drv_x(cn))
                               );

        rh_uy_new = rh*uy - dt*(
                                 drvSt_x(Jx*avr_x(uy)) + drvSt_y(Jy*avr_y(uy))
                               + avrSt_y(avr_y(rh)*drv_y(G))
                               - drvSt_x(P_NS_xy + P_tau_xy) - drvSt_y(P_NS_yy + P_tau_yy)
                               - avrSt_y(avr_y(rh)*avr_y(mu)*drv_y(cn))
                               );

        rh_cn_new = rh*cn - dt*(
                                 drvSt_x(Jx*avr_x(cn)) + drvSt_y(Jy*avr_y(cn))
                               - drvSt_x(M*drv_x(mu)) -  drvSt_y(M*drv_y(mu))
                               );

        totEnrg = rh*E_lmd + rh*Psi0 + 0.5*rh*(ux*ux + uy*uy);

        std::cout << "Total system energy "  
                  << std::accumulate(totEnrg.begin(), totEnrg.end(), 0.0)
                  << std::endl;

        rh = rh_new;
        cn = rh_cn_new/rh_new;
        ux = rh_ux_new/rh_new;
        uy = rh_uy_new/rh_new;

        if (iter%100 == 0)
            writeVtk(iter, cn, rh, h, CNTR_N);
    }

	return 0;
}
