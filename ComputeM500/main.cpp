#include <mpi.h>
#include <vector>
#include <cmath>
#include <iostream>
#include <fstream>
#include <filesystem>
#include <string>
#include <algorithm>

#include "Particles.H"
#include "IO.H"

namespace fs = std::filesystem;

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    std::string lightcone_dir, halo_dir;
    std::vector<std::filesystem::path> lightcone_bin_files, halo_bin_files;

    parse_and_check_dirs_and_get_bin_files(
        argc, argv,
        lightcone_dir, halo_dir,
        lightcone_bin_files, halo_bin_files
    );


    // In units of Mpc
    double xcen = 3850.0;
    double ycen = 3850.0;
    double zcen = 3850.0;

    const double G = 4.30091e-9; // Mpc * (km/s)^2 / Msun
    const double H0 = 70.0;      // km/s/Mpc
    const double Omega_m = 0.3;
    const double Omega_Lambda = 0.7;

    // Loop over halo files
    //for (size_t i = 2; i < halo_bin_files.size() - 2; ++i) {
    //for (size_t i = 2; i < halo_bin_files.size() - 2; ++i) {
    for (size_t i = 20; i < 21; ++i) {
        const auto& halo_file = halo_bin_files[i];

        // Start timer
        double t_start = MPI_Wtime();
        if (rank == 0) std::cout << "Processing halo file: " << halo_file << std::endl;

        // --- Read halo file ---
        std::vector<HaloParticle> halo_particles;
        readHaloBinary(halo_file.string(), halo_particles,
                                           xcen, ycen, zcen);

        if(rank == 0) {
            std::cout << "The number of halos is " << halo_particles.size() << std::endl;
        }
        // --- Read corresponding + 2 previous + 2 next lightcone files ---
        std::vector<LightConeParticle> combined_lightcone_particles;
        size_t start_idx = (i >= 1) ? i - 1 : 0;
        size_t end_idx   = std::min(i + 2, lightcone_bin_files.size() - 1);

        for (size_t j = start_idx; j <= end_idx; ++j) {
            const auto& lc_file = lightcone_bin_files[j];

            std::vector<LightConeParticle> lc_particles;
            readLightConeBinary(lc_file.string(), lc_particles,
                                                    xcen, ycen, zcen);

            // Append to the combined vector
            combined_lightcone_particles.insert(combined_lightcone_particles.end(),
                                            lc_particles.begin(), lc_particles.end());

            // --- Count particles ---
             get_particle_count_for_redshift(combined_lightcone_particles, lightcone_bin_files[i]);
        }

        
        double zval = get_redshift(halo_file);
        double H_z = H0 * sqrt( Omega_m * pow(1+zval,3) + Omega_Lambda ); // km/s/Mpc

        if (rank == 0) {
            std::cout << "Processing redshift = " << zval << std::endl;
        }

        long int counter=0;
        for (const auto& halo : halo_particles) {

            if(halo.mass < 1e13)continue;

            double R500 = 0.0;
            double Rmax = 5.0;   // choose physically (e.g. 5 Mpc)

            double M500 = ComputeM500(halo,
                                      combined_lightcone_particles,  // local only!
                                      H_z,
                                      G,
                                      Rmax,
                                      MPI_COMM_WORLD,
                                      R500);

            counter++;
            if (rank == 0) {
                //std::cout << "Halo M500 = " << M500
                  //    << "  R500 = " << R500 << " " << counter << std::endl;
            }
            
        }

        // End timer
        double t_end = MPI_Wtime();
        double elapsed = t_end - t_start;

        if (rank == 0) {
            std::cout << "Time taken to process halo file " << halo_file
                      << " with surrounding lightcones = " << elapsed << " seconds.\n";
        }
    }

    MPI_Finalize();
    return 0;
}
