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

    double xcen = 3850.0;
    double ycen = 3850.0;
    double zcen = 3850.0;

    // Loop over halo files
    for (size_t i = 2; i < halo_bin_files.size() - 2; ++i) {
        const auto& halo_file = halo_bin_files[i];

        // Start timer
        double t_start = MPI_Wtime();
        if (rank == 0) std::cout << "Processing halo file: " << halo_file << std::endl;

        // --- Read halo file ---
        std::vector<HaloParticle> halo_particles;
        readHaloBinary(halo_file.string(), halo_particles,
                                           xcen, ycen, zcen);

        // --- Read corresponding + 2 previous + 2 next lightcone files ---
        std::vector<LightConeParticle> combined_lightcone_particles;
        size_t start_idx = (i >= 2) ? i - 2 : 0;
        size_t end_idx   = std::min(i + 2, lightcone_bin_files.size() - 1);

        for (size_t j = start_idx; j <= end_idx; ++j) {
            const auto& lc_file = lightcone_bin_files[j];

            std::vector<LightConeParticle> lc_particles;
            readLightConeBinary(lc_file.string(), lc_particles,
                                                    xcen, ycen, zcen);

            // Append to the combined vector
            combined_lightcone_particles.insert(combined_lightcone_particles.end(),
                                            lc_particles.begin(), lc_particles.end());
        }

        // --- Count particles ---
        get_particle_count_for_redshift(combined_lightcone_particles, lightcone_bin_files[i]);
        get_particle_count_for_redshift(halo_particles, halo_file);

        // End timer
        double t_end = MPI_Wtime();
        double elapsed = t_end - t_start;

        if (rank == 0) {
            std::cout << "Time taken to process halo file " << halo_file
                      << " with surrounding lightcones = " << elapsed << " seconds.\n";
        }
    }



    // Loop over files and accumulate selected particles
    for (size_t i = 2; i < 10; i += 1) {
        const auto& lightcone_file = lightcone_bin_files[i];
        const auto& halo_file = halo_bin_files[i];

        // Start timer
        double t_start = MPI_Wtime();
        if (rank == 0) std::cout << "Processing files: " << lightcone_file << " and " << halo_file << std::endl;

        std::vector<LightConeParticle> lightcone_particles;
        readLightConeBinary(lightcone_file.string(), lightcone_particles,
                                       xcen, ycen, zcen);

        std::vector<HaloParticle> halo_particles;
        readHaloBinary(halo_file.string(), halo_particles,
                                       xcen, ycen, zcen);

        get_particle_count_for_redshift(lightcone_particles, lightcone_file);
        get_particle_count_for_redshift(halo_particles, halo_file);

        // End timer
        double t_end = MPI_Wtime();
        double elapsed = t_end - t_start;

        if (rank == 0) {
            std::cout << "Time taken to process file: " << lightcone_file 
                  << " = " << elapsed << " seconds.\n";
        } 
    }
    MPI_Finalize();
    return 0;
}
