#ifndef IO_H
#define IO_H

#include <mpi.h>
#include <string>

namespace fs = std::filesystem;

void parse_and_check_dirs_and_get_bin_files(
    int argc, char* argv[],
    std::string& lightcone_dir,
    std::string& halo_dir,
    std::vector<fs::path>& lightcone_bin_files,
    std::vector<fs::path>& halo_bin_files,
    MPI_Comm comm = MPI_COMM_WORLD);

double get_redshift(const fs::path& halo_file);

template <typename Particle>
void get_particle_count_for_redshift(const std::vector<Particle>& particles,
                                const fs::path& lightcone_file,
                                MPI_Comm comm = MPI_COMM_WORLD)
{
    int rank;
    MPI_Comm_rank(comm, &rank);

    // Per-rank count
    long long local_count =
        static_cast<long long>(particles.size());

    // Global count
    long long global_count = 0;
    MPI_Allreduce(&local_count, &global_count,
                  1, MPI_LONG_LONG, MPI_SUM, comm);

    if (rank == 0) {
        // Extract redshift from filename: e.g. "reeber_halos_0000395.bin"
        const std::string fname = lightcone_file.filename().string();
        const size_t uscore = fname.find_last_of('_');
        const size_t dot    = fname.find_last_of('.');

        if (uscore == std::string::npos || dot == std::string::npos || dot <= uscore) {
            std::cerr << "Warning: could not parse redshift from filename "
                      << fname << "\n";
            return;
        }

        const std::string num_str = fname.substr(uscore + 1,
                                                 dot - uscore - 1);

        const double redshift = std::stod(num_str) / 100.0;

        std::cout << std::fixed << std::setprecision(2)
                  << redshift << " "
                  << global_count << "\n";

        std::cout << "Redshift " << redshift
                  << " → total halos = " << global_count << std::endl;

        std::cout << "Total number of halos in file "
                  << lightcone_file << " = "
                  << global_count << std::endl;
    }
}



#endif // IO_H

