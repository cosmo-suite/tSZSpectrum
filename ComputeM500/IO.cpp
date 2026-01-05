#include <mpi.h>
#include <filesystem>
#include <string>
#include <iostream>
#include <vector>
#include <algorithm>

namespace fs = std::filesystem;

// -------------------------------------------------------------------
// Parse command-line arguments for --lightcone_dir= and --halo_dir=
// Aborts via MPI if missing or invalid
// -------------------------------------------------------------------

void parse_and_check_dirs_and_get_bin_files(
    int argc, char* argv[],
    std::string& lightcone_dir,
    std::string& halo_dir,
    std::vector<fs::path>& lightcone_bin_files,
    std::vector<fs::path>& halo_bin_files,
    MPI_Comm comm)
{
    int rank;
    MPI_Comm_rank(comm, &rank);

    // Parse arguments
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        const std::string lc_prefix   = "--lightcone_dir=";
        const std::string halos_prefix = "--halo_dir=";

        if (arg.rfind(lc_prefix, 0) == 0) {
            lightcone_dir = arg.substr(lc_prefix.size());
        } else if (arg.rfind(halos_prefix, 0) == 0) {
            halo_dir = arg.substr(halos_prefix.size());
        }
    }

    // Check required arguments
    if (lightcone_dir.empty()) {
        if (rank == 0) std::cerr << "Error: --lightcone_dir argument is required.\n";
        MPI_Abort(comm, 1);
    }
    if (halo_dir.empty()) {
        if (rank == 0) std::cerr << "Error: --halo_dir argument is required.\n";
        MPI_Abort(comm, 1);
    }

    // Check directories exist
    if (!fs::exists(lightcone_dir)) {
        if (rank == 0) std::cerr << "Error: Directory does not exist: " << lightcone_dir << "\n";
        MPI_Abort(comm, 1);
    }
    if (!fs::exists(halo_dir)) {
        if (rank == 0) std::cerr << "Error: Directory does not exist: " << halo_dir << "\n";
        MPI_Abort(comm, 1);
    }

    if (rank == 0) {
        std::cout << "Using lightcone_dir: " << lightcone_dir << "\n";
        std::cout << "Using halo_dir: " << halo_dir << "\n";
    }

    // Collect sorted .bin files
    lightcone_bin_files.clear();
    for (auto& p : fs::directory_iterator(lightcone_dir)) {
        if (p.path().extension() == ".bin") lightcone_bin_files.push_back(p.path());
    }
    std::sort(lightcone_bin_files.begin(), lightcone_bin_files.end());

    halo_bin_files.clear();
    for (auto& p : fs::directory_iterator(halo_dir)) {
        if (p.path().extension() == ".bin") halo_bin_files.push_back(p.path());
    }
    std::sort(halo_bin_files.begin(), halo_bin_files.end());
}

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

