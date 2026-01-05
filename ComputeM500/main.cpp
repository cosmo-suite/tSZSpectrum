#include <mpi.h>
#include <vector>
#include <cmath>
#include <iostream>
#include <fstream>
#include <filesystem>
#include <string>
#include <algorithm>

namespace fs = std::filesystem;

struct LightConeParticle {
    double x, y, z, vx, vy, vz;
};

struct HaloParticle {
    double x, y, z, mass;
    int ncells;
};

void SwapEnd(float& val) {
    char* bytes = reinterpret_cast<char*>(&val);
    std::swap(bytes[0], bytes[3]);
    std::swap(bytes[1], bytes[2]);
}

template <typename Particle>
void extractParticlesInSolidAngle(const std::vector<Particle>& all_particles,
                                  std::vector<Particle>& selected_particles,
                                  double xcen, double ycen, double zcen) {
    for (const auto& p : all_particles) {
        double dx = p.x - xcen;
        double dy = p.y - ycen;
        double dz = p.z - zcen;
        double r = std::sqrt(dx*dx + dy*dy + dz*dz);
        if (r < 1e-12) continue;
        selected_particles.push_back(p);
    }
}

void readLightConeBinaryAndExtractSolidAngle(const std::string& filename,
                                    std::vector<LightConeParticle>& solid_angle_particles,
                                    double xcen, double ycen, double zcen,
                                    double solid_angle_rad=2.0*M_PI) {

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Open MPI file
    MPI_File mpi_file;
    MPI_File_open(MPI_COMM_WORLD, filename.c_str(), MPI_MODE_RDONLY, MPI_INFO_NULL, &mpi_file);

    // Get file size
    MPI_Offset file_size;
    MPI_File_get_size(mpi_file, &file_size);

    MPI_Offset header_size = 0; // adjust if header exists
    long int total_floats = (file_size - header_size) / sizeof(float);
    long int total_particles = total_floats / 6;

    // Divide particles among ranks
    long int base_count = total_particles / size;
    long int remainder = total_particles % size;
    long int local_count = (rank < remainder) ? base_count + 1 : base_count;
    long int offset = rank * base_count + std::min(rank, (int)remainder);

    MPI_Offset byte_offset = header_size + offset * 6 * sizeof(float);

    // Read local data
    std::vector<float> data(6 * local_count);
    MPI_File_read_at_all(mpi_file, byte_offset, data.data(), data.size(), MPI_FLOAT, MPI_STATUS_IGNORE);
    MPI_File_close(&mpi_file);

    // Convert to LightConeParticle
    std::vector<LightConeParticle> local_particles(local_count);
    for (long int i = 0; i < local_count; ++i) {
        float x = data[6*i], y = data[6*i+1], z = data[6*i+2];
        float vx = data[6*i+3], vy = data[6*i+4], vz = data[6*i+5];

        SwapEnd(x); SwapEnd(y); SwapEnd(z);
        SwapEnd(vx); SwapEnd(vy); SwapEnd(vz);

        local_particles[i] = {x, y, z, vx, vy, vz};
    }

    // Extract particles in solid angle
    extractParticlesInSolidAngle(local_particles, solid_angle_particles, xcen, ycen, zcen);
}

void readHaloBinaryAndExtractSolidAngle(const std::string& filename,
                                    std::vector<HaloParticle>& solid_angle_particles,
                                    double xcen, double ycen, double zcen)
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Open MPI file
    MPI_File mpi_file;
    MPI_File_open(MPI_COMM_WORLD, filename.c_str(), MPI_MODE_RDONLY, MPI_INFO_NULL, &mpi_file);

    // Get file size
    MPI_Offset file_size;
    MPI_File_get_size(mpi_file, &file_size);

    MPI_Offset header_size = 0; // adjust if header exists
    long int total_floats = (file_size - header_size) / sizeof(float);
    long int total_particles = total_floats / 5;

    // Divide particles among ranks
    long int base_count = total_particles / size;
    long int remainder = total_particles % size;
    long int local_count = (rank < remainder) ? base_count + 1 : base_count;
    long int offset = rank * base_count + std::min(rank, (int)remainder);

    MPI_Offset byte_offset = header_size + offset * 5 * sizeof(float);

    // Read local data
    std::vector<float> data(5 * local_count);
    MPI_File_read_at_all(mpi_file, byte_offset, data.data(), data.size(), MPI_FLOAT, MPI_STATUS_IGNORE);
    MPI_File_close(&mpi_file);

    // Convert to HaloParticle
    std::vector<HaloParticle> local_particles(local_count);
    for (long int i = 0; i < local_count; ++i) {
        float x = data[5*i], y = data[5*i+1], z = data[5*i+2];
        float mass = data[5*i+3], ncells = data[5*i+4];

        SwapEnd(x); SwapEnd(y); SwapEnd(z);
        SwapEnd(mass); SwapEnd(ncells);

        local_particles[i].x = x;
        local_particles[i].y = y;
        local_particles[i].z = z;
        local_particles[i].mass = mass;
        local_particles[i].ncells = static_cast<int>(ncells);
    }

    // Extract particles in solid angle
    extractParticlesInSolidAngle(local_particles, solid_angle_particles, xcen, ycen, zcen);
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


int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    std::string lightcone_dir, halo_dir;

    // Parse command-line arguments
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        const std::string lc_prefix = "--lightcone_dir=";
        const std::string halos_prefix = "--halo_dir=";

        if (arg.rfind(lc_prefix, 0) == 0) {
            lightcone_dir = arg.substr(lc_prefix.size());
        }
        else if (arg.rfind(halos_prefix, 0) == 0) {
            halo_dir = arg.substr(halos_prefix.size());
        }
    }

    // Check if required arguments were provided
    if (lightcone_dir.empty()) {
        if (rank == 0) {
            std::cerr << "Error: --lightcone_dir argument is required.\n";
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    if (halo_dir.empty()) {
        if (rank == 0){
            std::cerr << "Error: --halo_dir argument is required.\n";
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Check if directories exist
    if (!std::filesystem::exists(lightcone_dir)) {
        if (rank == 0) {
            std::cerr << "Error: Directory does not exist: " << lightcone_dir << "\n";
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    if (!std::filesystem::exists(halo_dir)) {
        if (rank == 0) {
            std::cerr << "Error: Directory does not exist: " << halo_dir << "\n";
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    if (rank == 0) {
        std::cout << "Using lightcone_dir: " << lightcone_dir << std::endl;
        std::cout << "Using halo_dir: " << halo_dir << std::endl;
    }
   
    double xcen = 3850.0;
    double ycen = 3850.0;
    double zcen = 3850.0;

    // Get sorted list of all *.bin files
    std::vector<fs::path> lightcone_bin_files, halo_bin_files;
    for (auto& p : fs::directory_iterator(lightcone_dir)) {
        if (p.path().extension() == ".bin") lightcone_bin_files.push_back(p.path());
    }
    std::sort(lightcone_bin_files.begin(), lightcone_bin_files.end());

    for (auto& p : fs::directory_iterator(halo_dir)) {
        if (p.path().extension() == ".bin") halo_bin_files.push_back(p.path());
    }
    std::sort(halo_bin_files.begin(), halo_bin_files.end());

    // Loop over files and accumulate selected particles
    for (size_t i = 2; i < 10; i += 1) {
        const auto& lightcone_file = lightcone_bin_files[i];
        const auto& halo_file = halo_bin_files[i];

        // Start timer
        double t_start = MPI_Wtime();
        if (rank == 0) std::cout << "Processing files: " << lightcone_file << " and " << halo_file << std::endl;

        std::vector<LightConeParticle> lightcone_particles;
        readLightConeBinaryAndExtractSolidAngle(lightcone_file.string(), lightcone_particles,
                                       xcen, ycen, zcen);

        std::vector<HaloParticle> halo_particles;
        readHaloBinaryAndExtractSolidAngle(halo_file.string(), halo_particles,
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
