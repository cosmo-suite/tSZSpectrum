#include <mpi.h>
#include <vector>
#include <cmath>
#include <iostream>
#include <fstream>
#include <filesystem>
#include <string>
#include <algorithm>
#include <stdexcept>
#include <iomanip>


#include "Particles.H"

void readLightConeBinary(const std::string& filename,
                                    std::vector<LightConeParticle>& solid_angle_particles,
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
    extractParticles(local_particles, solid_angle_particles, xcen, ycen, zcen);
}

void readHaloBinary(const std::string& filename,
                    std::vector<HaloParticle>& solid_angle_particles,
                    double xcen, double ycen, double zcen)
{
    // Open file
    std::ifstream ifs(filename, std::ios::binary | std::ios::ate);
    if (!ifs) {
        throw std::runtime_error("Failed to open halo file: " + filename);
    }

    // Get file size
    std::streamsize file_size = ifs.tellg();
    ifs.seekg(0, std::ios::beg);

    constexpr std::size_t floats_per_halo = 5;
    std::size_t total_floats =
        static_cast<std::size_t>(file_size) / sizeof(float);

    if (total_floats % floats_per_halo != 0) {
        throw std::runtime_error("Halo file size not divisible by 5 floats: " + filename);
    }

    std::size_t total_halos = total_floats / floats_per_halo;

    // Read entire file
    std::vector<float> data(total_floats);
    if (!ifs.read(reinterpret_cast<char*>(data.data()),
                  total_floats * sizeof(float))) {
        throw std::runtime_error("Failed to read halo data from: " + filename);
    }

    // Convert to HaloParticle
    std::vector<HaloParticle> local_particles;
    local_particles.reserve(total_halos);

    for (std::size_t i = 0; i < total_halos; ++i) {
        float x      = data[5*i + 0];
        float y      = data[5*i + 1];
        float z      = data[5*i + 2];
        float mass   = data[5*i + 3];
        float ncells = data[5*i + 4];

        SwapEnd(x); SwapEnd(y); SwapEnd(z);
        SwapEnd(mass); SwapEnd(ncells);

        HaloParticle h;
        h.x = x;
        h.y = y;
        h.z = z;
        h.mass = mass;
        h.ncells = static_cast<int>(ncells);

        local_particles.push_back(h);
    }

    // Extract halos in solid angle
    extractParticles(local_particles,
                     solid_angle_particles,
                     xcen, ycen, zcen);
}

double ComputeM500(const HaloParticle& halo,
                   const std::vector<LightConeParticle>& local_lightcone_particles,
                   double H_z,
                   double G,
                   double Rmax,
                   MPI_Comm comm,
                   double& R500)
{
    constexpr int NBINS = 1024;
    const double dr = Rmax / NBINS;

    // ---------------------------------------------------
    // 1. Critical density at halo redshift
    // ---------------------------------------------------
    const double rho_crit_z =
        3.0 * H_z * H_z / (8.0 * M_PI * G);

    // ---------------------------------------------------
    // 2. Local radial mass histogram
    // ---------------------------------------------------
    std::vector<double> local_mass(NBINS, 0.0);

    for (const auto& p : local_lightcone_particles) {
        double r = distance(p, halo);
        if (r < Rmax) {
            int bin = static_cast<int>(r / dr);
            if (bin >= 0 && bin < NBINS) {
                local_mass[bin] += 2.76375e10;
            }
        }
    }

    // ---------------------------------------------------
    // 3. Global reduction
    // ---------------------------------------------------
    std::vector<double> global_mass(NBINS, 0.0);

    MPI_Allreduce(local_mass.data(),
                  global_mass.data(),
                  NBINS,
                  MPI_DOUBLE,
                  MPI_SUM,
                  comm);

     int rank;
     MPI_Comm_rank(comm, &rank);

    // ---------------------------------------------------
    // 4. Cumulative mass & find R500
    // ---------------------------------------------------
        
        double cumulative_mass = 0.0;
        double M500 = 0.0;
        R500 = 0.0;
        
        int binval = 0;

        for (int i = 0; i < NBINS; ++i) {
            cumulative_mass += global_mass[i];
              // Skip bins with too few particles

            double r = (i + 1) * dr;

            double mean_density = cumulative_mass / ((4.0/3.0) * M_PI * r*r*r);
            if (cumulative_mass > halo.mass and mean_density <= 500.0 * rho_crit_z) {
                R500 = r;
                M500 = cumulative_mass;
                if(rank == 0){
                    std::cout << std::scientific << std::setprecision(3)
                    << "Value of M500, halo mass, ratio, R500 is "
                    << cumulative_mass << " " << halo.mass << " " << cumulative_mass/halo.mass << " " 
                    << " " << r << std::endl;
                }
                binval = i;
                break;
            }
        }

        if(rank == 0) {
           
            if(binval==0) {
                std::cout  << "Converged in first bin itself" << std::endl; 
                exit(0);
            }
        }

        if(rank == 0) {
            if(M500 == 0.0) { 
                std::cout << "Could not converge R500 for this halo " << std::endl;
                exit(0);
            }
        }

    return M500;
}
