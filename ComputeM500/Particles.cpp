#include <mpi.h>
#include <vector>
#include <cmath>
#include <iostream>
#include <fstream>
#include <filesystem>
#include <string>
#include <algorithm>

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
    extractParticles(local_particles, solid_angle_particles, xcen, ycen, zcen);
}
