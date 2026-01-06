#ifndef PARTICLES_H
#define PARTICLES_H

#include <string>     // std::string
#include <vector>     // std::vector
#include <cmath>      // M_PI

struct LightConeParticle {
    double x, y, z, vx, vy, vz;
};

struct HaloParticle {
    double x, y, z, mass;
    int ncells;
};

inline void SwapEnd(float& val) {
    char* bytes = reinterpret_cast<char*>(&val);
    std::swap(bytes[0], bytes[3]);
    std::swap(bytes[1], bytes[2]);
}

// -------------------------------------------------------------------
// Function signatures
// -------------------------------------------------------------------

void readLightConeBinary(
    const std::string& filename,
    std::vector<LightConeParticle>& solid_angle_particles,
    double xcen, double ycen, double zcen);

void readHaloBinary(
    const std::string& filename,
    std::vector<HaloParticle>& solid_angle_particles,
    double xcen, double ycen, double zcen);

template <typename Particle>
void extractParticles(const std::vector<Particle>& all_particles,
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

#endif // PARTICLES_H
