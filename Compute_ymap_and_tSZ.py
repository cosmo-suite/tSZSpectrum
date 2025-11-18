import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import os
import sys
from dataclasses import dataclass
from mpi4py import MPI
from astropy.cosmology import Planck18 as cosmo
import astropy.units as u
from astropy.constants import sigma_T, m_e, c
from astropy.constants import G, M_sun
import matplotlib.ticker as mticker  # <-- add this
from matplotlib.lines import Line2D

sigma_T = sigma_T.value   # convert from Quantity to float (in SI units)
m_e = m_e.value
c = c.value

# ----------------------------
# Parameters
# ----------------------------
Nside = 1024
observer_pos = np.array([3850.0, 3850.0, 3850.0])

# ----------------------------
# Data structure
# ----------------------------
@dataclass
class Halo:
    x: float
    y: float
    z: float
    mass: float
    redshift: float

def RDelta(M, z, Delta=82):
    rho_c = cosmo.critical_density(z).to(u.Msun/u.Mpc**3).value
    Rdelta = (3 * M / (4 * np.pi * Delta * rho_c))**(1/3)
    return Rdelta  # in Mpc

#def R500(M500, z):
#    rho_c = cosmo.critical_density(z).to(u.Msun/u.Mpc**3).value
#    R500 = (3 * M500 / (4 * np.pi * 500 * rho_c))**(1/3)
#    return R500  # in Mpc

def pressure_gnfw(r, M500, z, h70=1.0,
                  P0=None, c500=1.177, alpha=1.051, beta=5.4905, gamma=0.3081,
                  alpha_P=0.12, cosmo=None):
    """
    Generalized NFW / Universal Pressure Profile (Arnaud et al. 2010)

    Parameters
    ----------
    r : array_like
        Radius [Mpc].
    M500 : float
        Cluster mass within R500 [Msun].
    z : float
        Redshift.
    h70 : float
        Hubble parameter normalized to 70 km/s/Mpc.
    P0, c500, alpha, beta, gamma, alpha_P : floats
        GNFW and mass-scaling parameters.
    cosmo : astropy.cosmology instance
        Cosmology object (e.g. Planck18).

    Returns
    -------
    P : ndarray
        Electron pressure [keV cm^-3].
    """
    if cosmo is None:
        raise ValueError("Must supply a cosmology (e.g., cosmo=Planck18).")

    if P0 is None:
        P0 = 8.403 * h70**(-1.5)

    # Cosmological factor
    E_z = cosmo.efunc(z)

    #print("E_z value is " , E_z);

    # Compute R500 from M500
    R500 = RDelta(M500, z) 

    # Dimensionless radius
    x = np.asarray(r) / R500

    # α'_P(x)
    alpha_P_prime = 0.1 - (alpha_P + 0.1) * (x / 0.5)**3 / (1 + (x / 0.5)**3)

    # P500 normalization
    P500 = (1.65e-3 * E_z**(8/3) *
            (M500 / (3e14 / h70))**(2/3 + alpha_P + alpha_P_prime) *
            h70**2)  # keV cm^-3

    #print("R value is ", R500)

    # GNFW shape
    p_x = P0 / ((c500 * x)**gamma * (1 + (c500 * x)**alpha)**((beta - gamma) / alpha))

    #print("px value is ", p_x)
    x = r / R500
    #print("x range (min, max) =", x.min(), x.max())

    return P500 * p_x

def add_halo_to_map(y_map, halo, Nside, observer_pos, dirs=None, box_size_mpc=7700.0):
    """
    Add the Compton-y contribution from a single halo to the given Healpix map.

    Parameters
    ----------
    y_map : np.ndarray
        Healpix map to be updated in-place.
    halo : Halo
        Halo object with (x, y, z, mass, redshift).
    Nside : int
        Healpix resolution.
    observer_pos : array-like
        3D position of the observer in Mpc.
    dirs : np.ndarray, optional
        Precomputed Healpix pixel direction unit vectors, shape (npix, 3).
        If None, it will be computed inside this function (slower).
    box_size_mpc : float
        Simulation box size (used for reference).

    Notes
    -----
    This computes the line-of-sight Compton-y signal for all Healpix pixels
    within the angular extent of the halo (up to 5×R500).
    """
    from scipy.constants import m_e, c
    sigma_T = 6.6524587158e-29  # Thomson cross section [m^2]

    if dirs is None:
        npix = hp.nside2npix(Nside)
        dirs = np.array(hp.pix2vec(Nside, np.arange(npix))).T  # shape (npix, 3)

    M = halo.mass
    z = halo.redshift
    r_halo = np.array([halo.x, halo.y, halo.z])
    R_500 = RDelta(M, z)

    # Maximum extent for integration (~5×R500)
    Rmax = 5.0 * R_500
    D_A = cosmo.angular_diameter_distance(z).to(u.Mpc).value


    # Angular size of halo on sky
    theta_max = np.arcsin(Rmax / D_A)
    #print("R500, Rmax and D_A theta_max are ", R_500, Rmax, D_A, theta_max)

    # Direction of halo from observer
    rel = r_halo - observer_pos
    r_norm = np.linalg.norm(rel)
    if r_norm == 0:
        return
    n_hat_halo = rel / r_norm

    # Find Healpix pixels within theta_max of halo center
    dotprod = np.dot(dirs, n_hat_halo)
    mask = dotprod > np.cos(theta_max)
    pix_indices = np.where(mask)[0]
    if len(pix_indices) == 0:
        return

    # Integrate y(b) along LOS for pixels in halo footprint
    for pix in pix_indices:
        n_hat = dirs[pix]
        b = D_A * np.arccos(np.clip(np.dot(n_hat, n_hat_halo), -1, 1))  # impact parameter (Mpc)
        if b > Rmax:
            continue

        #if(b > 0.2*Rmax):
            #print("b and Rmax are ", b, Rmax)

        l_max = np.sqrt(max(0.0, Rmax**2 - b**2))
        l_vals = np.linspace(-l_max, l_max, 10000)
        r_vals = np.sqrt(b**2 + l_vals**2)
        P_vals = pressure_gnfw(r_vals, M, z, cosmo=cosmo)  
 
        max_P = np.max(P_vals)

        #if(max_P > 0.05):
            #plt.figure(figsize=(7, 5))
            #plt.loglog(r_vals*1e3, P_vals, color='darkorange', lw=2)

            #plt.xlabel("Radius r [kpc]", fontsize=12)
            #plt.ylabel("Pressure $P_e$ [keV / cm$^3$]", fontsize=12)
            #plt.title(f"GNFW Pressure Profile (M = {M:.2e} M$_\odot$, z = {z:.2f})", fontsize=13)

            #plt.ylim([1e-3,1e-1])
            #plt.xlim([50,1e3])
            #plt.grid(True, which='both', ls='--', alpha=0.5)
            #plt.savefig("pressure_profile_loglog.png", dpi=300)

        #if max_P > 0.05:
        #    print(f"Max P = {max_P:.6e}, M = {M}, z = {z}")
        #    sys.exit()
        integral = np.trapz(P_vals, l_vals)  # ∫ P_e dl

        y_val = (sigma_T / (m_e * c**2)) * integral * 1.602176634e-10 * 3.085677581e22
        y_map[pix] += y_val

# ----------------------------
# Functions
# ----------------------------
def read_binary_points(filename):
    """Read binary halo file: x, y, z, mass, n_cells"""
    with open(filename, 'rb') as f:
        data = np.fromfile(f, dtype='>f4')  # big-endian float32
        if len(data) % 5 != 0:
            raise ValueError(f"Data size not multiple of 5 in file {filename}")
        points = data.reshape((-1, 5))
    return points

def extract_redshift_from_filename(fname):
    """Extract redshift from filename like 'reeber_halos_0000169.bin' -> 1.69"""
    base = os.path.basename(fname)
    num_str = base.split("_")[-1].split(".")[0]  # '0000169'
    # 0000169 -> 1.69
    z_str = f"{int(num_str)/100:.2f}"
    return float(z_str)

def halos_to_healpix(points, nside, observer_pos):
    """Project halo positions onto Healpix map, weighted by mass"""
    npix = hp.nside2npix(nside)
    sky_map = np.zeros(npix, dtype=np.float64)

    rel_pos = points[:, :3] - observer_pos
    masses = points[:, 3]

    x, y, z = rel_pos[:,0], rel_pos[:,1], rel_pos[:,2]
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(z / r)
    phi = np.arctan2(y, x)
    phi[phi < 0] += 2*np.pi

    pix = hp.ang2pix(nside, theta, phi)
    np.add.at(sky_map, pix, masses)  # faster than manual loop

    return sky_map


def compute_y_map_mpi(halos, Nside, observer_pos):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    npix = hp.nside2npix(Nside)
    y_local = np.zeros(npix, dtype=np.float64)

    # Split halos among ranks
    halos_local = halos[rank::size]

    for i, halo in enumerate(halos_local):
        #if(rank == 0 and (i+1)%100 == 0):
        print(f"[Rank {rank}] Processing halo {i+1}/{len(halos_local)}: Mass={halo.mass:.2e}, z={halo.redshift:.2f}")
        add_halo_to_map(y_local, halo, Nside, observer_pos)

    # Sum contributions across ranks
    y_global = np.zeros_like(y_local)
    comm.Allreduce(y_local, y_global, op=MPI.SUM)

    return y_global

# ----------------------------
# Main
# ----------------------------
def main():
    if len(sys.argv) != 2:
        print(f"Usage: python {os.path.basename(sys.argv[0])} <halo_folder>")
        sys.exit(1)

    halo_folder = sys.argv[1]
    if not os.path.isdir(halo_folder):
        print(f"Error: {halo_folder} is not a folder.")
        sys.exit(1)

    total_map = np.zeros(hp.nside2npix(Nside), dtype=np.float64)
    all_files = sorted(os.listdir(halo_folder))

    #print(all_files)

    halos_all = []  # list of Halo objects

    for i in range(0, len(all_files), 1):
        fname = all_files[i]
        if not fname.endswith(".bin"):
            continue
        path = os.path.join(halo_folder, fname)
        points = read_binary_points(path)
        z = extract_redshift_from_filename(fname)

        # store halos
        for p in points:
            halos_all.append(Halo(x=p[0], y=p[1], z=p[2], mass=p[3], redshift=z))

    num_halos = len(halos_all)
    print("Total number of halos:", num_halos)    
    #sys.exit(0)

        #sky_map = halos_to_healpix(points, Nside, observer_pos)
        #total_map += sky_map


    #y_map = compute_y_map_mpi(halos_all, Nside, observer_pos)
    # Select only halos with mass > 1e15
    massive_halos = [halo for halo in halos_all if halo.mass > 1e13]
    print(f"Number of massive halos: {len(massive_halos)}")
    y_map = compute_y_map_mpi(massive_halos, Nside, observer_pos)

    # y_map: HEALPix map in RING ordering
    # Planck 143 GHz FWHM in arcminutes
    fwhm_arcmin = 7.3
    fwhm_rad = np.deg2rad(fwhm_arcmin / 60.0)

    # Smooth the map
    y_map_smoothed = y_map;#hp.smoothing(y_map, fwhm=fwhm_rad, verbose=True)
    y_map_smoothed = y_map;#hp.ud_grade(y_map_smoothed, nside_out=2048)  # or match Planck Nside



    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    if rank == 0:

        # Compute percentile clipping values
        vmin, vmax = np.percentile(y_map_smoothed, [1, 99])

        # True min and max
        true_min = y_map_smoothed.min()
        true_max = y_map_smoothed.max()

        print("the min and max values are ", true_min, true_max)

        #fig = plt.figure(figsize=(8,6))

        ## Plot with percentile clipping for better visibility
        #hp.mollview(
        #    y_map_smoothed,
        #    title="Compton-y Map",
        #    unit="Compton-y",
        #    cmap=plt.cm.plasma,
        #    min=vmin,
        #    max=vmax,
        #    cbar=False,   # disable default colorbar
        #    fig=fig.number 
        #)


        # Apply a small floor to avoid log10(0)
        y_safe = np.where(y_map_smoothed > 0, y_map_smoothed, 1e-30)

        # Compute log10
        y_log = np.log10(y_safe)

        # Compute percentiles in log space for clipping
        # Set fixed log10 min/max
        vmin_log = np.log10(1e-7)
        vmax_log = np.log10(4e-4)

        # True min and max for reference
        true_min_log = y_log.min()
        true_max_log = y_log.max()
        print("log10 min and max values:", true_min_log, true_max_log)

        # Plot
        fig = plt.figure(figsize=(8,6))
        hp.mollview(
            y_log,
            title="log10 Compton-y Map",
            unit="log10(Compton-y)",
            cmap=plt.cm.plasma,
            min=vmin_log,
            max=vmax_log,
            cbar=False,
            fig=fig.number
        )


       # Create ScalarMappable with true min/max
        sm = plt.cm.ScalarMappable(cmap=plt.cm.plasma, norm=plt.Normalize(vmin=1e-7, vmax=4e-4))
        sm.set_array([])

        # create a wide horizontal colorbar
        cbar_ax = fig.add_axes([0.15, 0.07, 0.7, 0.03])  # left, bottom, width, height
        cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
        cbar.set_label("Compton-y (true min/max scale)")

        # use scientific notation (1e-4 format)
        cbar.ax.xaxis.set_major_formatter(mticker.ScalarFormatter(useMathText=True))
        cbar.ax.xaxis.get_offset_text().set_visible(False)  # hide offset text if any

        plt.savefig("y_map_mollview.png", dpi=300)

        fig = plt.figure(figsize=(8,6))

              
        hp.orthview(
            y_map_smoothed,
            rot=(0, 0, 0),
            half_sky=False,
            title="Compton-y Map",
            unit="Compton-y",
            cmap=plt.cm.rainbow,
            xsize=800,
            min=vmin,
            max=vmax,
            cbar=False,   # disable default colorbar
            fig=fig.number
        )

        # Create ScalarMappable with true min/max
        sm = plt.cm.ScalarMappable(cmap=plt.cm.rainbow, norm=plt.Normalize(vmin=true_min, vmax=true_max))
        sm.set_array([])

        # create a wide horizontal colorbar
        cbar_ax = fig.add_axes([0.15, 0.07, 0.7, 0.03])  # left, bottom, width, height
        cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
        cbar.set_label("Compton-y (true min/max scale)")

        # use scientific notation (1e-4 format)
        cbar.ax.xaxis.set_major_formatter(mticker.ScalarFormatter(useMathText=True))
        cbar.ax.xaxis.get_offset_text().set_visible(False)  # hide offset text if any

        plt.savefig("y_map_orthographic.png", dpi=300)

        # Compute the spherical harmonic coefficients (alm)
        alm = hp.map2alm(y_map_smoothed)

        lmax = 3*Nside - 1
        alm = hp.map2alm(y_map, lmax=lmax)

        # Compute C_ell from alm
        cl = hp.alm2cl(alm)  # returns array of length lmax+1

        ell = np.arange(len(cl))

        plt.figure(figsize=(8,5))
        T_CMB = 2.7255e6  # in μK
        f_nu = -1.04      #
        Cl_TT = (T_CMB * f_nu)**2 * cl
        plt.loglog(ell, ell*(ell+1)/(2*np.pi) * Cl_TT, color='green')
        plt.xlabel(r'Multipole $\ell$')
        plt.ylabel(r'$\ell(\ell+1) C_\ell^{\rm tSZ} / 2\pi\ \ [{\mu}\mathrm{K}^2]$')
        plt.title('tSZ Angular Power Spectrum (One-Halo Map)')
        plt.grid(True, which='both', ls='--', alpha=0.5)

        
        # Load the digitized points
        data = np.loadtxt("Planck_error_bars.txt")

        # Check shape
        N = data.shape[0]
        if N % 2 != 0:
            raise ValueError("Number of points must be even. Each error bar requires 2 points.")

        x = data[:, 0]
        y = data[:, 1]

        # Plot error bars by pairing points
        for i in range(0, N, 2):
            x1, y1 = x[i], y[i]
            x2, y2 = x[i+1], y[i+1]

            # Plot as a line between the two points
            plt.plot([x1, x2], [y1, y2], color="blue", linewidth=1.8)

        plt.xlim([5, 4e3])
        plt.ylim([1e-2, 20])

        data = np.loadtxt("Planck_dots_bars.txt")
        x = data[:, 0]
        y = data[:, 1]
        plt.plot(x, y, 'ob',markersize=5)

        # Load the (x, y) points
        data = np.loadtxt("Planck_C_l_tSZ_BAO.txt")
        x = data[:, 0]
        y = data[:, 1]

        # --- IMPORTANT ---
        # The region must be a CLOSED polygon.
        # If the first and last points are not the same, we close it manually.
        if x[0] != x[-1] or y[0] != y[-1]:
            x = np.append(x, x[0])
            y = np.append(y, y[0])

        # Plot the outline
        plt.plot(x, y, color='orange', lw=0.1)
        # Make a custom legend entry

        # Shade the inside
        plt.fill(x, y, color='orange', alpha=0.5)

        plt.xlabel(r'$\ell$')
        plt.ylabel(r'$\ell(\ell+1) C_\ell^{\rm tSZ} / 2\pi \; [\mu{\rm K}^2]$')

        data = np.loadtxt("Planck_CMB.txt")
        x = data[:, 0]
        y = data[:, 1]
        plt.plot(x, y, '--', color='brown')

        # Custom legend entries
        custom_line_Nyx = Line2D([0], [0], color='green', lw=2, linestyle='-')
        custom_line_Planck_bars = Line2D([0], [0], color='blue', lw=2, linestyle='--')
        custom_line_CltSZ_BAO = Line2D([0], [0], color='orange', lw=5)   # thick for legend
        custom_line_CMB = Line2D([0], [0], color='brown', lw=2, linestyle='--')
        
        plt.legend([custom_line_Nyx,custom_line_Planck_bars,custom_line_CltSZ_BAO, custom_line_CMB],
           ['Nyx', 'Planck 2015', r'$C_\ell^{\rm tSZ}$+BAO', 'CMB & (1-b)=0.8'])


        plt.savefig("tSZ_spectrum.png", dpi=300)

if __name__ == "__main__":
    main()

