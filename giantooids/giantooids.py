import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
import sys
from tqdm import tqdm
from math import pi


class DeqSolver:
    def __init__(self, *args, **kwargs):
        # parameters and constants for abrasion rate calcs
        self.kv = kwargs.get("kv", 90e4)  # dimensionless tuning parameter
        self.rho_s = kwargs.get("rho_s", 2800)  # [kg/m^3] density of aragonite
        self.rho_w = kwargs.get("rho_w", 1025)  # [kg/m^3] density of seawater
        self.R = (self.rho_s - self.rho_w) / self.rho_w  # submerged specific density
        self.g = kwargs.get("g", 9.81)  # [m/s^2]
        self.nu = kwargs.get("nu", 1.3e-6)  # kinematic viscosity of water
        self.young = kwargs.get("young", 20e9)  # young's modulus
        self.strength = kwargs.get("strength", 1e6)  # tensile strength
        self.tauc = kwargs.get(
            "tauc", 0.03
        )  # critical shields number.  0.03 is good for sand.
        self.gaurds2 = kwargs.get(
            "gaurds2", 1
        )  # this sets limit to Ub if  = 1;  important for fine sediment
        self.Stc = kwargs.get("Stc", 9)  # critical Stokes threshold for viscous damping
        self.A1 = kwargs.get("A1", 0.36)  # dimensionless constant for transport calcs
        self.intermittency = kwargs.get("intermittency", 0.05)

        # in general larger bedforms = less frequent movement = smaller value for
        # intermittency in the range (0 1]
        self.H = kwargs.get("H", 2)  # [m] water depth

        # parameters and constants for precipitation rate calcs
        self.omega1 = kwargs.get("omega1", np.array([12, 14, 16, 18]))
        self.n = kwargs.get("n", 2.26)  # empirical values from Zhong & Mucci 1989
        self.k = kwargs.get("k", 10 ** (1.11))

        # parameters for settling vel and drag coeffs calculation
        self.CSF = kwargs.get("CSF", 1)  # 1 is for spheres, 0.8 is for natural
        self.PS = kwargs.get("PS", 6)  # 6 is for spheres, 3.5 is for natural

        # parameters for calculating shear velocities
        self.P_crit = kwargs.get("P_crit", np.array([0.8, 1.2, 2.5, 7.5]))
        self.beta = kwargs.get("beta", 1)

        # preallocate spaces
        self.D_fOmega = np.empty([])

    def __calculate_settling_vel_and_drag_coeffs(self, D1=None):
        Dstar = (self.R * self.g * np.power(D1, 3)) / (self.nu ** 2)
        X = np.log10(Dstar)

        R1 = (
            -3.76715
            + 1.92944 * X
            - 0.09815 * (X ** 2)
            - 0.00575 * (X ** 3)
            + 0.00056 * (X ** 4)
        )
        R2 = (
            np.log10(1 - ((1 - self.CSF) / 0.85))
            - ((1 - self.CSF) ** 2.3) * np.tanh(X - 4.6)
            + 0.3 * (0.5 - self.CSF) * ((1 - self.CSF) ** 22) * (X - 4.6)
        )
        R3 = (0.65 - ((self.CSF / 2.83) * np.tanh(X - 4.6))) ** (
            1 + (3.5 - self.PS) / 2.5
        )

        Wstar = R3 * (10 ** (R2 + R1))
        self.ws1 = (self.R * self.g * self.nu * Wstar) ** (1.0 / 3)
        self.cdrag1 = (4 / 3) * np.divide((self.R * self.g * D1), (self.ws1 ** 2))

    def run_calculation(self, D1=None, ustar1=None):
        self.ustar1 = ustar1
        self.D1 = D1

        self.__calculate_settling_vel_and_drag_coeffs(D1=self.D1)

        self.D_fOmega = np.zeros((len(self.ustar1), len(self.omega1)))

        pbar = tqdm(
            desc="running calculations",
            total=len(self.omega1) * len(self.ustar1) * len(self.D1),
            unit="it",
            unit_scale=True,
        )
        for i, omega in enumerate(self.omega1):
            precip_rate = self.k * (omega - 1) ** self.n  # [umol/m^2/hr]
            SA_ssa = pi * self.D1 ** 2 * 23  # specific surface area estimate
            precip_rate_vol = (
                precip_rate * 1e-6 * 1e-3 * 100.0869 / self.rho_s * 1e18 * SA_ssa
            )

            for j, ustar in enumerate(self.ustar1):
                En_St_st = np.zeros((1, len(self.D1)))

                for k, D in enumerate(self.D1):
                    ws = self.ws1[k]
                    cdrag = self.cdrag1[k]
                    tau = ustar ** 2 / (self.R * self.g * D)
                    tstage = tau / self.tauc
                    (
                        En_suspt_st,
                        Efactor,
                    ) = self.__susp_abrasion_calculations_giantooids(
                        D=D, ustar=ustar, ws=ws, cdrag=cdrag, tstage=tstage
                    )

                    En_St_st[0, k] = self.A1 * En_suspt_st * Efactor
                    pbar.update(1)

                dVA = 4 * pi * (self.D1 / 2) ** 2 * En_St_st / 1000  # [m^3/yr]

                differences = (
                    precip_rate_vol - dVA * 1e18 / 365 / 24 * self.intermittency
                )

                # check for correctness to confirm
                differences[differences <= 0] = np.inf
                index_min = np.argmin(differences)
                self.D_fOmega[j, i] = self.D1[index_min]

        pbar.close()

    def calculate_shear_vel(self):
        self.ustar_Rouse = np.zeros((len(self.D1), len(self.P_crit)))
        for i, p_crit in enumerate(self.P_crit):
            self.ustar_Rouse[:, i] = self.ws1 / (p_crit * 0.41 * self.beta)

    def plot_results(self):
        self.calculate_shear_vel()
        plt.figure()
        for i in range(self.D_fOmega.shape[1]):
            plt.plot(self.ustar1, self.D_fOmega[:, i] * 1e6)

        plt.xlabel("shear velocity (m/s)")
        plt.ylabel("equilibrium ooid size (/mum)")

        for i in range(self.ustar_Rouse.shape[1]):
            plt.plot(self.ustar_Rouse[:, i], self.D1 * 1e6, "k")

        plt.ylim(1000, 5000)
        plt.xlim(0, 0.8)
        plt.show()

    def __susp_abrasion_calculations_giantooids(
        self, D=None, ustar=None, ws=None, cdrag=None, tstage=None
    ):
        # compute flow velocity
        z0 = (
            3 * D / 30
        )  # This is a roughness coefficient that needs to be set.  Here set by grainsize
        dz = (self.H - z0) / 1000
        z = np.arange(
            z0, self.H, dz
        )  # (start, stop, range), in matlab it was [z0, dz, H]

        # assigned but never used
        # flow_vel = (ustar / 0.41) * np.log(z / z0)
        # flow_z = z

        Uf = np.sum((ustar / 0.41) * np.log(z / z0)) * dz / self.H

        # compute bed load height and velocity
        hb = D * 1.44 * (tstage - 1) ** 0.5  # height of the bed load layer
        # try:
        #     hb = D * 1.44 * (tstage - 1) ** 0.5  # height of the bed load layer

        # except RuntimeWarning:
        #     print(D, tstage)
        #     print(D * 1.44 * (tstage - 1) ** 0.5)
        #     sys.exit()
        Us = (
            (self.R * self.g * D) ** 0.5 * 1.56 * (tstage - 1) ** 0.56
        )  # bed load velocity

        # assigned but never used
        # Us1 = Us

        # don't let particle velocity exceed fluid velocity - this is important
        if self.gaurds2 == 1:
            Us = Uf if Us > Uf else Us

        # Set the sediment supply to be equal to the transport capacity
        # I think the how the particle suspension code is set up, it does need this
        # to drive the concentraiton profile and predict the fall height.  However,
        # it actually does not depend on it. It could be removed, but code needs to
        # be rewritten in terms of concentration gradients rather than absolut
        # concentration
        # qt = 10*5.7.*(R.*g.*D.^3).^0.5.*(tau- tauc).^(3/2);
        qt = 1
        q = qt

        # compute suspended load
        if hb < self.H:
            # sets minimum ht of bed load layer
            hb = D if hb < D else hb
            b = hb  # bottom of the suspended load layer - same as height of bedload

            betta = 2  # Based on Scheingross et al. (2014) best fit
            P = ws / (0.41 * ustar * betta)  # Rouse number

            # This Log scale cleans up the integration
            di5 = 0
            i5 = 0
            res = 1000
            di5 = (np.log(self.H) - np.log(b)) / res
            i5 = np.arange(np.log(b), np.log(self.H), di5)

            z = np.exp(i5)
            z[len(z) - 1] = self.H
            dz = np.diff(z)
            dz = np.append(dz[0], dz)
            a1 = (
                np.sum(
                    (
                        ((1 - (z[z > z0] / self.H)) / (1 - (b / self.H)))
                        * (b / z[z > z0])
                    )
                    ** P
                    * np.log(z[z > z0] / z0)
                    * dz[z > z0]
                )
                / (Uf * self.H)
                * (ustar / 0.41)
            )  # This is just in case b is less than z0

            cb = q / (Us * hb + Uf * self.H * a1)

            # find concentration profile
            c = np.zeros(len(z) + 1)
            c[0] = cb

            c[1 : len(z) + 1] = (
                cb * (((1 - (z / self.H)) / (1 - (b / self.H))) * (b / z)) ** P
            )

            z = np.append(0, z)
            c[z == self.H] = 0

            # calculate the fall distance
            gradc = np.zeros(len(c))
            gradc[1 : len(c)] = -np.diff(c)
            Hfall = (1 / cb) * np.sum(z * gradc)

        else:
            hb = self.H
            cb = q / (Us * hb)
            Hfall = hb
            a1 = 0

        if cb == 0:
            Hfall = 0

        # assigned but never used
        # Sklar settling velocity
        # psi_b_sklar = (
        #     0.84 * (self.R * self.g * D) ** 0.5 * (tstage - 1) ** 0.18 / ws
        # )

        # Probability for fluctuations
        sig = ustar
        dx = (
            sig / 100
        )  # the number of bins to subdivide - can play with this for resolution

        X = np.arange(
            -6 * sig, 6 * sig, dx
        )  # spread distribution for six sigma = w'/ws

        f = scipy.stats.norm.pdf(
            X, 0, sig
        )  # centered at zero normal gausian distribution

        X = X / ws  # to normalize as w'/ws same as psi

        # Calculate impact velocity due to gravity and turbulence
        # cosine of the angle of the bed for impacts.  Assume flat for ocean
        Scos = 1
        wfall = (
            Scos
            * (
                (2 * (2 / 3) * D * self.g / cdrag * self.R)
                * (
                    1
                    - np.exp(
                        -cdrag * self.rho_w / self.rho_s * (Hfall / Scos) / (2 / 3 * D)
                    )
                )
            )
            ** 0.5
        )

        # wfall(Hfall<=(0.5.*D))=0;

        psifall = wfall / ws
        settlematrix = 0
        settlematrix = psifall + X

        # assigned but never used
        # settlematrix1 = settlematrix

        settlematrix[settlematrix < 0] = 0  # no negative impacts

        # assigned but never used
        # psifall_turb = np.sum((settlematrix) * f) * dx
        # psi_fall3 = np.sum((settlematrix ** 3) * f) * dx
        # E1 = psi_fall3  # erosion with turbulence

        # Stokes number correction
        wi_st = settlematrix
        # critical stokes number of 70 based on Scheingross et al 2014
        wi_st[(D * wi_st * ws * self.rho_s / (9 * self.nu * self.rho_w)) < self.Stc] = 0
        psi_fall3_st = np.sum((wi_st ** 3) * f) * dx
        E1_st = psi_fall3_st  # erosion with turbulence and stokes correction

        # NEW - Calculate dimensionless inverse hop times;
        # Don't allow particles to fall from less than half a diameter up.
        # Making this really small rather than zero so it plots
        ti = 0 if Hfall <= 0.5 * D else D / Hfall

        # assigned but never used
        # hoplength = Us * hb / (ws / 3)

        En_suspt_st = 1 / self.kv * (ws / (self.g * D) ** 0.5) ** 3 * E1_st * ti / 6
        En_suspt_st = 0 if En_suspt_st < 0 else En_suspt_st

        # Calculate Sklar Erosion
        En_Sklar = (
            0.456
            * (self.R ** 1.5)
            * self.tauc ** 1.5
            / self.kv
            * (q / qt)
            * (tstage - 1)
            * (1 - (ustar / ws) ** 2) ** (3.0 / 2)
        )

        # This should really be set to zero, but making it small for plotting
        En_Sklar = 0 if ustar > ws else En_Sklar

        Efactor = (
            1000.0
            * 60
            * 60
            * 24
            * 365
            * self.rho_s
            * self.young
            * (self.g * D) ** (3 / 2)
            / (self.strength ** 2)
        )  # multiply by En to get mm/yr

        if tstage <= 1:  # Set erosion to very small if particles stop moving
            En_suspt_st = 0

            # assigned but never used
            # En_suspt = 0

            En_Sklar = 0

        if self.H < D:  # Don't let flow depth be less than the grain size
            # assigned but never used
            # En_suspt = 0
            En_suspt_st = 0

        return En_suspt_st, Efactor
