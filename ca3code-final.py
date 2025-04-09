import numpy as np
import matplotlib.pyplot as plt


class QuantumOscillatorPIMC:
    """
    Path Integral Monte Carlo simulation for harmonic and anharmonic oscillators.
    Includes reweighting, error estimation via binning, and parameter sweeps.
    """

    def __init__(self, m=1.0, omega=1.0, a=0.1, Nt=100, num_sweeps=10000, step_size=1.0,
                 lam=0.0, double_well=False):
        """
        Initialize the simulation parameters.
        """
        self.m = m
        self.omega = omega
        self.a = a
        self.Nt = Nt
        self.num_sweeps = num_sweeps
        self.step_size = step_size
        self.lam = lam
        self.double_well = double_well
        self.path = np.random.randn(Nt)
        self.accepts = 0
        self.collected_energies = []

    def V(self, x):
        """
        Potential energy function:
        Harmonic:       V(x) = ½ m ω² x²
        Anharmonic:     V(x) += λ x⁴
        Double well:    Flip sign of quadratic term
        """
        sign = -1 if self.double_well else 1
        return 0.5 * self.m * sign * self.omega ** 2 * x ** 2 + self.lam * x ** 4

    def kinetic(self, x_prev, x_next):
        """ Discretized kinetic energy term """
        return 0.5 * self.m * ((x_next - x_prev) / self.a) ** 2

    def local_action(self, i):
        """
        Local Euclidean action contribution at site i
        """
        x_i = self.path[i]
        x_prev = self.path[(i - 1) % self.Nt]
        x_next = self.path[(i + 1) % self.Nt]
        T = self.kinetic(x_prev, x_i) + self.kinetic(x_i, x_next)
        V = self.V(x_i) * self.a
        return T + V

    def metropolis_step(self):
        """ Perform a single Metropolis sweep """
        for i in range(self.Nt):
            old_x = self.path[i]
            old_S = self.local_action(i)
            new_x = old_x + np.random.uniform(-self.step_size, self.step_size)
            self.path[i] = new_x
            new_S = self.local_action(i)

            dS = new_S - old_S
            if np.random.rand() >= np.exp(-dS):
                self.path[i] = old_x
            else:
                self.accepts += 1

    def run(self, burn_in=1000):
        """
        Run the Monte Carlo simulation and collect configurations after burn-in.
        """
        configs = []
        energies = []

        for sweep in range(self.num_sweeps):
            self.metropolis_step()
            if sweep >= burn_in:
                configs.append(self.path.copy())
                E, _, _ = self.compute_single_energy(self.path)
                energies.append(E)

        self.collected_energies = np.array(energies)
        acc_rate = self.accepts / (self.Nt * self.num_sweeps)
        print(f"Acceptance Rate: {acc_rate:.3f}")
        return np.array(configs)

    def compute_single_energy(self, path):
        """
        Compute kinetic, potential and total energy for a single path.
        """
        T_sum, V_sum = 0.0, 0.0
        for i in range(self.Nt):
            x_i = path[i]
            x_next = path[(i + 1) % self.Nt]
            T = 0.5 * self.m * ((x_next - x_i) / self.a) ** 2
            V = self.V(x_i)
            T_sum += T
            V_sum += V
        return T_sum / self.Nt + V_sum / self.Nt, T_sum / self.Nt, V_sum / self.Nt

    def compute_energies(self, paths):
        """
        Average over all paths to estimate E0 = <T> + <V>
        """
        T_vals, V_vals = [], []
        for path in paths:
            E, T, V = self.compute_single_energy(path)
            T_vals.append(T)
            V_vals.append(V)

        T_mean = np.mean(T_vals)
        V_mean = np.mean(V_vals)
        return T_mean + V_mean, T_mean, V_mean

    def plot_probability_distribution(self, paths):
        """
        Plot histogram of sampled x-values approximating |ψ₀(x)|².
        """
        all_x = paths.flatten()
        plt.hist(all_x, bins=100, density=True, label='MC P(x)')
        plt.xlabel("x")
        plt.ylabel("Probability Density")
        plt.title(f"Ground State Probability Distribution, m={m},k={k}, ({system_type})")
        plt.grid(True)
        plt.legend()
        plt.show()

    def estimate_first_excited_state(self, paths):
        """
        Estimate energy gap E1 - E0 from autocorrelation decay C(t).
        """
        x0s = paths[:, 0]
        correlations = []
        for t in range(1, self.Nt // 2):
            xts = paths[:, t]
            corr = np.mean(x0s * xts)
            correlations.append(corr)

        correlations = np.array(correlations)
        times = np.arange(1, self.Nt // 2) * self.a
        log_corr = np.log(np.abs(correlations))

        slope, intercept = np.polyfit(times, log_corr, 1)
        gap = -slope

        plt.plot(times, log_corr, label='log C(t)')
        plt.plot(times, slope * times + intercept, '--', label=f'Slope = {-gap:.3f}')
        plt.xlabel("Imaginary Time")
        plt.ylabel("log C(t)")
        plt.title(f"Autocorrelation Decay, m={m},k={k}, ({system_type})")
        plt.legend()
        plt.grid(True)
        plt.show()

        return gap

    def bin_data(self, data, bin_size):
        """
        Bin data and return array of binned means.
        """
        n_bins = data.size // bin_size
        binned = np.array([
            np.mean(data[i * bin_size:(i + 1) * bin_size])
            for i in range(n_bins)
        ])
        return binned

    def standard_error_vs_bin_size(self):
        """
        Estimate standard error for various bin sizes and plot.
        """
        bin_sizes = np.arange(1, 20)
        SEs = []
        for b in bin_sizes:
            binned = self.bin_data(self.collected_energies, b)
            SEs.append(np.std(binned) / np.sqrt(len(binned)))

        plt.plot(bin_sizes, SEs, 'o-')
        plt.xlabel("Bin Size")
        plt.ylabel("Standard Error")
        plt.title("Standard Error vs Bin Size")
        plt.grid(True)
        plt.show()

        return SEs

    def reweight_energy(self, beta_current, beta_target):
        """
        Reweight energy estimate to a different inverse temperature β'.
        """
        d_beta = beta_target - beta_current
        weights = np.exp(-d_beta * self.collected_energies)
        return np.sum(self.collected_energies * weights) / np.sum(weights)

    def sweep_parameters(self, m_values, omega_values):
        """
        Sweep over mass and frequency values, collect results.
        """
        results = []
        for m in m_values:
            for omega in omega_values:
                print(f"\nRunning: m = {m}, omega = {omega}")
                self.__init__(m=m, omega=omega, a=self.a, Nt=self.Nt,
                              num_sweeps=self.num_sweeps, step_size=self.step_size,
                              lam=self.lam, double_well=self.double_well)
                paths = self.run(burn_in=1000)
                E0, T, V = self.compute_energies(paths)
                gap = self.estimate_first_excited_state(paths)
                results.append({
                    'm': m, 'omega': omega,
                    'E0': E0, 'T': T, 'V': V,
                    'gap': gap
                })
        return results







            
            
            
# === Constants ===
hbar = 1.0545718e-34      # [J⋅s]
m_e = 9.10938356e-31      # [kg]
J_to_eV = 1 / 1.602176634e-19  # [eV/J]




# Sweep physical m (in multiples of m_e) and physical k (N/m)
m_values = [0.1, 0.5, 1.0,2.0]               # dimensionless, relative to m_e
k_values = [0.1, 1.0,5.0, 10.0]                # elastic constant [N/m]

# Constants
Nt = 100
a = 0.1
num_sweeps = 5000
burn_in = 500
lam = 1.0

system_types = ['harmonic', 'anharmonic', 'double_well']

for system_type in system_types:
    print("\n" + "="*60)
    print(f"STARTING SIMULATION FOR: {system_type.upper()}")
    print("="*60)

    for m in m_values:
        for k in k_values:
            omega = (k / m)**0.5  # rad/s
            # omega = k

            print(f"\n--- Parameters: m = {m}, k = {k} N/m, ω = {omega:.2e} rad/s ---")
            
            # Configure system based on type
            if system_type == 'harmonic':
                sim = QuantumOscillatorPIMC(m=m, omega=omega, a=a, Nt=Nt,
                                            num_sweeps=num_sweeps, step_size=1.0,
                                            lam=0.0, double_well=False)
            elif system_type == 'anharmonic':
                sim = QuantumOscillatorPIMC(m=m, omega=omega, a=a, Nt=Nt,
                                            num_sweeps=num_sweeps, step_size=1.0,
                                            lam=lam, double_well=False)
            elif system_type == 'double_well':
                sim = QuantumOscillatorPIMC(m=m, omega=omega, a=a, Nt=Nt,
                                            num_sweeps=num_sweeps, step_size=1.0,
                                            lam=lam, double_well=True)

            

        
            paths = sim.run(burn_in=burn_in)
            E0, T, V = sim.compute_energies(paths)
            gap = sim.estimate_first_excited_state(paths)
            E1 = E0 + gap


            print("\n📊 Results (converted to eV):")
            print(f"Ground State Energy E₀ = {E0:.6e} eV")
            print(f"First Excited State Energy E₁ = {E1:.6e} eV")
            print(f"Energy Gap ΔE = {gap:.6e} eV")

            print("Plotting |ψ₀(x)|² ...")
            sim.plot_probability_distribution(paths)
