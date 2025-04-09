import numpy as np
import matplotlib.pyplot as plt

class QuantumHarmonicOscillatorPIMC:
    def __init__(self, m=1.0, omega=1.0, a=0.1, Nt=100, num_sweeps=10000, step_size=1.0):
        self.m = m
        self.omega = omega
        self.a = a
        self.Nt = Nt
        self.num_sweeps = num_sweeps
        self.step_size = step_size
        self.path = np.random.randn(Nt)
        self.accepts = 0

    def V(self, x):
        return 0.5 * self.m * self.omega**2 * x**2

    def kinetic(self, x_prev, x_next):
        return 0.5 * self.m * ((x_next - x_prev) / self.a)**2

    def local_action(self, i):
        x_i = self.path[i]
        x_prev = self.path[(i - 1) % self.Nt]
        x_next = self.path[(i + 1) % self.Nt]
        T = self.kinetic(x_prev, x_i) + self.kinetic(x_i, x_next)
        V = self.V(x_i) * self.a
        return T + V

    def metropolis_step(self):
        for i in range(self.Nt):
            old_x = self.path[i]
            old_S = self.local_action(i)

            new_x = old_x + np.random.uniform(-self.step_size, self.step_size)
            self.path[i] = new_x
            new_S = self.local_action(i)

            dS = new_S - old_S
            if np.random.rand() >= np.exp(-dS):
                self.path[i] = old_x  # reject
            else:
                self.accepts += 1

    def run(self, burn_in=1000):
        configs = []
        for sweep in range(self.num_sweeps):
            self.metropolis_step()
            if sweep >= burn_in:
                configs.append(self.path.copy())
        accept_rate = self.accepts / (self.Nt * self.num_sweeps)
        print(f"Acceptance Rate: {accept_rate:.3f}")
        return np.array(configs)

    def compute_energies(self, paths):
        T_vals = []
        V_vals = []

        for path in paths:
            T_sum = 0
            V_sum = 0
            for i in range(self.Nt):
                x_i = path[i]
                x_next = path[(i + 1) % self.Nt]
                T = 0.5 * self.m * ((x_next - x_i) / self.a)**2
                V = self.V(x_i)
                T_sum += T
                V_sum += V
            T_avg = T_sum / self.Nt
            V_avg = V_sum / self.Nt
            T_vals.append(T_avg)
            V_vals.append(V_avg)

        T_mean = np.mean(T_vals)
        V_mean = np.mean(V_vals)
        E0 = T_mean + V_mean

        return E0, T_mean, V_mean

    def plot_probability_distribution(self, paths):
        all_x = paths.flatten()
        plt.hist(all_x, bins=100, density=True, label='P(x) from MC')
        x = np.linspace(-4, 4, 400)
        psi_sq = np.sqrt(self.m * self.omega / np.pi) * np.exp(-self.m * self.omega * x**2)
        plt.plot(x, psi_sq, label='|ψ₀(x)|² (theory)', linestyle='--')
        plt.xlabel("x")
        plt.ylabel("Probability Density")
        plt.title("Ground State Probability Distribution")
        plt.legend()
        plt.grid(True)
        plt.show()

    def estimate_first_excited_state(self, paths):
        x0s = paths[:, 0]
        correlations = []

        for t in range(1, self.Nt // 2):
            xts = paths[:, t]
            corr = np.mean(x0s * xts)
            correlations.append(corr)

        correlations = np.array(correlations)
        times = np.arange(1, self.Nt // 2) * self.a
        log_corr = np.log(np.abs(correlations))

        # Fit a line to log(C(t)) to extract the gap E1 - E0
        coeffs = np.polyfit(times, log_corr, 1)
        gap = -coeffs[0]

        plt.plot(times, log_corr, label='log(C(t))')
        plt.plot(times, np.polyval(coeffs, times), '--', label=f'Fit: slope = {-gap:.3f}')
        plt.xlabel("Imaginary Time")
        plt.ylabel("log(C(t))")
        plt.title("Autocorrelation Function for First Excited State")
        plt.legend()
        plt.grid(True)
        plt.show()

        return gap
sim = QuantumHarmonicOscillatorPIMC(m=1.0, omega=1.0, a=0.1, Nt=100, num_sweeps=5000)
paths = sim.run(burn_in=500)

E0, T_mean, V_mean = sim.compute_energies(paths)
print(f"Ground State Energy Estimate: E0 = {E0:.4f} (T = {T_mean:.4f}, V = {V_mean:.4f})")

sim.plot_probability_distribution(paths)

gap = sim.estimate_first_excited_state(paths)
print(f"Estimated Energy Gap (E1 - E0): {gap:.4f}")


