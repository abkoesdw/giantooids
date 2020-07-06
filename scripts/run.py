from giantooids.giantooids import DeqSolver
from numpy import linspace


def main():
    deqsolver = DeqSolver()

    # range of shear velocities to model
    ustar1 = linspace(0.05, 0.8, 200)

    # range of equilibrium sizes to calculate; can do finer spacing, but makes
    # the code run slower
    D1 = linspace(400, 5000, 200) * 1e-6
    deqsolver.run_calculation(D1=D1, ustar1=ustar1)

    deqsolver.plot_results()


if __name__ == "__main__":
    main()
