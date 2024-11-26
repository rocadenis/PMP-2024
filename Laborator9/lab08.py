import pymc as pm
import arviz as az
import matplotlib.pyplot as plt

Y_clienti = [0, 5, 10] #clienti
theta_values = [0.2, 0.5]#probabilitati de succes

fig, axes = plt.subplots(len(Y_clienti), len(theta_values), figsize=(12, 8), constrained_layout=True)

for i, Y_observed in enumerate(Y_clienti):
    for j, theta in enumerate(theta_values):
        with pm.Model() as model:
            n = pm.Poisson("n", mu=10)
            Y = pm.Binomial("Y", n=n, p=theta, observed=Y_observed)
            trace = pm.sample(draws=1000, tune=1000, cores=4, return_inferencedata=True, progressbar = False)

        ax = axes[i, j]
        az.plot_posterior(trace, var_names=["n"], hdi_prob=0.95, ax=ax)
        ax.set_title(f"Y = {Y_observed}, theta = {theta}")

plt.show()