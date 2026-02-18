import bilby
import numpy as np
from scipy.special import logsumexp
from scipy.stats import beta
import superradiance as sr

class CatalogSpinLikelihood(bilby.Likelihood):

	def __init__(self, event_data):
		"""
		Catalog likelihood with a single shared parameter (the axion mass) and per-event nuisance parameters.

		Args:
			event_data (list): List of dictionaries containing the beta distribution parameters for each event.
		"""
		# Basic setup
		parameters = {"axion_mass": None}
		self.num_events = len(event_data) 
		
		# Create nuisance parameters for each event
		self.mass_keys = [f"BH_1_mass_event_{i}" for i in range(self.num_events)]
		self.spin_keys = [f"BH_1_spin_event_{i}" for i in range(self.num_events)]

		for i in range(self.num_events):
			parameters[self.mass_keys[i]] = None
			parameters[self.spin_keys[i]] = None
		
		# Instantiate as a bilby.Likelihood
		super().__init__(parameters=parameters)

		self.beta_a = np.array([e['spin_alpha'] for e in event_data]) 
		self.beta_b = np.array([e['spin_beta'] for e in event_data])
	
	
	def log_likelihood(self) -> float: # pyright: ignore[reportIncompatibleMethodOverride]
		"""
		Calculates the log likelihood, overriding the base Bilby method.

		Returns:
			float: The combined log-likelihood across all events in the catalog.
		"""
		# Grab parameters
		current_masses = np.array([self.parameters[k] for k in self.mass_keys])
		current_spins = np.array([self.parameters[k] for k in self.spin_keys])
		axion_mass = self.parameters['axion_mass']

		# Convert masses
		current_masses = current_masses * sr.M_SOL_TO_GEOMETRIC
		axion_mass = axion_mass * sr.EV_TO_GEOMETRIC

		# Predict spins based on parameters
		predicted_spins = sr.final_BH_spin_vec(axion_mass, current_masses, current_spins, merger_timescale=sr.TEN_BILLION_YEARS_IN_SECONDS)

		log_probs = beta.logpdf(predicted_spins, self.beta_a, self.beta_b)
		total_log_likelihood = np.sum(log_probs)

		return float(np.nan_to_num(total_log_likelihood, nan=-np.inf))
	

class MarginalizedCatalogSpinLikelihood(bilby.Likelihood):

	def __init__(self, event_data, num_MC_samples=1000):
		"""
		Catalog likelihood in which we marginalize over 

		Args:
			event_data (list): List of dictionaries containing the beta distribution parameters for each event.
			num_MC_samples (int, optional): Number of Monte Carlo samples we take. Defaults to 1000.
		"""
		super().__init__(parameters={'axion_mass': None})

		self.num_events = len(event_data)
		self.num_MC_samples = num_MC_samples

		# Extract measurement distributions for events
		self.beta_a = np.array([e['spin_alpha'] for e in event_data]) 
		self.beta_b = np.array([e['spin_beta'] for e in event_data])

		# Create MC spins and masses
		self.MC_spins = np.random.uniform(0, 1, self.num_MC_samples)

		masses_uniform = np.random.uniform(0, 1, self.num_MC_samples)
		MIN_BH_MASS = 5.0 * sr.M_SOL_TO_GEOMETRIC
		MAX_BH_MASS = 50.0 * sr.M_SOL_TO_GEOMETRIC
		MASS_DIST_EXPONENT = -1.35
		self.MC_masses = ( (MAX_BH_MASS**MASS_DIST_EXPONENT - MIN_BH_MASS**MASS_DIST_EXPONENT)*masses_uniform + MIN_BH_MASS**MASS_DIST_EXPONENT) ** (1.0/MASS_DIST_EXPONENT)

	def log_likelihood(self) -> float: # type: ignore
		"""
		Calculates the log likelihood, overriding the base Bilby method.

		Returns:
			float: The combined log-likelihood across all events in the catalog.
		"""
		axion_mass = self.parameters['axion_mass'] * sr.EV_TO_GEOMETRIC

		flat_MC_masses = self.MC_masses.flatten()
		flat_MC_spins = self.MC_spins.flatten()

		# Calculate predicted spin for each MC sample
		flat_predicted_spins = sr.final_BH_spin_vec(axion_mass,
											  flat_MC_masses,
											  flat_MC_spins,
											  merger_timescale=sr.TEN_BILLION_YEARS_IN_SECONDS)

		# Calculate the log probability of the predicted spins under the given beta distribution
		log_probs = beta.logpdf(
			flat_predicted_spins[np.newaxis, :], 
			self.beta_a[:, np.newaxis], 
			self.beta_b[:, np.newaxis]
		)

		# Marginalize over the MC samples for each event 
		# log( (1/K) * sum(exp(log_probs)) ) = logsumexp(log_probs) - log(K)
		event_log_likelihoods = logsumexp(log_probs, axis=1) - np.log(self.num_MC_samples)

		# Return the sum over all events
		return float(np.nan_to_num(np.sum(event_log_likelihoods), nan=-np.inf))

