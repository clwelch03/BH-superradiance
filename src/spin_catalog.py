import bilby
import numpy as np
from scipy.stats import beta
import superradiance as sr

class CatalogSpinLikelihood(bilby.Likelihood):
	"""
	Catalog likelihood with a single shared parameter (the axion mass) and per-event nuisance parameters.

	Args:
		event_data (list): List of dictionaries containing the beta distribution parameters for each event.
	"""
	def __init__(self, event_data):
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

		self.beta_a = np.array([e['beta_a'] for e in event_data]) 
		self.beta_b = np.array([e['beta_b'] for e in event_data])
	
	
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