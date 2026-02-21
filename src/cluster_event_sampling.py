import spin_catalog as sc

import sys
import bilby
import json

# --- Run sampler --- #

# Define the prior for the log10 of the axion mass
priors = bilby.core.prior.PriorDict()

priors['axion_mass'] = bilby.core.prior.LogUniform(
    minimum=1e-16, 
    maximum=1e-8, 
    name='axion_mass', 
    latex_label=r'\mu'
)

# Load event data #
with open('catalog_metadata.json', 'r') as file_object:
    # Use json.load() to deserialize the file content into a Python dictionary
    catalog_data = json.load(file_object)

event_index = int(sys.argv[1]) 
single_event = catalog_data[event_index]

event_name = single_event['event']
print(f"--- Running Sampler for {event_name} ---")
    
# Initialize likelihood with a list containing ONLY this event
single_likelihood = sc.MarginalizedCatalogSpinLikelihood(
    event_data=[single_event],
    num_MC_samples=1000
)

# Run the sampler for one event at a time
result = bilby.run_sampler(
    likelihood=single_likelihood,
    priors=priors,
    sampler='dynesty',
    nlive=500, # Adjust based on how fast/accurate you need it
    npool=128,
    outdir='outdir_individuals',
    overwrite=True, # 
    label=f'{event_name}_run'
)

# print("--- Running Final Joint Catalog Sampler ---")

# joint_likelihood = sc.MarginalizedCatalogSpinLikelihood(
#     event_data=catalog_data,  # Pass the whole list here
#     num_MC_samples=1000
# )

# joint_result = bilby.run_sampler(
#     likelihood=joint_likelihood,
#     priors=priors,
#     sampler='dynesty',
#     nlive=1000, # Bumping nlive for the joint run for better resolution
#     outdir='outdir_joint',
#     label='joint_catalog_run'
# )