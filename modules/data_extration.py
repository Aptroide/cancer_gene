from collections import defaultdict
from bravado.exception import HTTPNotFound, HTTPError
import pandas as pd
import os
from rich import print

def fetch_all_studies(cbioportal):
    """
    Retrieves all available studies using the cBioPortal API.

    Returns:
        list: List of study identifiers.
    """
    try:
        studies = cbioportal.Studies.getAllStudiesUsingGET().result()
        study_ids = [study['studyId'] for study in studies]
        return study_ids
    except Exception as e:
        print(f"Error retrieving the list of studies: {e}")
        return []

def get_mutation_profile_id(cbioportal, study_id):
    """
    Retrieves the mutation molecular profile ID for a given study.

    Args:
        cbioportal: Configured cBioPortal client.
        study_id: Study ID.

    Returns:
        Mutation molecular profile ID or None if not found.
    """
    try:
        molecular_profiles = cbioportal.molecular_profiles.getAllMolecularProfilesInStudyUsingGET(
            studyId=study_id
        ).response().result

        # Filter mutation profiles
        mutation_profiles = [mp['molecularProfileId'] for mp in molecular_profiles if 'mutation' in mp['molecularProfileId'].lower()]

        if mutation_profiles:
            return mutation_profiles[0]  # Assume at least one mutation profile exists
        else:
            # print(f"No molecular mutation profile found for study '{study_id}'.")
            return None
    except HTTPError as e:
        print(f"Error retrieving molecular profiles for study '{study_id}': {e}")
        return None

def get_sample_list_id(cbioportal, study_id):
    """
    Retrieves the "all" sample list ID for a given study.

    Args:
        cbioportal: Configured cBioPortal client.
        study_id: Study ID.

    Returns:
        "All" sample list ID or None if not found.
    """
    try:
        sample_lists = cbioportal.sample_lists.getAllSampleListsInStudyUsingGET(
            studyId=study_id
        ).response().result

        # Filter sample lists with 'all' in the ID or name
        all_sample_lists = [sl['sampleListId'] for sl in sample_lists if 'all' in sl['sampleListId'].lower()]

        if all_sample_lists:
            return all_sample_lists[0]  # Assume at least one "all" sample list exists
        else:
            print(f"No 'all' sample list found for study '{study_id}'.")
            return None
    except HTTPError as e:
        print(f"Error retrieving sample lists for study '{study_id}': {e}")
        return None

def process_study(cbioportal, study_id, max_gen=None):
    """
    Processes a given study by its study_id: retrieves mutations, calculates statistics, and saves a CSV.

    Args:
        cbioportal: Configured cBioPortal client.
        study_id: Study ID.
    """
    # print(f"\nProcessing study: {study_id}")

    # Retrieve the mutation molecular profile ID
    molecular_profile_id = get_mutation_profile_id(cbioportal, study_id)
    if not molecular_profile_id:
        print(f"Skipping study '{study_id}' due to missing mutation molecular profile.")
        return

    # Retrieve the "all" sample list ID
    sample_list_id = get_sample_list_id(cbioportal, study_id)
    if not sample_list_id:
        print(f"Skipping study '{study_id}' due to missing 'all' sample list.")
        return

    # Retrieve all samples in the study
    try:
        samples_response = cbioportal.samples.getAllSamplesInStudyUsingGET(
            studyId=study_id
        ).response().result
        total_samples = len(samples_response)
        # print(f"Total samples profiled in study '{study_id}': {total_samples}")
    except HTTPNotFound as e:
        print(f"Error: Study '{study_id}' not found.")
        return
    except HTTPError as e:
        print(f"HTTP Error retrieving samples for study '{study_id}': {e}")
        return

    # Retrieve all mutations in the molecular profile and sample list
    try:
        mutations_response = cbioportal.mutations.getMutationsInMolecularProfileBySampleListIdUsingGET(
            molecularProfileId=molecular_profile_id,
            sampleListId=sample_list_id,
            projection="DETAILED"
        ).response().result
        # print(f"Total mutations retrieved: {len(mutations_response)}")
    except HTTPNotFound as e:
        print(f"Error: Molecular profile '{molecular_profile_id}' or sample list '{sample_list_id}' not found for study '{study_id}'.")
        return
    except HTTPError as e:
        print(f"HTTP Error retrieving mutations for study '{study_id}': {e}")
        return

    # Process mutations to aggregate data by gene
    gene_mutation_count = defaultdict(int)
    gene_sample_set = defaultdict(set)

    for mut in mutations_response:
        gene_info = mut.get('gene', {})
        gene = gene_info.get('hugoGeneSymbol')  # Access gene symbol correctly
        sample_id = mut.get('sampleId')
        if gene and sample_id:
            gene_mutation_count[gene] += 1
            gene_sample_set[gene].add(sample_id)

    # Filter only genes with mutations for optimization
    genes_with_mutations = list(gene_mutation_count.keys())
    # print(f"Total genes with at least one mutation: {len(genes_with_mutations)}")

    if not genes_with_mutations:
        print(f"No genes with mutations for study '{study_id}'.")
        return

    # Build the DataFrame
    data = []
    for gene in genes_with_mutations:
        total_mutations = gene_mutation_count.get(gene, 0)
        samples_with_mutations = len(gene_sample_set.get(gene, set()))
        frequency = (samples_with_mutations / total_samples) * 100 if total_samples > 0 else 0
        data.append({
            'Gene': gene,
            '# Mut': total_mutations,
            '#': samples_with_mutations,
            'Frequency (%)': round(frequency, 2)
        })

    df = pd.DataFrame(data)

    if max_gen:
        # Sort the DataFrame by Frequency in descending order
        df_sorted = df.sort_values(by="Frequency (%)", ascending=False).reset_index(drop=True)
        # Select the top genes
        df_final = df_sorted.head(max_gen)

        current_dir = os.getcwd()
        csv_path = os.path.join(current_dir, "Frequencies", f"{study_id}_highFreq.csv")
        df_final.to_csv(csv_path, index=False)

    else:
        # Save the DataFrame to a CSV file
        current_dir = os.getcwd()
        csv_path = os.path.join(current_dir, "Frequencies", f"{study_id}_highFreq.csv")
        df.to_csv(csv_path, index=False)
