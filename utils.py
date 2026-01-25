import pandas as pd
import re
from tqdm import tqdm

def get_allele_resolution(allele):
    """
    Determine the resolution level of an HLA allele.
    Returns: '2-digit', '4-digit', '6-digit', '8-digit', or 'unknown'
    """
    if pd.isna(allele):
        return 'unknown'
    
    allele_str = str(allele)
    # Count number of colons
    colon_count = allele_str.count(':')
    
    if colon_count == 0:
        return '2-digit'  # e.g., A*01
    elif colon_count == 1:
        return '4-digit'  # e.g., A*01:01
    elif colon_count == 2:
        return '6-digit'  # e.g., A*01:01:01
    elif colon_count == 3:
        return '8-digit'  # e.g., A*01:01:01:01
    else:
        return 'unknown'

def extract_allele_parts(allele):
    """
    Extract components of an HLA allele name.
    Returns dict with locus, 2-digit, 4-digit, 6-digit, 8-digit
    """
    if pd.isna(allele):
        return None
    
    allele_str = str(allele)
    
    # Pattern: LOCUS*XX:XX:XX:XX
    match = re.match(r'([A-Z]+)\*(\d+)(?::(\d+))?(?::(\d+))?(?::(\d+))?', allele_str)
    
    if not match:
        return None
    
    locus = match.group(1)
    field1 = match.group(2)
    field2 = match.group(3) if match.group(3) else None
    field3 = match.group(4) if match.group(4) else None
    field4 = match.group(5) if match.group(5) else None
    
    return {
        'locus': locus,
        '2digit': f"{locus}*{field1}",
        '4digit': f"{locus}*{field1}:{field2}" if field2 else None,
        '6digit': f"{locus}*{field1}:{field2}:{field3}" if field3 else None,
        '8digit': f"{locus}*{field1}:{field2}:{field3}:{field4}" if field4 else None,
        'resolution': get_allele_resolution(allele_str)
    }


# Check for G groups (identified by * in frequency columns like "8.8 (*)")
def is_g_group(value):
    if pd.isna(value):
        return False
    return '*' in str(value)




def clean_data(df, class1_only=True, remove_g_groups=True, verbose=True):
    """
    Clean the allele frequency dataset.
    """
    df_clean = df.copy()
    
    if verbose:
        print(f"Starting shape: {df_clean.shape}")
    
    # Filter for HLA group only
    hla_mask = df_clean["group"] == "hla"
    df_clean = df_clean[hla_mask]
    if verbose:
        print(f"After filtering for HLA: {df_clean.shape}")
    
    # Filter for Class I alleles if specified
    if class1_only:
        class1_mask = df_clean["gene"].isin(["A", "B", "C"])
        df_clean = df_clean[class1_mask]
        if verbose:
            print(f"After filtering for Class I (A, B, C): {df_clean.shape}")
    
    # Identify G-group entries
    df_clean["is_ggroup_allele"] = df_clean["alleles_over_2n"].apply(is_g_group)
    
    # Remove G-group rows if specified
    if remove_g_groups:
        g_group_count = df_clean["is_ggroup_allele"].sum()
        df_clean = df_clean[df_clean["is_ggroup_allele"] == False]
        if verbose:
            print(f"After removing {g_group_count} G-group rows: {df_clean.shape}")
    
    # Drop unnecessary columns
    df_clean = df_clean.drop(columns=["group", "indivs_over_n", "is_ggroup_allele"], errors='ignore')
    if verbose:
        print(f"Final shape after dropping columns: {df_clean.shape}")

    # parse n column 
    df_clean["n"] = df_clean["n"].str.replace(",", "", regex=False)

    # Define proper column data types
    dtype_dict = {
        'gene': 'category',         # Limited values (A, B, C, DRB1, etc.)
        'allele': 'string',         # Text but many unique values
        'population': 'string',     # Text with many unique values
        'alleles_over_2n': 'float64', # Frequency, can have NaN
        'n': 'Int64'                # Integer, using nullable Int64 for potential missing values
    }
    df_clean = df_clean.astype(dtype_dict)

    return df_clean


def filter_data(df, min_n=100, min_allele_variety=50, verbose=True):
    """
    Filter the allele frequency dataset based on quality criteria.
    """
    df_filtered = df.copy()
    
    if verbose:
        print(f"Starting shape: {df_filtered.shape}")
        print(f"Starting number of populations: {df_filtered['population'].nunique()}")
    
    # Count entries per population
    pop_counts = df_filtered.groupby('population').size().reset_index(name='num_entries')
    
    # Filter populations with enough allele variety
    good_pops = pop_counts[pop_counts['num_entries'] >= min_allele_variety]['population']
    df_filtered = df_filtered[df_filtered['population'].isin(good_pops)]
    
    if verbose:
        print(f"After filtering for populations with >= {min_allele_variety} allele entries: {df_filtered.shape}")
        print(f"  Populations remaining: {df_filtered['population'].nunique()}")
    
    # Filter by minimum sample size
    df_filtered = df_filtered[df_filtered['n'] > min_n]
    
    if verbose:
        print(f"After filtering for n > {min_n}: {df_filtered.shape}")
        print(f"  Populations remaining: {df_filtered['population'].nunique()}")
    
    return df_filtered


def collapse_8digit_to_6digit(df, verbose=True):
    """
    For each study (population), collapse 8-digit alleles into their 6-digit parents.
    
    - If a 6-digit parent exists: update it to max(parent_freq, sum of 8-digit children freq)
    - If no 6-digit parent exists: create one with freq = sum of 8-digit children freq
    - Remove all 8-digit entries after processing
    
    Returns:
        tuple: (collapsed_df, changes_log_df)
    """
    df_result = df.copy()
    
    # Ensure resolution column exists
    if 'resolution' not in df_result.columns:
        df_result['resolution'] = df_result['allele'].apply(get_allele_resolution)
    
    # Add parent allele column for 8-digit entries
    df_result['parent_6digit'] = df_result.apply(
        lambda row: extract_allele_parts(row['allele'])['6digit'] 
        if row['resolution'] == '8-digit' and extract_allele_parts(row['allele']) 
        else None, 
        axis=1
    )
    
    changes_log = []
    
    populations = df_result['population'].unique()
    for pop in tqdm(populations, desc="Collapsing 8-digit to 6-digit", disable=not verbose):
        pop_mask = df_result['population'] == pop
        pop_df = df_result[pop_mask]
        
        # Get 8-digit entries for this population
        eight_digit_entries = pop_df[pop_df['resolution'] == '8-digit']
        
        if len(eight_digit_entries) == 0:
            continue
        
        # Group 8-digit entries by their 6-digit parent
        for parent_6d in eight_digit_entries['parent_6digit'].unique():
            if parent_6d is None:
                continue
            
            # Sum of children frequencies
            children_mask = (pop_mask) & (df_result['parent_6digit'] == parent_6d)
            children_freq_sum = df_result.loc[children_mask, 'alleles_over_2n'].sum()
            
            # Check if 6-digit parent exists
            parent_mask = (pop_mask) & (df_result['allele'] == parent_6d)
            parent_exists = parent_mask.sum() > 0
            
            if parent_exists:
                # Get current parent frequency
                parent_freq = df_result.loc[parent_mask, 'alleles_over_2n'].values[0]
                new_freq = max(parent_freq, children_freq_sum)
                
                # Update parent frequency
                df_result.loc[parent_mask, 'alleles_over_2n'] = new_freq
                
                changes_log.append({
                    'population': pop,
                    'parent_6digit': parent_6d,
                    'action': 'updated',
                    'old_freq': parent_freq,
                    'children_sum': children_freq_sum,
                    'new_freq': new_freq
                })
            else:
                # Create new 6-digit parent entry
                template_idx = df_result[children_mask].index[0]
                template_row = df_result.loc[template_idx].copy()
                
                template_row['allele'] = parent_6d
                template_row['resolution'] = '6-digit'
                template_row['alleles_over_2n'] = children_freq_sum
                template_row['parent_6digit'] = None
                
                df_result = pd.concat([df_result, pd.DataFrame([template_row])], ignore_index=True)
                
                changes_log.append({
                    'population': pop,
                    'parent_6digit': parent_6d,
                    'action': 'created',
                    'old_freq': None,
                    'children_sum': children_freq_sum,
                    'new_freq': children_freq_sum
                })
    
    # Remove all 8-digit entries
    df_result = df_result[df_result['resolution'] != '8-digit']
    
    # Drop the helper column
    df_result = df_result.drop(columns=['parent_6digit'])
    
    if verbose:
        print(f"Collapsed 8-digit to 6-digit: {df.shape} -> {df_result.shape}")
        print(f"  Updates: {sum(1 for c in changes_log if c['action'] == 'updated')}")
        print(f"  Created: {sum(1 for c in changes_log if c['action'] == 'created')}")
    
    return df_result, pd.DataFrame(changes_log)


def collapse_6digit_to_4digit(df, verbose=True):
    """
    For each study (population), collapse 6-digit alleles into their 4-digit parents.
    
    - If a 4-digit parent exists: update it to max(parent_freq, sum of 6-digit children freq)
    - If no 4-digit parent exists: create one with freq = sum of 6-digit children freq
    - Remove all 6-digit entries after processing
    
    Returns:
        tuple: (collapsed_df, changes_log_df)
    """
    df_result = df.copy()
    
    # Ensure resolution column exists
    if 'resolution' not in df_result.columns:
        df_result['resolution'] = df_result['allele'].apply(get_allele_resolution)
    
    # Add parent allele column for 6-digit entries
    df_result['parent_4digit'] = df_result.apply(
        lambda row: extract_allele_parts(row['allele'])['4digit'] 
        if row['resolution'] == '6-digit' and extract_allele_parts(row['allele']) 
        else None, 
        axis=1
    )
    
    changes_log = []
    
    populations = df_result['population'].unique()
    for pop in tqdm(populations, desc="Collapsing 6-digit to 4-digit", disable=not verbose):
        pop_mask = df_result['population'] == pop
        pop_df = df_result[pop_mask]
        
        # Get 6-digit entries for this population
        six_digit_entries = pop_df[pop_df['resolution'] == '6-digit']
        
        if len(six_digit_entries) == 0:
            continue
        
        # Group 6-digit entries by their 4-digit parent
        for parent_4d in six_digit_entries['parent_4digit'].unique():
            if parent_4d is None:
                continue
            
            # Sum of children frequencies
            children_mask = (pop_mask) & (df_result['parent_4digit'] == parent_4d)
            children_freq_sum = df_result.loc[children_mask, 'alleles_over_2n'].sum()
            
            # Check if 4-digit parent exists
            parent_mask = (pop_mask) & (df_result['allele'] == parent_4d)
            parent_exists = parent_mask.sum() > 0
            
            if parent_exists:
                # Get current parent frequency
                parent_freq = df_result.loc[parent_mask, 'alleles_over_2n'].values[0]
                new_freq = max(parent_freq, children_freq_sum)
                
                # Update parent frequency
                df_result.loc[parent_mask, 'alleles_over_2n'] = new_freq
                
                changes_log.append({
                    'population': pop,
                    'parent_4digit': parent_4d,
                    'action': 'updated',
                    'old_freq': parent_freq,
                    'children_sum': children_freq_sum,
                    'new_freq': new_freq
                })
            else:
                # Create new 4-digit parent entry
                template_idx = df_result[children_mask].index[0]
                template_row = df_result.loc[template_idx].copy()
                
                template_row['allele'] = parent_4d
                template_row['resolution'] = '4-digit'
                template_row['alleles_over_2n'] = children_freq_sum
                template_row['parent_4digit'] = None
                
                df_result = pd.concat([df_result, pd.DataFrame([template_row])], ignore_index=True)
                
                changes_log.append({
                    'population': pop,
                    'parent_4digit': parent_4d,
                    'action': 'created',
                    'old_freq': None,
                    'children_sum': children_freq_sum,
                    'new_freq': children_freq_sum
                })
    
    # Remove all 6-digit entries
    df_result = df_result[df_result['resolution'] != '6-digit']
    
    # Drop the helper column
    df_result = df_result.drop(columns=['parent_4digit'])
    
    if verbose:
        print(f"Collapsed 6-digit to 4-digit: {df.shape} -> {df_result.shape}")
        print(f"  Updates: {sum(1 for c in changes_log if c['action'] == 'updated')}")
        print(f"  Created: {sum(1 for c in changes_log if c['action'] == 'created')}")
    
    return df_result, pd.DataFrame(changes_log)


def find_2digit_larger_than_children(df, threshold=0.001, verbose=True):
    """
    Find 2-digit alleles where the parent frequency is greater than 
    the sum of its 4-digit children frequencies.
    
    Returns:
        DataFrame with columns: population, n, parent_2digit, parent_freq, 
                               children_sum, difference, num_4digit_children
    """
    df_analysis = df.copy()
    
    # Ensure resolution column exists
    if 'resolution' not in df_analysis.columns:
        df_analysis['resolution'] = df_analysis['allele'].apply(get_allele_resolution)
    
    # Get 2-digit parent for 4-digit entries
    df_analysis['parent_2digit'] = df_analysis.apply(
        lambda row: extract_allele_parts(row['allele'])['2digit'] 
        if row['resolution'] == '4-digit' and extract_allele_parts(row['allele']) 
        else None, 
        axis=1
    )
    
    results = []
    
    populations = df_analysis['population'].unique()
    for pop in tqdm(populations, desc="Finding 2-digit inconsistencies", disable=not verbose):
        pop_mask = df_analysis['population'] == pop
        pop_df = df_analysis[pop_mask]
        
        # Get all 2-digit entries for this population
        two_digit_entries = pop_df[pop_df['resolution'] == '2-digit']
        
        for _, parent_row in two_digit_entries.iterrows():
            parent_allele = parent_row['allele']
            parent_freq = parent_row['alleles_over_2n']
            
            # Find 4-digit children of this 2-digit parent
            children_mask = (pop_mask) & (df_analysis['parent_2digit'] == parent_allele)
            children_freq_sum = df_analysis.loc[children_mask, 'alleles_over_2n'].sum()
            num_children = children_mask.sum()
            
            # Check if parent > sum of children
            if parent_freq > children_freq_sum + threshold and num_children > 0:
                results.append({
                    'population': pop,
                    'n': parent_row['n'],
                    'parent_2digit': parent_allele,
                    'parent_freq': parent_freq,
                    'children_sum': children_freq_sum,
                    'difference': parent_freq - children_freq_sum,
                    'num_4digit_children': num_children
                })
    
    return pd.DataFrame(results)


def remove_inconsistent_2digit_studies(df, max_total_diff=0.005, verbose=True):
    """
    Remove studies where the total frequency difference between 2-digit parents 
    and their 4-digit children exceeds the threshold.
    
    Args:
        df: DataFrame with allele data
        max_total_diff: Maximum allowed total frequency difference per study
        verbose: Print progress information
    
    Returns:
        DataFrame with inconsistent studies removed
    """
    larger_parents_df = find_2digit_larger_than_children(df, verbose=verbose)
    
    if len(larger_parents_df) == 0:
        if verbose:
            print("No inconsistent 2-digit studies found")
        return df
    
    # Summarize by study
    study_summary = larger_parents_df.groupby('population').agg({
        'difference': 'sum'
    }).reset_index()
    study_summary.columns = ['population', 'total_diff']
    
    # Identify studies to remove
    studies_to_remove = study_summary[study_summary['total_diff'] > max_total_diff]['population'].tolist()
    
    df_filtered = df[~df['population'].isin(studies_to_remove)]
    
    if verbose:
        print(f"Removed {len(studies_to_remove)} studies with total_diff > {max_total_diff}")
        print(f"  Shape: {df.shape} -> {df_filtered.shape}")
        print(f"  Studies: {df['population'].nunique()} -> {df_filtered['population'].nunique()}")
    
    return df_filtered


def validate_frequency_sums(df, threshold=0.1, verbose=True):
    """
    Validate that allele frequency sums are within [1-threshold, 1+threshold] per gene per population.
    
    Returns:
        tuple: (valid_combinations_df, invalid_combinations_df)
    """
    # Group by population and gene
    freq_sums = df.groupby(['population', 'gene']).agg({
        'alleles_over_2n': 'sum',
        'n': 'first'
    }).reset_index()
    freq_sums.columns = ['population', 'gene', 'total_freq', 'n']
    
    # Find invalid entries
    invalid_mask = (freq_sums['total_freq'] > 1.0 + threshold) | (freq_sums['total_freq'] < 1.0 - threshold)
    invalid_entries = freq_sums[invalid_mask].copy()
    valid_entries = freq_sums[~invalid_mask]
    
    if verbose:
        print(f"Frequency sum validation (threshold={threshold}):")
        print(f"  Valid (pop, gene) combinations: {len(valid_entries)}")
        print(f"  Invalid (pop, gene) combinations: {len(invalid_entries)}")
        if len(invalid_entries) > 0:
            print(f"  Unique studies with invalid sums: {invalid_entries['population'].nunique()}")
    
    return valid_entries, invalid_entries


def remove_invalid_freq_combinations(df, threshold=0.1, verbose=True):
    """
    Remove (population, gene) combinations where frequency sums are outside [1-threshold, 1+threshold].
    
    Returns:
        DataFrame with invalid combinations removed
    """
    _, invalid_entries = validate_frequency_sums(df, threshold, verbose=False)
    
    if len(invalid_entries) == 0:
        if verbose:
            print("All frequency sums are valid")
        return df
    
    invalid_combinations = set(zip(invalid_entries['population'], invalid_entries['gene']))
    
    df_result = df.copy()
    df_result['pop_gene'] = list(zip(df_result['population'], df_result['gene']))
    df_cleaned = df_result[~df_result['pop_gene'].isin(invalid_combinations)]
    df_cleaned = df_cleaned.drop(columns=['pop_gene'])
    
    if verbose:
        print(f"Removed {len(invalid_combinations)} invalid (population, gene) combinations")
        print(f"  Shape: {df.shape} -> {df_cleaned.shape}")
        print(f"  Studies: {df['population'].nunique()} -> {df_cleaned['population'].nunique()}")
    
    return df_cleaned


def collapse_to_4digit(df, 
                       remove_inconsistent_studies=True,
                       max_2digit_diff=0.005,
                       freq_sum_threshold=0.1,
                       min_sample_size=100,
                       verbose=True):
    """
    Complete pipeline to collapse allele data to 4-digit resolution with proper frequency sums.
    
    Pipeline steps:
    1. Add resolution column
    2. Collapse 8-digit → 6-digit alleles
    3. Collapse 6-digit → 4-digit alleles
    4. Remove studies with inconsistent 2-digit parent frequencies (optional)
    5. Remove all 2-digit entries
    6. Validate and remove (population, gene) combinations with invalid frequency sums
    7. Filter by minimum sample size
    
    Args:
        df: Input DataFrame with allele frequency data (should be cleaned first with clean_data)
        remove_inconsistent_studies: Whether to remove studies where 2-digit parent > children sum
        max_2digit_diff: Maximum allowed total frequency difference for 2-digit inconsistency
        freq_sum_threshold: Threshold for valid frequency sum range [1-threshold, 1+threshold]
        min_sample_size: Minimum sample size (n) to include
        verbose: Print progress information
    
    Returns:
        DataFrame with only 4-digit resolution alleles and validated frequency sums
    """
    if verbose:
        print("="*80)
        print("Starting collapse_to_4digit pipeline")
        print("="*80)
        print(f"Input shape: {df.shape}")
        print(f"Input studies: {df['population'].nunique()}")
    
    # Step 1: Add resolution column if not present
    df_result = df.copy()
    if 'resolution' not in df_result.columns:
        df_result['resolution'] = df_result['allele'].apply(get_allele_resolution)
    
    if verbose:
        print(f"\nResolution distribution before collapse:")
        print(df_result['resolution'].value_counts().to_dict())
    
    # Step 2: Collapse 8-digit to 6-digit
    if verbose:
        print(f"\n--- Step 1: Collapse 8-digit to 6-digit ---")
    df_result, _ = collapse_8digit_to_6digit(df_result, verbose=verbose)
    
    # Step 3: Collapse 6-digit to 4-digit
    if verbose:
        print(f"\n--- Step 2: Collapse 6-digit to 4-digit ---")
    df_result, _ = collapse_6digit_to_4digit(df_result, verbose=verbose)
    
    # Step 4: Remove inconsistent 2-digit studies
    if remove_inconsistent_studies:
        if verbose:
            print(f"\n--- Step 3: Remove inconsistent 2-digit studies ---")
        df_result = remove_inconsistent_2digit_studies(df_result, max_total_diff=max_2digit_diff, verbose=verbose)
    
    # Step 5: Remove all 2-digit entries
    if verbose:
        print(f"\n--- Step 4: Remove 2-digit entries ---")
    before_count = len(df_result)
    df_result = df_result[df_result['resolution'] == '4-digit']
    if verbose:
        print(f"Removed {before_count - len(df_result)} 2-digit entries")
        print(f"  Shape: {before_count} -> {len(df_result)}")
    
    # Step 6: Validate and remove invalid frequency combinations
    if verbose:
        print(f"\n--- Step 5: Validate frequency sums ---")
    df_result = remove_invalid_freq_combinations(df_result, threshold=freq_sum_threshold, verbose=verbose)
    
    # Step 7: Filter by minimum sample size
    if min_sample_size > 0:
        if verbose:
            print(f"\n--- Step 6: Filter by sample size >= {min_sample_size} ---")
        before_studies = df_result['population'].nunique()
        df_result = df_result[df_result['n'] >= min_sample_size]
        if verbose:
            print(f"Studies: {before_studies} -> {df_result['population'].nunique()}")
    
    # Final summary
    if verbose:
        print(f"\n{'='*80}")
        print("Pipeline complete!")
        print(f"{'='*80}")
        print(f"Final shape: {df_result.shape}")
        print(f"Final studies: {df_result['population'].nunique()}")
        print(f"Resolution distribution: {df_result['resolution'].value_counts().to_dict()}")
    
    return df_result


