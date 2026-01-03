import pandas as pd
import re

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