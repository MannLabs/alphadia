# Features with zero or minimal positive/negative impact (within ±5 IDs or 0)
zero_impact_features = [
    "num_fragments",
    "num_scans",
    "num_over_50",
    "num_over_0",
    "num_over_0_rank_0_5",
    "num_over_0_rank_6_11",
    "num_over_0_rank_12_17",
    "num_over_0_rank_18_23",
    "num_over_50_rank_0_5",
    "num_over_50_rank_6_11",
    "num_over_50_rank_12_17",
    "num_over_50_rank_18_23",
    "num_profiles",
]

# Features that hurt performance (positive impact when removed)
# Sorted by impact magnitude
bad_features = [
    "mean_correlation",  # +16,867 IDs - WORST
    "idf_xic_dot_product",  # +4,999 IDs
    "hyperscore_inverse_mass_error",  # +3,815 IDs
    "idf_intensity_dot_product",  # +3,056 IDs
    "weighted_mass_error",  # +3,048 IDs
    "delta_rt",  # +1,048 IDs
    "num_over_0_top6_idf",  # +1,092 IDs
    "log10_b_ion_intensity",  # +488 IDs
    "num_profiles_filtered",  # +294 IDs
    "num_over_50_top6_idf",  # +71 IDs
    "longest_b_series",  # +62 IDs
    "log10_y_ion_intensity",  # +15 IDs
]
