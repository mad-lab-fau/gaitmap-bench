shared_metadata = {
    "references": ["https://www.mdpi.com/1424-8220/15/3/6419"],
    "code_authors": ["MaD-DiGait"],
    "algorithm_authors": ["Barth et al."],
    "implementation_url": "https://github.com/mad-lab-fau/gaitmap/blob/master/gaitmap_mad/gaitmap_mad/"
    "stride_segmentation/dtw/_barth_dtw.py",
}

metadata_default = {
    "short_description": "DTW based stride segmentation algorithm from Barth et al. (2014)",
    "long_description": "This algorithms finds strides by matching them with a predefined template using DTW. "
    "This entry uses the default parameters from the gaitmap library that are not specifically "
    "tuned for any dataset. "
    "Further, it uses the default template from the original publication. ",
    **shared_metadata,
}

metadata_optimized = {
    "short_description": "DTW based stride segmentation algorithm from Barth et al. (2014) with optimized "
    "hyper-parameters",
    "long_description": "This algorithms finds strides by matching them with a predefined template using DTW. "
    "This entry optimizes the `max_cost` hyper parameters and the sensor axis used for matching using a TPE optimizer. "
    "However, the template is still the default template from the original publication. ",
    **shared_metadata,
}
