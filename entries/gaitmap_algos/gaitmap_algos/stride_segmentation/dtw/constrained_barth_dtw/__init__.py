shared_metadata = {
    "references": ["https://www.mdpi.com/1424-8220/15/3/6419"],
    "code_authors": ["MaD-DiGait"],
    "algorithm_authors": ["Jens Barth et al."],
    "implementation_url": "https://github.com/mad-lab-fau/gaitmap/blob/master/gaitmap_mad/gaitmap_mad/"
    "stride_segmentation/dtw/_constrained_barth_dtw.py",
}

metadata_default = {
    "short_description": "DTW based stride segmentation algorithm from Barth et al. (2014) with additional constraints",
    "long_description": "This algorithms finds strides by matching them with a predefined template using DTW. "
    "Compared to the normal DTW algorithm, this version set an lower and an upper limit for the stride duration that "
    " can be matched. "
    "This entry uses the default parameters from the gaitmap library that are not specifically "
    "tuned for any dataset. "
    "Further, it uses the default template from the original publication. ",
    **shared_metadata,
}

metadata_optimized = {
    "short_description": "DTW based stride segmentation algorithm from Barth et al. (2014) with additional constraints "
    "and optimized hyper parameters",
    "long_description": "This algorithms finds strides by matching them with a predefined template using DTW. "
    "Compared to the normal DTW algorithm, this version set an lower and an upper limit for the stride duration that "
    " can be matched. "
    "This entry optimizes the `max_cost` and the `max_template_stretch` hyper parameters and the sensor axis used for"
    " matching using a TPE optimizer. "
    "However, the template is still the default template from the original publication. ",
    **shared_metadata,
}
