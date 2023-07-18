shared_metadata = {
    "references": [
        "https://doi.org/10.1186/s12984-021-00883-7",
        "https://www.mad.tf.fau.de/person/liv-herzer-2/",
        "TODO: CITATION FOR RTS KALMAN",
    ],
    "code_authors": ["MaD-DiGait"],
    "algorithm_authors": ["See source and references for individual algorithm authors"],
    "implementation_url": "https://github.com/mad-lab-fau/gaitmap",
}

default_metadata = {
    "short_description": "Modern MaD pipeline",
    "long_description": "Full gait analysis pipeline using more recent algorithms published by the MaD-Lab. "
    "The parameters of these algorithms are not specifically tuned for this dataset.",
    **shared_metadata,
}

improved_zupt_metadata = {
    "short_description": "Modern MaD pipeline with optimized ZUPT detection",
    "long_description": "A version of the modern MaD pipeline with more advanced ZUPT detection. Specfically, this"
    "method forces the existence of one ZUPT event per stride. "
    "This can help with fast walks, where a simple threshold based ZUPT detector tuned for "
    "normal speeds might not be able to detect ZUPT events.",
    **shared_metadata,
}
