default_metadata = {
    "short_description": "Stride level double-integration with dedrifting",
    "long_description": "Spatial parameters are estimated by first integrating the gyro data to get the orientation "
    "and then integrating the accelerometer data to get the position. "
    "The acc integration is corrected using a forward-backward integration approach with zero-velocity and level "
    "walking assumptions.",
    "references": [
        "http://www.mdpi.com/1424-8220/17/9/1940",
        "https://linkinghub.elsevier.com/retrieve/pii/S1350453304001195",
    ],
    "code_authors": ["MaD-DiGait"],
    "algorithm_authors": ["See references"],
    "implementation_url": "https://github.com/mad-lab-fau/gaitmap",
}
