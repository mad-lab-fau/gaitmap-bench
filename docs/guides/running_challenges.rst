Running a Challenge
===================

.. note:: This guide focuses on getting your algorithm ready to run a challenge after install the `gaitmap-challenges`
   package as dependency to your own project.
   This is a pre-requisite reading before you attempt to add a new entry to the `gaitmap-challenges` repository.
   For more information about this process, please refer to the :ref:`"New Entry" <new_entry>` guide.

Challenges are Python Classes that you can access through the `gaitmap-challenges` package.
Let's first understand the basic structure of them, before we learn how to run them.

To get started, create a new Python project and then install the `gaitmap-challenges` package as dependency:

.. code-block:: bash

    $ pip install gaitmap-challenges


Within a python file in your project or in a Python console, you should now be able to import a challenge:

.. code-block:: python

    >>> from gaitmap_challenges.stride_segmentation.egait_segmentation_validation_2014 import Challenge
    >>>
    >>> Challenge

The code for this challenge is located in the `gaitmap-challenges` repository at
`gaitmap_challenges/stride_segmentation/egait_segmentation_validation_2014.py`.
Note, that challenges are implemented as dataclasses for convenience.
If you don't know what dataclasses are, you can read more about them `here <https://docs.python.org/3/library/dataclasses.html>`_.

Inspecting the the code and the object in your console, you can see that the challenge object is a class, that expects
information about the data and the cross-validation scheme that should be used.
This can be used to modify how the challenge is run during testing.

Further, most challenges have class-level attributes that can be used to modify how the challenge is fundamentally
executed (Indicated by the `ClassVar` type annotation in the class body).
If you want to change this configuration, we recommend to create a new class that inherits from the challenge class
and overwrite the class-level attributes.
This way, you can follow the same structure as the original challenge, but modify the configuration to your needs.

Besides the attributes, the challenge class implements a couple of helper methods to make it easier for algorithm
developers to access the relevant data (e.g. the `get_imu_data` method).
These methods are implemented as static methods on the class and make use of the class-level attributes to return the
correct data.

Further, the challenge contains helper methods to save and load results (e.g. the `save_core_results` method).
These methods usually don't need to be called directly, but are called by higher-level methods like
:func:`~gaitmap_challenges.results.save_run` or :func:`~gaitmap_challenges.results.load_run`.

Finally, the challenge class implements a `run` method, that is used to run the challenge.
This method takes a `tpcp.BaseOptimize` object as input (we will learn in a second what this is) and will (in this case)
run a cross-validation scheme on the configured dataset and specified algorithm pipeline (the optimize object).
The results of this optimization are then stored in the `cv_results_` (and other) attributes on the challenge instance.

This means running a challenge typically looks something like this:

.. code-block:: python

    >>> dataset = ...
    >>> challenge = Challenge(dataset=dataset)
    >>> my_pipeline = ...
    >>> challenge.run(my_pipeline)
    >>> print(pd.DataFrame(challenge.cv_results_)

So what we are missing is, an understanding of how we define a dataset and how we build a pipeline.
Let's start with the dataset.

Getting Data
------------

All challenges use `gaitmap-datasets <https://github.com/mad-lab-fau/gaitmap-datasets>`_ to load data.
This package provides a dataset interface using :class:`tpcp.Dataset` objects and the download links to the respecitve
datasets in the README.
We can import the respective dataset objects from the `gaitmap-datasets` package and or for convenience directly import
them from the respective challenge module under the `ChallengeDataset` alias:

.. code-block:: python

    >>> from gaitmap_challenges.stride_segmentation.egait_segmentation_validation_2014 import ChallengeDataset
    >>>
    >>> dataset = ChallengeDataset()

You will see, that just doing that, will result in an error, as we haven't specified a data directory yet.
We can do that in two different ways:

1. By just passing the path to the data directory to the dataset object:

.. code-block:: python

    >>> dataset = ChallengeDataset(data_folder="/path/to/data")

2. (Preferred) To avoid hard-coding the path we can also use the `global config <global_config>`_ functionality of
   `gaitmap-challenges`.
   But, we will stick with the first option for this example to keep things simple.

With a dataset object loaded, you can inspect the data, by accessing the `index` attribute, which shows an overview of
the available data-points (i.e. the subjects and trials):

.. code-block:: python

    >>> dataset.index

For more information, check out the respective `gaitmap-datasets example <https://mad-lab-fau.github.io/gaitmap-datasets/auto_examples/index.html>`_
for your dataset.



Building a algorithm pipeline (without Optimization)
----------------------------------------------------

The entire structure of `gaitmap-challenges` is based on the concept of a :class:`tpcp.Pipeline` objects.
A pipeline is a Python object, that gets all its configuration parameters as part of the `__init__` and then has a
`run` method that takes a `tpcp.Dataset` object with only a single datapoint (e.g. a single trial) as input and then
saves results on the pipeline object itself.

There is also a :class:`tpcp.OptimizablePipeline` that implements a similar interface for algorithms that require a
train step, but for simplicity we will stick with the `Pipeline` for now.

The best way, to build a pipeline is to first play with the data in an interactive manner.
To mimic as close as possible, what we need to implement in the pipeline, we directly use the helper methods on our
Challenge.

.. code-block:: python

    >>> from gaitmap_challenges.stride_segmentation.egait_segmentation_validation_2014 import Challenge, ChallengeDataset
    >>>
    >>> dataset = ChallengeDataset()
    >>> # Just one datapoint for testing
    >>> datapoint = dataset[0]
    >>> imu_data = Challenge.get_imu_data(datapoint)
    >>> imu_data

For this challenge, this data represents the input, you algorithm will get.
The imu-data (and all other datatypes) are standardized according to the `gaitmap` specifications.
You can read more about this
`here <https://gaitmap.readthedocs.io/en/latest/source/user_guide/datatypes.html>`__,
`here <https://gaitmap.readthedocs.io/en/latest/source/user_guide/coordinate_systems.html>`__,
and `here <https://gaitmap.readthedocs.io/en/latest/source/user_guide/prepare_data.html>`__.

In addition to the data we can also extract the sampling rate:

.. code-block:: python

    >>> sampling_rate = datapoint.sampling_rate_hz

Now, we can start building and testing our algorithm to produce the desired output given these inputs.
You can either implement your algorithm as a :class:`tpcp.Algorithm` object
(as explained `here <https://gaitmap.readthedocs.io/en/latest/source/user_guide/create_own_algorithm.html>`__) or just a
simple Python function.
We will stick with the function for now.

We will name our algorithm `my_segmentation_algorithm` and we assume that it has a parameter `threshold` that we need to
specify (and later also want to optimize).
The function signature could look something like this:

.. code-block:: python

    >>> def my_segmentation_algorithm(imu_data_single_foot, sampling_rate, threshold):
    ...    # Do something with the data
    ...    segmentation = pd.DataFrame({"start": stride_starts, "end": stride_ends}).rename_axis("s_id")
    ...    return segmentation

Now, we can test our algorithm:

.. code-block:: python

    >>> segmentation = my_segmentation_algorithm(imu_data["left_sensor"], sampling_rate, threshold=0.5)
    >>> segmentation

Note, that we only use the data from one sensor here.
Most of the time it is the best idea to implement your algorithm in a way that it implements the logic for just a
single sensor and you then handle the complexity of multiple sensors in the calling function.
The only exception of course, is when your algorithms needs the data from multiple sensors at the same time.

Now, that we have a working algorithm, we can implement it as a pipeline.
Note, that we expose the `threshold` parameter as a pipeline parameter, so that we can later optimize it:

.. code-block:: python

    from tpcp import Pipeline
    from gaitmap_challenges.stride_segmentation.egait_segmentation_validation_2014 import Challenge

    class MySegmentationPipeline(Pipeline):
        segmentation_ : Dict[str, pd.DataFrame]

        def __init__(self, threshold):
            self.threshold = threshold

        def run(self, datapoint):
            imu_data = Challenge.get_imu_data(datapoint)
            sampling_rate = datapoint.sampling_rate_hz

            segmentations = {
                k: my_segmentation_algorithm(v, sampling_rate, self.threshold) for k, v in imu_data.items()
                }

            self.segmentation_ = segmentations
            return self

For more information on how to implement pipelines and some nice tricks, check out the
`tpcp examples <https://tpcp.readthedocs.io/en/latest/auto_examples/index.html>`_.

Now, we should test our pipeline again to verify, that our pipeline implementation works as intended:

.. code-block:: python

    >>> pipeline = MySegmentationPipeline(threshold=0.5)
    >>> pipeline.run(datapoint)
    >>> pipeline.segmentation_

Now we are ready to run the pipeline as part of a challenge.
We are missing only one thing: Our challenge expects and :class:`tpcp.BaseOptimze` object as input, but we have just a
pipeline.
The reason for this is, that the challenge assumes that we want to optimize/re-train your algorithm on the train sets of
the challenge data.
But, in our case, there is no optimization step, so we can just use the :class:`tpcp.optimize.DummyOptimize` to wrap our
pipeline and then pass it to the challenge run method.
Putting everything together this looks as follows:

.. code-block:: python

    >>> from tpcp.optimize import DummyOptimize
    >>> from gaitmap_challenges.stride_segmentation.egait_segmentation_validation_2014 import Challenge, ChallengeDataset
    >>>
    >>> dataset = ChallengeDataset(data_folder="path/to/your/data")
    >>> pipeline = MySegmentationPipeline(threshold=0.5)
    >>> optimize = DummyOptimize(pipeline)
    >>> challenge = Challenge(dataset=dataset)
    >>> challenge.run(dataset, optimize)
    >>> # Inspect the results
    >>> pd.DataFrame(challenge.cv_results_)

If you want to save the algorithm results, you can use :func:`gaitmap_challenges.utils.save_results`:

.. code-block:: python

    >>> from gaitmap_challenges.utils import save_results
    >>> save_results(challenge, entry_name="MySegmentationPipeline", path = "path/to/your/results")

.. warning:: There is a `save_results` method in `gaitmap-challenges` and one in `gaitmap-bench`.
   If you are using `gaitmap-challenges` standalone, you should use the one from `gaitmap-challenges`, however,
   if you are planning to add an entry to `gaitmap-bench`, you should use the one from `gaitmap-bench`.
   The latter fixes some settings and enforces some conventions on the metadata.

To see a full example of this checkout the
`this entry <https://github.com/mad-lab-fau/gaitmap-bench/blob/main/entries/gaitmap_algos/gaitmap_algos/stride_segmentation/dtw/barth_dtw/egait_segmentation_validation_2014_default.py>`__
for the same challenge.
Note, that we use the entire algorithm as "parameter" here.

Building an Algorithm Pipeline - with Optimization
--------------------------------------------------

In the previous section, we have seen how to build a basic pipeline for a challenge.
For this pipeline there was no (hyper)parameter optimization or model training required.

However, all challenges support this functionality and correctly support optimization within the cross-validation.
To implement this into your pipeline, we need to differentiate between hyperparameter optimization
(external optimization) and model training (internal optimization).

For hyperparameter optimization, we (most likely) don't need to modify our pipeline at all, but just change the
optimization-wrapper.
The `tpcp` package provides a number of different optimization wrappers, that can be used for this purpose.
The easiest one is :class:`~tpcp.optimize.GridSearch`, but in many cases :class:`~tpcp.optimize.optuna.OptunaSearch` is
is a better choice, with access to more powerful optimization algorithms.
For this example, we will use the :class:`~tpcp.optimize.GridSearch`.

To implement a GridSearch, we need to talk about scoring functions.
Each challenge has a scoring method that is used to calculate the error values for each datapoint.
For our optimization, we likely want to use the same scoring function (though we don't have to).
The only thing, we have to keep in mind, is that the scoring function returns multiple metrics, and we need to specify
which want we want to optimize for by setting `return_optimized` to the name of the metric.

.. code-block:: python

    >>> from gaitmap_challenges.stride_segmentation.egait_segmentation_validation_2014 import Challenge
    >>> from tpcp.optimize import GridSearch
    >>>
    >>> optimizer = GridSearch(
    ...     pipeline=MySegmentationPipeline(),
    ...     param_grid={"threshold": [0.1, 0.2, 0.3, 0.4, 0.5]},
    ...     scoring=Challenge.get_scorer(),
    ...     return_optimized="f1_score"
    ... )
    >>> challenge = Challenge(dataset=dataset)
    >>> challenge.run(optimizer)

For a full example using `OptunaSearch` checkout
`this entry <https://github.com/mad-lab-fau/gaitmap-bench/blob/main/entries/gaitmap_algos/gaitmap_algos/stride_segmentation/dtw/barth_dtw/egait_segmentation_validation_2014.py>`__.

In case your algorithm requires model training (e.g. for machine learning), you need to explicitly implement a
:class:`tpcp.OptimizablePipeline`.
How to do this is explained in detail `here <https://tpcp.readthedocs.io/en/latest/auto_examples/parameter_optimization/_02_optimizable_pipelines.html#sphx-glr-auto-examples-parameter-optimization-02-optimizable-pipelines-py>`__.

Once you have implemented your pipeline, you can use the :class:`tpcp.optimize.Optimize` wrapper to pass your new pipeline to the challenge

.. code-block:: python

    >>> from gaitmap_challenges.stride_segmentation.egait_segmentation_validation_2014 import Challenge
    >>> from tpcp.optimize import Optimize
    >>>
    >>> optimizer = Optimize(
    ...     pipeline=MyOptimizableSegmentationPipeline(),
    ... )
    >>> challenge = Challenge(dataset=dataset)
    >>> challenge.run(optimizer)

You could even use :class:`tpcp.optimize.GridSearchCv` (or similar methods) to further optimize hyperparameters of
the training process.

For a full example of using `Optimize` checkout `this entry <https://github.com/mad-lab-fau/gaitmap-bench/blob/main/entries/gaitmap_algos/gaitmap_algos/stride_segmentation/roth_hmm/egait_segmentation_validation_2014_trained.py>`__.