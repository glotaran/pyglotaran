Quickstart/Cheat-Sheet
======================

.. warning::

   pyglotaran is in very early stage of development. You should not use it for
   actual science at the moment.

To start using pyglotaran in your project, you have to import it first:

.. ipython::

   In [1]: import glotaran as gta

Let us get some data to analyze:

.. ipython::

   In [1]: from glotaran.examples.sequential import dataset

   In [2]: dataset

Like all data in ``pyglotaran``, the dataset is a :class:`xarray:xarray.Dataset`.
You can find more information about the ``xarray`` library the `xarray hompage`_.

.. _xarray hompage: http://xarray.pydata.org/en/stable/

The loaded dataset is a simulated sequential model.

To plot our data, we must first import matplotlib.

.. ipython::

   In [1]: import matplotlib.pyplot as plt

Now we can plot some time traces.

.. ipython::

   In [1]: plot_data = dataset.data.sel(spectral=[620, 630, 650], method='nearest')

   @savefig plot_usage_dataset_traces.png width=8in
   In [1]: plot_data.plot.line(x='time', aspect=2, size=5);

We can also plot spectra at different times.

.. ipython::

   In [1]: plot_data = dataset.data.sel(time=[1, 10, 20], method='nearest')

   @savefig plot_usage_dataset_spectra.png width=8in
   In [1]: plot_data.plot.line(x='spectral', aspect=2, size=5);

To get an idea about how to model your data, you should inspect the singular
value decomposition. Pyglotaran has a function to calculate it (among other
things).

.. ipython:: python

   dataset = gta.io.prepare_time_trace_dataset(dataset)
   dataset

First, take a look at the first 10 singular values:

.. ipython::

   In [1]: plot_data = dataset.data_singular_values.sel(singular_value_index=range(0, 10))

   @savefig quickstart_data_singular_values.png width=8in
   In [1]: plot_data.plot(yscale='log', marker='o', linewidth=0, aspect=2, size=5);

To analyze our data, we need to create a model. Create a file called ``model.py``
in your working directory and fill it with the following:


.. code-block:: yaml

   type: kinetic-spectrum

   initial_concentration:
     input:
       compartments: [s1, s2, s3]
       parameters: [input.1, input.0, input.0]

   k_matrix:
     k1:
       matrix:
         (s2, s1): kinetic.1
         (s3, s2): kinetic.2
         (s3, s3): kinetic.3

   megacomplex:
     m1:
       k_matrix: [k1]

   irf:
     irf1:
       type: gaussian
       center: irf.center
       width: irf.width

   dataset:
     dataset1:
       initial_concentration: input
       megacomplex: [m1]
       irf: irf1


Now you can load the model file.

.. ipython::

   @verbatim
   In [1]: model = gta.read_model_from_yml_file('model.yml')

   @suppress
   In [1]: model_spec = """
      ...: type: kinetic-spectrum
      ...:
      ...: initial_concentration:
      ...:   input:
      ...:     compartments: [s1, s2, s3]
      ...:     parameters: [input.1, input.0, input.0]
      ...:
      ...: k_matrix:
      ...:   k1:
      ...:     matrix:
      ...:       (s2, s1): kinetic.1
      ...:       (s3, s2): kinetic.2
      ...:       (s3, s3): kinetic.3
      ...:
      ...: megacomplex:
      ...:   m1:
      ...:     k_matrix: [k1]
      ...:
      ...: irf:
      ...:   irf1:
      ...:     type: gaussian
      ...:     center: irf.center
      ...:     width: irf.width
      ...:
      ...: dataset:
      ...:   dataset1:
      ...:     initial_concentration: input
      ...:     megacomplex: [m1]
      ...:     irf: irf1
      ...: """
      ...: model = gta.read_model_from_yml(model_spec)

You can check your model for problems with ``model.validate``.

.. ipython:: python

   print(model.validate())

Now define some starting parameters. Create a file called ``parameter.yml`` with
the following content.

.. code-block:: yaml

   input:
     - ['1', 1, {'vary': False, 'non-negative': False}]
     - ['0', 0, {'vary': False, 'non-negative': False}]

   kinetic: [
        0.5,
        0.3,
        0.1,
   ]

   irf:
     - ['center', 0.3]
     - ['width', 0.1]

.. ipython::

   @verbatim
   In [1]: parameter = gta.read_parameter_from_yml_file('parameter.yml')

   @suppress
   In [1]: parameter = gta.read_parameter_from_yml("""
      ...:  input:
      ...:    - ['1', 1, {'vary': False, 'non-negative': False}]
      ...:    - ['0', 0, {'vary': False, 'non-negative': False}]
      ...:  kinetic: [
      ...:       0.5,
      ...:       0.3,
      ...:       0.1,
      ...:  ]
      ...:  irf:
      ...:    - ['center', 0.3]
      ...:    - ['width', 0.1]
      ...: """)

You can ``model.validate`` also to check for missing parameters.

.. ipython:: python

   print(model.validate(parameter=parameter))

Since not all problems in the model can be detected automatically it is wise to
visually inspect the model. For this purpose, you can just print the model.

.. ipython:: python

   print(model)

The same way you should inspect your parameters.

.. ipython:: python

   print(parameter)

Now we have everything together to optimize our parameters.

.. ipython:: python

   result = model.optimize(parameter, {'dataset1': dataset})
   print(result)
   print(result.optimized_parameter)

You can get the resulting data for your dataset with ``result.get_dataset``.

.. ipython:: python

   result_dataset = result.get_dataset('dataset1')
   result_dataset

The resulting data can be visualized the same way as the dataset. To judge the
quality of the fit, you should look at first left and right singular vectors of
the residual.

.. ipython::

   In [1]: plot_data = result_dataset.residual_left_singular_vectors.sel(left_singular_value_index=0)

   @savefig plot_quickstart_lsv.png width=8in
   In [1]: plot_data.plot.line(x='time', aspect=2, size=5);

.. ipython::

   In [1]: plot_data = result_dataset.residual_right_singular_vectors.sel(right_singular_value_index=0)

   @savefig plot_quickstart_rsv.png width=8in
   In [1]: plot_data.plot.line(x='spectral', aspect=2, size=5);

Finally, you can save your result.

.. ipython:: python
   :verbatim:

   result_dataset.to_netcdf('dataset1.nc')
