=====
Usage
=====

.. warning::

   pyglotaran is in very early stage of development. You should not use it for
   actual science at the moment.

To start using glotaran in your project, you have to import it first:

.. ipython::

   In [1]: import glotaran as gta

Let us get some data to analyze:

.. ipython::

   In [1]: from glotaran.examples.sequential import dataset

   In [2]: dataset

Like all data in `glotaran`, the dataset is a :class:`xarray:xarray.Dataset`.
You can find more infomation about the `xarray` library the `xarray hompage`_.

.. _xarray hompage: http://xarray.pydata.org/en/stable/

The loaded dataset is a simulated sequential model.

.. ipython::

   In [1]: import matplotlib.pyplot as plt

   @savefig plot_usage_dataset_traces.png width=4in
   In [1]: plt.figure()
      ...:
      ...: for i in [620, 630, 650]:
      ...:     dataset.data.sel(spectral=i, method='nearest').plot()

   @savefig plot_usage_dataset_spectra.png width=4in
   In [1]: plt.figure()
      ...: for i in [1, 10, 20]:
      ...:     dataset.data.sel(time=i, method='nearest').plot()

To analyze our data, we need to create a model. Create a file called `model.py`
in your working directory and fill it with the following:


.. code-block:: yaml

   type: kinetic

   initial_concentration:
     input:
       compartments: [s1, s2, s3]
       parameters: [input.1, input.0, input.0]

   k_matrix:
     k1:
       matrix: {
         !tuple '(s2, s1)': kinetic.1,
         !tuple '(s3, s2)': kinetic.2,
         !tuple '(s3, s3)': kinetic.3,
       }

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

.. ipython:: python

   @verbatim
   In [1]: model = gta.load_yml_file('model.yml')

   @suppress
   In [1]: model_spec = """
      ...: type: kinetic
      ...:
      ...: initial_concentration:
      ...:   input:
      ...:     compartments: [s1, s2, s3]
      ...:     parameters: [input.1, input.0, input.0]
      ...:
      ...: k_matrix:
      ...:   k1:
      ...:     matrix: {
      ...:       !tuple '(s2, s1)': kinetic.1,
      ...:       !tuple '(s3, s2)': kinetic.2,
      ...:       !tuple '(s3, s3)': kinetic.3,
      ...:     }
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
      ...: model = gta.load_yml(model_spec)

You can check your model for errors.

.. ipython:: python

   model.valid()
   model.errors()

Now define some starting parameters. Create a file called `parameter.yml` with
the following content.

.. code-block:: yaml

   input:
     - ['1', 1, {'vary': False, 'non-negative': False}]
     - ['0', 0, {'vary': False, 'non-negative': False}]

   kinetic: [
        5,
        0.3,
        0.1,
   ]

   irf:
     - ['center', 0.3]
     - ['width', 0.1]

.. ipython::

   @verbatim
   In [1]: parameter = gta.read_parameter_yaml_file('parameter.yml')

   @suppress
   In [1]: parameter = gta.read_parameter_yaml("""
      ...:  input:
      ...:    - ['1', 1, {'vary': False, 'non-negative': False}]
      ...:    - ['0', 0, {'vary': False, 'non-negative': False}]
      ...:  kinetic: [
      ...:       5,
      ...:       0.3,
      ...:       0.1,
      ...:  ]
      ...:  irf:
      ...:    - ['center', 0.3]
      ...:    - ['width', 0.1]
      ...: """)

   In [1]: print(parameter)

