==============================
Data types
==============================

ModelData
---------
Defines a custom data type called `ModelData` in AiiDA, which is a subclass of the `SinglefileData` type. `ModelData` is used to handle model files and provides functionalities for handling local files and downloading files from URLs.
Additional features compared to `SinglefileData`:

- it can take a relative path as an argument

- if given a URL it will download the file, save it in a folder of choice (default = ./cache/mlips, if given an architecture it will create a subfolder with that name), and save the file as AiiDA data type. If the file is downloaded twice it will be canceled if there are duplicates in the same folder, however this is not related with AiiDA caching for its own database. For that, enable the caching for the ModelData class, or the whole aiida-mlip plugin (https://aiida.readthedocs.io/projects/aiida-core/en/latest/topics/provenance/caching.html)

- *others to be added? e.g. read architecture from file itself if possible?*

Usage
^^^^^

- To create a `ModelData` object from a local file:

.. code-block:: python

    model = ModelData.local_file('/path/to/file', filename='model', architecture='mace')

- To download a file and save it as a `ModelData` object:

.. code-block:: python

    model = ModelData.download('http://yoururl.test/model', filename='model', cache_dir='/home/mlip/', architecture='mace')
