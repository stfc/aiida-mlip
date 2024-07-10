==============================
Data types
==============================

ModelData
---------
Defines a custom data type called `ModelData` in AiiDA, which is a subclass of the `SinglefileData` type. `ModelData` is used to handle model files and provides functionalities for handling local files and downloading files from URLs.
Additional features compared to `SinglefileData`:

- It can take a relative path as an argument

- It takes the argument "architecture" which is specifically related to the mlip model and it is added to the node attributes.

- Download functionality:
    - When provided with a URL, `ModelData` automatically downloads the file.
    - Saves the downloaded file in a specified folder (default: `./cache/mlips`), creating a subfolder if the architecture, and stores it as an AiiDA data type.
    - Handles duplicate files: if the file is downloaded twice, duplicates within the same folder are canceled, unless `force_download=True` is stated.

Usage
^^^^^

- To create a `ModelData` object from a local file:

.. code-block:: python

    model = ModelData.from_local('/path/to/file', filename='model', architecture='mace')

- To download a file and save it as a `ModelData` object:

.. code-block:: python

    model = ModelData.from_url('http://yoururl.test/model', architecture='mace', filename='model', cache_dir='/home/mlip/', force_download=False)

- The architecture of the model file can be accessed using the `architecture` property:

.. code-block:: python

    model_arch = model.architecture



JanusConfigfile
---------------

The `JanusConfigfile` class is designed to handle config files written for janus-core in YAML format within the AiiDA framework.
This class inherits from `SinglefileData` in the AiiDA, and extends it to support YAML config files.
It provides methods for reading, storing, and accessing the content of the config file.

Usage
^^^^^

- To create a `JanusConfigfile` object:

.. code-block:: python

    config_file = JanusConfigfile('/path/to/config.yml')


- To read the content of the config file as a dictionary, you can use the `read_yaml()` method:

.. code-block:: python

    config_dict = config_file.read_yaml()


- To store the content of the config file in the AiiDA database, you can use the `store_content()` method:

.. code-block:: python

    config_file.store_content(store_all=False, skip=[])

The `store_content()` method accepts the following parameters:

    - `store_all` (bool):
        Determines whether to store all parameters or only specific ones.
        By default, it's set to `False`.
        When set to `False`, only the key parameters relevant for the provenance graph are stored: `code`, `structure`, `model`, `architecture`, `fully_opt` (for GeomOpt), and `ensemble` (for MD).
        However, all inputs can be accessed in the config file at any time (just the config file will appear in the provenance graph as JanusConfigfile).
        If `store_all` is set to `True`, all inputs are stored, either as specific data types (e.g. the input 'struct' is recognised as a StructureData type) or as Str.

    - `skip` (list):
        Specifies a list of parameters that should not be stored.
        In the source code of the calcjobs, when the same parameter is provided both as an AiiDA input and within the config file, the parameter from the config file is ignored and not stored.
        These parameters are added to the `skip` list to ensure they are excluded from storage.


- The filepath of the config file can be accessed using the `filepath` property:

.. code-block:: python

    file_path = config_file.filepath

.. warning::

    When sharing data, using the ``filepath`` could point to a location inaccessible on another computer.
    So if you are using data from someone else, for both the modeldata and the configfile, consider using the ``get_content()`` method to create a new file with identical content.
    Then, use the filepath of the newly created file for running calculation.
    A more robust solution to this problem is going to be implemented.


- The content of the config file can be accessed as a dictionary using the `as_dictionary` property:

.. code-block:: python

    config_dict = config_file.as_dictionary
