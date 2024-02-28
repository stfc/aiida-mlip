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

    model = ModelData.local_file('/path/to/file', filename='model', architecture='mace')

- To download a file and save it as a `ModelData` object:

.. code-block:: python

    model = ModelData.download('http://yoururl.test/model', architecture='mace', filename='model', cache_dir='/home/mlip/', force_download=False)
