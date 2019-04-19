!!! Do note that running the full notebooks might take in excess on 1 hour !!!



If you have Conda (Python 2, 64 bits) you can simply run the ipynb notebooks with Jupyter, just make sure that the data folder is extracted at the same location as the notebooks, and the following packages are installed:

pandas
graphviz
pygraphviz
pydotplus



For running the code in the IPython notebooks, I used Jupyter, after installing Anaconda2, v5.2, with Python 2.7, 64bits.

For installing Anaconda, you can refer to the following address:

https://conda.io/docs/user-guide/install/index.html



On Windows, this should be not much of an issue, but setup on Windows can be a pain, and I recommend just using a Linux machine or VM.

In particular, if you are using Windows, you might have to modify the path to GraphViz on the function fix_paths() in utils.py, and change the following line to suit your environment:

	os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
	
If you are using Windows, you might also need to follow the instructions at this link:

https://stackoverflow.com/questions/40809758/howto-install-pygraphviz-on-windows-10-64bit



If you value your time and do not use Windows, then keep reading.

If you wish to use the Ubuntu VM that I use, please refer to the following link:

https://ros-industrial.github.io/industrial_training/_source/setup/PC-Setup---ROS-Kinetic.html

And download (and import) the ROS kinetic OVA, the download link for the OVA is:

http://aeswiki.datasys.swri.edu/vm/ROSI_Training_Kinetic_latest.ova

You will have to install conda, refer to the top link (if you wish to use the VM). If you already have a version of conda that matches my setup, then keep reading.

If you are already using conda, it should be simple to just create a virtual environment, but I cannot provide much guidance on that, since I usually just change all packages on the go if there is a versioning issue.

On Linux, setup should be simple, the following packages need to be installed:

pandas
graphviz
pygraphviz
pydotplus

You can run the following on the terminal:

    conda install pandas
    conda install -c anaconda graphviz
    pip install graphviz
    conda install python-graphviz
    pip install pydotplus

Then, open the following two notebooks using Jupyter, on the terminal, run:

    jupyter notebook

And finally navigate to the following notebooks, and open them:

spam_analysis.ipynb
digits_analysis.ipynb

Just make sure that the data folder is at the same level that the notebook.

Also keep in mind that some functions from the utils.py file are imported as if it was a module. I found surprisingly hard to do so from PyCharm, but never had an issue with Jupyter.



!!! Do note that running the full notebooks might take in excess on 1 hour !!!