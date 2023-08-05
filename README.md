# python dictionary based experiment tracking
Utilize Python dictionary-based configuration files for seamless execution and tracking of experiments. The benefit of using Python dictionaries, as opposed to yaml, json and the like, is that it allows direct inclusion of Python objects in your configuration. This eliminates the need for clumsy alias strings.

# How to use
- All essential functions can be found in ```funcs.py```.
- The 'configs' folder provides a sample master configuration file. This serves as a template, which is populated with various parameters during your grid search.
- ```main.py``` illustrates how to construct and execute a grid search. It triggers a script(```run_NN.py```) which trains a CNN and stores the resulting output and corresponding configuration file.
- Lastly, ```analyze_results.py``` shows a basic example for retrieving grid search results. 
