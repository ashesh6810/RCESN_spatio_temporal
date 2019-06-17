These codes are supplements to the paper titled "Data-driven prediction of a multi-scale Lorenz 96 chaotic system using a hierarchy of deep learning methods: Reservoir computing, ANN, and RNN-LSTM"

They contain versions of the Echo State Network, The ANN, and the LSTM. It also contains a data set of the normalized Lorenz96 eqquations integrated upto 1M time steps.

Alternatively you can use the lorenz solver which is provided as a .m file to integrate the system to as many time steps as you want

For any questions, reproducibility issues or concerns about the initial conditons to choose, email me at akc6@rice.edu/ashesh6810@gmail.com


Please note that apart from the ESN, which is implemented from scratch the other codes are supported by Keras on a Tensorflow backend. Depending on the Keras version, you may need to change "epochs" to "nb_epoch" or the other way around in the model.fit() API in Keras.

As always, if you use it, please fork it.   
