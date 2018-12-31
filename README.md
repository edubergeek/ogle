# ogleThe file OGLE-ALL-CEP.txt contains the manifest of all OGLE cepheid variables
The light curves of each object in two filters, I and V, are in the directories of the same name.
Each star by id in OGLE-ALL-CEP.txt has a '.dat' file in I and V dirs with light curve data.

For example:
OGLE-ALL-CEP.txt row 18  is star id=OGLE-LMC-CEP-0010
./I/OGLE-LMC-CEP-0010.dat has the I filter light curve data
./V/OGLE-LMC-CEP-0010.dat has the V filter light curve data

Input layer to RNN
jd is time of measurement (julian date %7.6f)
mi is I magnitude (%2.6f)
ei is error estimate for ei
mv is V magnitude (%2.6f)
ev is error estimate for mv

Transform Input to 3 dimensions and sort by d
d is date
f is filter 0 (I) or 1 (V)
e is error
m is magnitude

ogle2tfr.py is a file walker that converts .dat files into TFRecord format with train/valid/test splits.
plottfr.py visualizes the training examples reading the TFRecord training set.
train.py is the first attempt at training an RNN (GRU) model.
traintfr.py is the RNN using the TFR training data
train-fc.py is an ANN (FC) model
train-cnn.py is a CNN model
periodogram.py performs inference on .dat files (file walker) using a trained model then prints and  plots results.
