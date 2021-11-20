#!/usr/bin/env shell
python Step1.Measurement.py ./info/test.net
python Step2.Constrain.moveavg.py ./info/test.net 1
python Step3.Decomposition.py ./info/test.net 1
