# ShallowSeismicImaging

Tools to image the shallow seismic structure, above 10 km, based on the ZH ratio measured from the ambient seismic noise, and P polarization from tele-seismic P wave

<!-- ![poolagency logo](http://dashboard.tritontracking.com:5000/static/admin/resources/images/triton-logo.png) -->

## Installation
First clone this repository or download code on machine where you would like to setup.       

#### - Clone the repo
  
```
$ git clone https://github.com/SeisPider/ShallowSeismicImaging.git
$ cd ShallowSeismicImaging
```
 
#### - Install Python

[Windows](http://timmyreilly.azurewebsites.net/python-flask-windows-development-environment-setup/),[Mac](http://docs.python-guide.org/en/latest/starting/install/osx/),[Linux](https://docs.aws.amazon.com/cli/latest/userguide/awscli-install-linux-python.html)

### - Instal SAC
[SAC](https://seiscode.iris.washington.edu/projects/sac/wiki/Binary_Installation)

#### - Install requirements.txt 
 
```
$ pip install -r requirements.txt
```

Above command will install all the dependencies of project.



## Folder structure

```shell
.
├── LICENSE
├── Noise.ZHratio.Measurement             # Measurment technique to obtain ZH ratio from ambient seismic noise
│   ├── Measurement                       # Measurement of ZH ratio
│   │   ├── Step1.Measurement.py         
│   │   ├── Step2.Constrain.moveavg.py
│   │   ├── Step3.Decomposition.py
│   │   ├── __pycache__
│   │   ├── info                          # Info. of seismic stations to measure
│   │   ├── measured.new3.Verification    # Verification for measurement result
│   │   ├── run.test.sh
│   │   └── utils.py                    
│   ├── data                              # Seismic wvaeform stored by date, with instrumental response removed to velocity
│   │   └── getData.py
│   └── src                               # Source code for ZH ratio measurement
│       ├── Noise.py
│       ├── __init__.py
│       ├── __pycache__
│       └── utils.py
├── Polarization.Analysis                 # Measurement and inverison of P wave polarization
│   ├── analysis
│   │   ├── MOD.Verification
│   │   ├── POL.Verification
│   │   ├── Step1.Ppicks.py               # Automatic pick P arrival
│   │   ├── Step2.MeasureP.py             # Measure P apparent polariztaion angle
│   │   ├── Step3.PureInvertS.py          # Inversion for the Near-surface Vs 
│   │   ├── info                          # Info. of seismic stations to measure P polarization 
│   │   └── src                           # Source code for P polarization analysis
│   └── data                              # Seismic wvaeform stored by event, with instrumental response removed to velocity
│       └── getData.py
├── README.md
└── requirements.txt
   
```

## Data preprocessing


**Continous waveform for ZH ratio measurements:** decimate to 1Hz, remove instrumental response to velocity, badnpass filter and cut into daily segments.

> :warning: **Don't change the waveform phase:** Bandpass filter first and then decimate the data.



**Earthquake data for P polarization analysis:** remove instrumental response to velocity and badnpass filter 


## Support

If you face any problem or issue in configuration or usage of poolagency  project as per the instruction documented above, Please feel free to communicate with ShallowSeismicImaging Development Team.

## Reference

1. Xiao X, Cheng S, Wu J, et al. Shallow seismic structure beneath the continental China revealed by P-wave polarization, Rayleigh wave ellipticity and receiver function[J]. Geophysical Journal International, 2021, 225(2): 998-1019.



