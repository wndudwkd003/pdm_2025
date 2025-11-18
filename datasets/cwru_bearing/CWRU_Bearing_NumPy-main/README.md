# CWRU Bearing Dataset in .npz format

The present repository presents the CWRU Bearing dataset in .npz format, corrected and reduced as far as some metadata are concerned. All credits for the dataset belong to CWRU.

## Dataset Overview

The CWRU Bearing dataset is an open-source dataset provided [here](https://engineering.case.edu/bearingdatacenter) by the Case School of Engineering of the Case Western Reserve University. The data correspond to time-series measured at locations near to and remote from motor bearings of a 2 HP Reliance Electric motor. As far as the experimental procedure is concerned, the CWRU webpage states:

> Motor bearings were seeded with faults using electro-discharge machining (EDM). Faults ranging from 0.007 inches in diameter to 0.040 inches in diameter were introduced separately at the inner raceway, rolling element (i.e. ball) and outer raceway. Faulted bearings were reinstalled into the test motor and vibration data was recorded for motor loads of 0 to 3 horsepower (motor speeds of 1797 to 1720 RPM).

A more detailed presentation of the methodology can be found [here](https://engineering.case.edu/bearingdatacenter/apparatus-and-procedures).

## Motivation

The present repository was created for two main reasons:

1. The original dataset is given in .mat format, as MATLAB was (and still is, to some extent) the mainstream tool for data analysis in Engineering problems. Nonetheless, with the advances that Deep Learning has seen in the past decade, this dataset is being widely used to train, evaluate and deploy DL models. The mainstream frameworks for such tasks are written in Python, so converting the files to .npz format allows DL researchers and enthusiasts to easily and quickly load and subsequently convert them to tensor objects and feed them to NNs or other models.

2. The original files contain some inconsistencies and (perhaps) redundancies when it comes to their metadata (see [Changes](#changes) for more information). The version presented here contains only the time-series data necessary for analysis and DL.

## Data Files

The original data files are in .mat format and are split into four different "families", depending on the motor's load and by extension its speed in RPM: 1797 (Load: 0 HP), 1772 (Load: 1 HP), 1750 (Load: 2 HP) and 1730 (Load: 3 HP). For each RPM value, the data are split in three main categories: Normal Baseline Data, containing time-series for normal bearings, Drive End (DE) Bearing Fault Data, containing time-series for bearings with single-point drive end defects and Fan-End (FE) Bearing Fault Data, containing time-series for bearings with fan end defects. For the DE case, data have been collected with two different frequencies, namely 12 kHz and 48 kHz, while FE data were collected at 12k samples/second.

As far as the faulty data are concerned, the files are further split into categories based on:

1. the fault diameter, which can be 0.007", 0.014", 0.021" or 0.028" and
2. the type of fault, depending on whether it was introduced in the inner raceway (IR), the rolling element (i.e. ball (B)) or the outer raceway (OR).

Specifically for the OR faults, a further differentiation is being made with regards to the position relative to the Load Zone, which is centered at 6:00. Based on this, a time-series can correspond to an OR Centered fault (at 6:00), an OR Orthogonal fault (at 3:00) and an OR Opposite fault (at 12:00).

Each .mat file may contain one or more time-series related to accelerometer data. The time-series can correspond to DE accelerometer data, FE accelerometer data or base (BA) accelerometer data. Additionally, each .mat file has a unique identifier which can be found as the last key of the .mat files' metadata. Its format is `X___`, where `___` is a 3-digit code. For our purposes, we will use the `RPM_Fault_Diameter_End` format to identify files that contain anomalies, where:

* `RPM` identifies the family in terms of RPM,
* `Fault` identifies the anomaly type of the file's time-series (can be `IR`, `B`, `OR@6`, `OR@3` or `OR@12`),
* `Diameter` identifies the fault's diameter in milli-inches (can be `7`, `14`, `21` or `28`) and
* `End` identifies the location and can be either `FE` or `DE12` / `DE48`, depending on the sampling rate (12 kHz and 48 kHz, respectively).

When it comes to baseline data, we will refer to them simply as `RPM_Normal`. Note that the `X___` format is used in the [Changes](#changes) section to highlight the issues with the original .mat files.

## Changes

1. The convention presented in the previous section has been followed to name the .npz files.
2. All .mat files' metadata have been removed. Besides, there were inconsistencies between metadata (e.g. some IR files contained an `'i'` key, while others did not). Note that this means that there is no retained information as far as the `X___` naming convention is concerned.
3. The .npz files only contain time-series data. Their keys can be either DE, FE or BA, based on the type of accelerometer data they contain. Note that the number of time-series varies between different files (some may contain only 1, while others contain 3).
4. The `1750_Normal.mat` file contains two sets of DE/FE time-series. Upon inspection, it was found that one of the pairs (the two `X098` time-series) is identical to the `1772_Normal.mat` file's time-series. For this reason, the redundancy was removed.
5. The `1730_IR_21_DE48` file contains two sets of DE/FE time-series. Upon inspection, it was found that one of the pairs (the two `X215` time-series) is identical to the `1750_IR_21_DE48` file's time-series. For this reason, the redundancy was removed.
6. The `1772_IR_14_DE48` file contains a pair of DE/FE time-series which is probably the correct one, but also contains another single DE time-series. This other time-series is identified as `X217`, but bears no resemblance to the two `X217` time-series found in the `1730_IR_21_DE48` file. Additionally, there was no `X217RPM` key in the file. For these reasons the `X217` time-series was removed.

Based on these, the "Contents" table that can be found [here](/Contents.md) contains the list of all .npz files, as well as the types of time-series they contain (DE, FE and/or BA) and the number of entries.

## Loading Files

To load any .npz datafile, use the `numpy.load()` function, the documentation of which can be found [here](https://numpy.org/doc/stable/reference/generated/numpy.load.html). Note that the keys of the .npz file's arrays are DE, FE and BA, as long as the corresponding time-series exist for the specified file (to confirm which time-series exist for which files, you can read the "Contents" table [here](/Contents.md)).

## Attribution

All credits for the dataset belong to CWRU. This corrected and reduced version of the dataset was developed for the purposes of the following publication:

```
@article{s24165310,
  author = {Rigas, Spyros and Tzouveli, Paraskevi and Kollias, Stefanos},
  title = {An End-to-End Deep Learning Framework for Fault Detection in Marine Machinery},
  journal = {Sensors},
  volume = {24},
  year = {2024},
  number = {16},
  doi = {10.3390/s24165310}
}
```

Nonetheless, citing this paper is not required for researchers who used this version of the dataset.
