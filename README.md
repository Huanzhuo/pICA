[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

# Progressive ICA (pICA)

## Table of Contents
- [Progressive ICA (pICA)](#progressive-ica-pica)
  - [Table of Contents](#table-of-contents)
  - [Description](#description)
  - [Requirements](#requirements)
  - [Usage](#usage)
  - [Citation](#citation)
  - [About Us](#about-us)
  - [License](#license)
  - [Todo](#todo)


## Description


## Requirements

This project uses Python. Following packages are required:
- numpy
- scipy
- museval
- progressbar2
- ffmpeg
- librosa

These could be installed by `conda install numpy scipy museval progressbar2 ffmpeg` and `conda install -c conda-forge librosa`, if the environment is managed by Anaconda.

## Usage
Testing setup can be done in `pybss_example.py`. Test results, including Separation Accuracy and Separation Time, are stored as ***.csv*** in the folder `measurement/`.

## Citation

```
@INPROCEEDINGS{Wu2112:Network,
  AUTHOR="Huanzhuo Wu and Yunbin Shen and Xun Xiao and Artur Hecker and Frank H.P. Fitzek",
  TITLE="{In-Network} Processing Acoustic Data for Anomaly Detection in Smart Factory",
  BOOKTITLE="2021 IEEE Global Communications Conference: IoT and Sensor Networks (Globecom2021 IoTSN)",
  ADDRESS="Madrid, Spain",
  MONTH=dec,
  YEAR=2021
}
```

## About Us

We are researchers at the Deutsche Telekom Chair of Communication Networks (ComNets) at TU Dresden, Germany. Our focus is on in-network computing.

* **Huanzhuo Wu** - huanzhuo.wu@tu-dresden.de or wuhuanzhuo@gmail.com
* **Yunbin Shen** - yunbin.shen@mailbox.tu-dresden.de or shenyunbin@outlook.com

## License

This project is licensed under the [MIT license](./LICENSE).

## Todo
1. Add test data set.
2. Merge version 1.0 with interval extration to the branch pICA-v1.0
