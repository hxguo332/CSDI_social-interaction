# CSDI
This is the github repository for the NeurIPS 2021 paper "[CSDI: Conditional Score-based Diffusion Models for Probabilistic Time Series Imputation](https://arxiv.org/abs/2107.03502)".

## Requirement

Please install the packages in requirements.txt

You may want to delete torch in requirement and install torch manually (check the torch official website for installation).

## Preparation
### Download the healthcare dataset 
```shell
python download.py physio
```
### Download the air quality dataset 
```shell
python download.py pm25
```

### Download the elecricity dataset 
Please put files in [GoogleDrive](https://drive.google.com/drive/folders/1krZQofLdeQrzunuKkLXy8L_kMzQrVFI_?usp=drive_link) to the "data" folder.

## Experiments 

### training and imputation for track dataset

```shell
python exe_simulation.py --config exp_scenmap_nonaug_mix_two_ends_train.yaml --data_length 200
```

### training and imputation for the healthcare dataset
```shell
python exe_physio.py --testmissingratio [missing ratio] --nsample [number of samples]
```

### imputation for the healthcare dataset with pretrained model
```shell
python exe_physio.py --modelfolder pretrained --testmissingratio [missing ratio] --nsample [number of samples]
```

### training and imputation for the healthcare dataset
```shell
python exe_pm25.py --nsample [number of samples]
```

### training and forecasting for the electricity dataset
```shell
python exe_forecasting.py --datatype electricity --nsample [number of samples]
```

### Visualize results
'visualize_examples.ipynb' is a notebook for visualizing results.

## Acknowledgements

A part of the codes is based on [BRITS](https://github.com/caow13/BRITS) and [DiffWave](https://github.com/lmnt-com/diffwave)

## Citation
If you use this code for your research, please cite our paper:

```
@inproceedings{tashiro2021csdi,
  title={CSDI: Conditional Score-based Diffusion Models for Probabilistic Time Series Imputation},
  author={Tashiro, Yusuke and Song, Jiaming and Song, Yang and Ermon, Stefano},
  booktitle={Advances in Neural Information Processing Systems},
  year={2021}
}
```
