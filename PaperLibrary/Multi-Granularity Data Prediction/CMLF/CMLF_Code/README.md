## CMLF: Stock Trend Prediction with Multi-granularity Data: A Contrastive Learning Approach with Adaptive Fusion

> This directory provides the code for paper "Stock Trend Prediction with Multi-granularity Data: A Contrastive Learning Approach with Adaptive Fusion." [**Ref. Code**](https://github.com/CMLF-git-dev/CMLF).

```python
CMLF_Code/
├── mg_datasets # Codes of the Multi-granularity datasets.
    ├── datasets_download # Cods for downloading datasets.
        ├── download_daily_stock_dataset.py
        ├── download_hf_stock_dataset.py
├── models  # The CMLF models.
└── utils.py  # Some util functions.
```

## Dataset Acquisition

This work collected 3 stock datasets from [**`Qlib`**](https://github.com/microsoft/qlib) , an AI-oriented quantitative investment platform. I use the **`CSI300`** as an example. There are six commonly used features extracted as features for **`CSI300`** datasets, including the **highest price**, the **opening price**, the **lowest price**, the **closing price**, **volume-weighted average price**, and **trading volume**.

Installing `Qlib` on Mac might be difficult, so I only suggest you install it by `Download => python setup.py install`. Although it is not the officially recommended method, it is the only one that works. You can follow this link to install `Qlib`.

You will definitely encounter problems with dependency modules, don't worry, just install them one by one using `pip`. Because this study  just use the `Qlib` interface for data download and preprocessing.

After install the `Qlib` you need to **DOWNLOAD** the raw data by:

- **Way 1. Get with module** (You can run the command everywhere ! But really slow !)

  ```shell
  # get 1d data
  python -m qlib.run.get_data qlib_data --target_dir ~/.qlib/qlib_data/cn_data --region cn
  
  # get 1min data
  python -m qlib.run.get_data qlib_data --target_dir ~/.qlib/qlib_data/cn_data_1min --region cn --interval 1min
  ```

- **Way 2. Get from source** (Faster, but you should change the path into `Qlib` directory and run the command)

  ```shell
  # get 1d data
  python scripts/get_data.py qlib_data --target_dir ~/.qlib/qlib_data/cn_data --region cn
  
  # get 1min data
  python scripts/get_data.py qlib_data --target_dir ~/.qlib/qlib_data/cn_data_1min --region cn --interval 1min

After downloading the raw data, you will get the following directory structure in `username/.qlib/qlib_data/`:

```python
qlib_data/

```



## Data Preprocess and `torch.Dataset`

Re-Frequency. The data are adjusted for dividends and splits, and normalized by the Z-Score method.



## Training & Prediction



### **Dependencies**

Install packages from `requirements.txt`.  

### **Load Data using qlib**
```linux
$ cd ./load_data
```

#### Download daily data:

```python
$ python load_dataset.py
```
* Change parameter `market` to get data from different dataset: `csi300`, `csi800`, `NASDAQ` etc.

  ##### Data Sample - SH600000 in CSI300

  ![](https://ftp.bmp.ovh/imgs/2021/02/28e2e1b545cf8ffc.png)

  features dimensions = 6 * 20 + 1 = 121

#### Download high-frequency data:

```python
$ python high_freq_resample.py
```

* Change parameter `N` to get data from different frequencies: `15min`, `30min`, `120min` etc.

  ##### Data Sample - SH600000 in CSI300

     ![](https://ftp.bmp.ovh/imgs/2021/02/21213511c92c4c44.png)

  features dimensions = 16 * 6 * 20 + 1 = 1921



### **Framework**

* Pre-training Stage: Contrastive Mechanisms：`./framework/models/contrastive_all_2_encoder.py`
* Adaptive Multi-granularity Feature Fusion: `./framework/models/contrastive_all_2_stage.py`
3. ### **Run**
  ```linux
  $ cd ./framework
  ```

  #### Train `Pre-train` model:

  ```python
  $ python main_contrast.py with config/contrast_all_2_encoder.json model_name=contrastive_all_2_encoder
  ```

  * Add `hyper-param` = {`values`} after `with` or change them in `config/main_model.json`
  * Prediction results of each model are saved as `pred_{model_name}.pkl` in `./out/`.

  #### Train `Adaptive Multi-granularity Feature Fusion` model:

  ```python
  $ python main_contrast_2_stage.py with config/contrast_all_2_stage.json model_name=contrastive_all_2_stage
  ```


  #### Run `Market Trading Simulation`:
  * Prerequisites:   
  	* Server with qlib
  	* Prediction results 
  ```linux
  $ cd ./framework
  ```
  ```python
  $ python trade_sim.py
  ```


### **Records**

Records for each experiment are saved in `./framework/my_runs/`. 
Each record file includes: 

> config.json
* contains the parameter settings and data path.

> cout.txt
* contains the name of dataset, detailed model output, and experiment results.

> pred_{model_name}_{seed}.pkl

  >
	>  * contains the  `score` (model prediction) and `label`

	> run.json
	
	* contains the hash ids of every script used in the experiment. And the source code can be found in `./framework/my_runs/source/`.
