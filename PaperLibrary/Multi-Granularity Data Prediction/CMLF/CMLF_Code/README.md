## CMLF: Stock Trend Prediction with Multi-granularity Data: A Contrastive Learning Approach with Adaptive Fusion

> This directory provides the code for paper "Stock Trend Prediction with Multi-granularity Data: A Contrastive Learning Approach with Adaptive Fusion." [**Ref. Code**](https://github.com/CMLF-git-dev/CMLF).

```python
CMLF_Code/
├── mg_datasets # Codes of the Multi-granularity datasets.
    ├── datasets_preprocess # Codes for preprocessing datasets.
    ├── loaders # Data loader for train.
├── models  # The CMLF models.
    ├── contrastive_all_2_encoder.py # The pre-train stage: Contrastive Mechanisms.

└── utils.py  # Some util functions.
```

## Dataset Acquisition

This work collected 3 stock (CSI300, CSI800 and NASDAQ100) datasets from [**`Qlib`**](https://github.com/microsoft/qlib) , an AI-oriented quantitative investment platform. I use the **CSI300** as an example in this directory. There are 5 commonly used statistics extracted as features for **CSI300** datasets, including the **highest price**, the **opening price**, the **lowest price**, the **closing price** and **trading volume**. Because the `Qlib` doesn't provide the **volume-weighted average price** directly, I don't use this feature while changing the label to $𝑦 = 𝑝_{𝑇+2} /𝑝_{𝑇+1} − 1$, where $𝑝_{t}$ is no longer the volume-weighted average price at day 𝑡 but the $(p_{open} + p_{close})/2$.

> Installing `Qlib` on Mac (M2) might be difficult, so I only suggest you install it by `Download => python setup.py install`. Although it is not the officially recommended method, it is the only one that works. You will definitely encounter problems with dependency modules, don't worry, just install them one by one using `pip`. Because this study  just use the `Qlib` interface for data download and preprocessing.

After installing the `Qlib` you need to **DOWNLOAD** the raw stock datasets by:

- **Way 1. Get with module** (Suggested, in `stock_download.sh`)

  ```shell
  # download 1-day data
  python -m qlib.run.get_data qlib_data --target_dir ~/.qlib/qlib_data/cn_data --region cn
  
  # download 1-min data
  python -m qlib.run.get_data qlib_data --target_dir ~/.qlib/qlib_data/cn_data_1min --region cn --interval 1min
  ```

- **Way 2. Get from source**

  ```shell
  # download 1-day data
  python scripts/get_data.py qlib_data --target_dir ~/.qlib/qlib_data/cn_data --region cn
  
  # download 1-min data
  python scripts/get_data.py qlib_data --target_dir ~/.qlib/qlib_data/cn_data_1min --region cn --interval 1min

After downloading the raw data, you will get the following directory structure in `/Users/username/.qlib/qlib_data/`:

```python
qlib_data/
├── cn_data # 1-day data
    ├── calendars
    ├── features
    └── instruments
├── cn_data_1min # 1-min data
    ├── calendars
    ├── features
    └── instruments
```

**ATTENTION**: There will be a lot of difficulties throughout the way to download the data, so if you have any questions, you can communicate them by sending **issues**.



## Data Preprocess and `torch.Dataset`

After downloading the datasets following the **Dataset Acquisition**, data preprocessing is needed to get the structured dataset. I have released preprocess code for datasets, please read them carefully and **follow the guidelines in the top comment rather than running the shell command directly !** I have also released `torch.Dataset` code for datasets.

After downloading the data, I find that the period of dataset is not right ! So I have no other choices but to change the downloading way.



## Model Structure
