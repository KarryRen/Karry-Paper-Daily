# @Time    : 2024/4/10 10:45
# @Author  : Karry Ren

# download the stock dataset

# get 1d data
python -m qlib.run.get_data qlib_data --target_dir ~/.qlib/qlib_data/cn_data --region cn

# get 1min data
python -m qlib.run.get_data qlib_data --target_dir ~/.qlib/qlib_data/cn_data_1min --region cn --interval 1min
