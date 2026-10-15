# 官方数据放置位置

本次实际使用第三方 GitHub 镜像的 Parquet 文件：`app_train.parquet`、`bureau.parquet`，来源与哈希记录在 `source.json`，可运行 `../../src/prepare_data.py` 重新获取。以下为可选的官方 CSV 获取方法。

来源：https://www.kaggle.com/competitions/home-credit-default-risk/data

请登录 Kaggle，阅读并接受该竞赛适用的数据使用条件，然后下载：

- `application_train.csv`
- `bureau.csv`
- `HomeCredit_columns_description.csv`

可以直接放置以上三个文件，也可以把包含它们的官方 ZIP 放在本目录。程序仅提取这三个成员，不解压其他文件。不使用比赛的 application_test.csv（没有公开真实标签）。

不要在此处存储账号、密码或 API 密钥。原始数据不随报告重新分发。程序会记录文件 SHA-256 和来源地址；哈希用于识别本次输入，并非由 Kaggle 签名的真实性证明。
