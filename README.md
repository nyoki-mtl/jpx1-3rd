# JPXファンダメンタルズ分析チャレンジ3位解法

## 概要

このリポジトリはSIGNATEで開催された[日本取引所グループ ファンダメンタルズチャレンジ](https://signate.jp/competitions/423) における３位入賞コードです。  
7/22現在はコンペで用いられていたデータにアクセスできなくなっているため、再現することは難しいかもしれません。  
SIGNATEのランタイムコンペに参加する際の、ひとつの書き方として参考にしていただければ幸いです。

## 準備

### 実行環境

signateのruntime環境と同一のdockerイメージ（continuumio/anaconda3:2019.03）を基に構築しています。

### ディレクトリ構造

dataディレクトリにjpxのデータを配置してください。  
コンペ時はdata/jpxにコンペページで配布されていたデータ、data/jpx_latestにAPIで取得した最新データをマージしたデータを配置していました。  
すなわちdata/jpx_latestのデータはdata/jpxのデータを包含しているため、jpx_latestのみの準備で大丈夫です。  
特にtrainの再現には2021年の3/26までのデータが準備されていれば十分です。  

```
.
├── Makefile
├── README.md
├── configs
├── data
│  ├── jpx
│  │  ├── stock_fin.csv.gz
│  │  ├── stock_fin_price.csv.gz
│  │  ├── stock_labels.csv.gz
│  │  ├── stock_list.csv.gz
│  │  └── stock_price.csv.gz
│  └── jpx_latest
│      ├── stock_fin.csv.gz
│      ├── stock_labels.csv.gz
│      ├── stock_list.csv.gz
│      └── stock_price.csv.gz
├── docker
├── external
├── setup.py
├── src
├── submit_works
├── tools
└── work_dirs
```

## 実行手順

### train

optunaでハイパラサーチをしたあと、３つのseedで高値予測/安値予測のモデルをlightGBMで学習します。  
すなわち提出用に６つのモデルが用意されます。
学習のログやモデルはwork_dirs/fsub1に出力されます。  

```bash
$ export COMMAND="python tools/train.py fsub1 --data_dir ./data/jpx_latest --train_all --tune_hp"
$ make docker-run
```

### validation

predictor.pyを使ってvalidationを行うコマンドです。  
predictor.pyはsubmit_works/fsub1以下に用意されています。  
具体的にはtrainで生成したwork_dirs/fsub1以下のモデルをsubmit_works/fsub1/modelにコピーして、validationを行います。

```bash
$ export SUBMIT_NAME=fsub1
$ make docker-validate-submit-files 
```

### 提出ファイルの作成

提出ファイルのsubmit.zipを作成するコマンドです。  
validationと同様に、trainで生成したモデルをsubmit_worksにコピーし、動作を確認したあとzipファイルをwork_dirs/fsub1に作成します。

```bash
$ export SUBMIT_NAME=fsub1
$ make docker-create-submit-files
```
