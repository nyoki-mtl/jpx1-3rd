import shutil
import io
import os
import sys
from argparse import ArgumentParser

import pandas as pd
import subprocess

from src.utils.project_dirs import data_dir, work_dir, submit_dir, tool_dir, root_dir


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('exp_name')
    parser.add_argument('--validate', action='store_true')
    return parser.parse_args()


def main():
    args = parse_args()
    # Dataset directory
    DATASET_DIR = data_dir / 'jpx_latest'
    # Log directory
    LOG_DIR = work_dir / args.exp_name
    # Submit directory
    SUBMIT_DIR = submit_dir / args.exp_name
    SUBMIT_MODEL_DIR = SUBMIT_DIR / 'model'
    # shutil.copy(root_dir / 'docker' / 'requirements.txt', SUBMIT_DIR)
    if not SUBMIT_MODEL_DIR.exists():
        SUBMIT_MODEL_DIR.mkdir()
        src_files = list(LOG_DIR.glob('final_lgb_label*'))
        src_files += [LOG_DIR / 'le_dict.pkl', LOG_DIR / 'normalize_value.csv', tool_dir / 'infos' / 'nikkei225.txt']
        for src_file in src_files:
            shutil.copy(src_file, SUBMIT_MODEL_DIR)

    # Change directory
    os.chdir(SUBMIT_DIR / 'src')
    # Add python path
    sys.path.append(str(SUBMIT_DIR / 'src'))

    from predictor import ScoringService
    inputs = {
        'stock_list': str(DATASET_DIR / 'stock_list.csv.gz'),
        'stock_price': str(DATASET_DIR / 'stock_price.csv.gz'),
        'stock_fin': str(DATASET_DIR / 'stock_fin.csv.gz'),
        'stock_labels': str(DATASET_DIR / 'stock_labels.csv.gz'),
    }
    if args.validate:
        if ScoringService.get_model():
            ScoringService.predict(inputs, check_val_score=True)
        else:
            raise NotImplementedError
    else:
        if ScoringService.get_model():
            result = ScoringService.predict(inputs)
        else:
            raise NotImplementedError

        # Check result
        df = pd.read_csv(io.StringIO(result), header=None)
        print(df.shape)
        print(df.head())

        os.chdir('../')
        subprocess.call('zip -v submit.zip requirements.txt src/*.py model/*', shell=True)

        out_dir = work_dir / args.exp_name
        subprocess.call(f'mv submit.zip {out_dir}', shell=True)


if __name__ == '__main__':
    main()
