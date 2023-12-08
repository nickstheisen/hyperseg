from hyperseg.datasets import download_dataset
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                    description='Script to download Hyko2-VIS dataset and convert it to hdf5-file.',
             )
    parser.add_argument('savedir')
    args = parser.parse_args()
    filepath = download_dataset(args.savedir, 'HyKo2-VIS_Semantic')

    print(f"Successfully extracted dataset to {filepath}!")
