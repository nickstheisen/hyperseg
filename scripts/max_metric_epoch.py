#!/usr/bin/env python
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # turn off annoying tf info messages
from tbparse import SummaryReader
from argparse import ArgumentParser

def get_maxmetric_epoch(logdir, metric):
    print("\n\nReading event logs...")
    reader = SummaryReader(logdir)
    print("Done!")

    df = reader.scalars
    filtered = df[df['tag'] == metric].reset_index()
    max_epoch = filtered['value'].idxmax()
    max_val = filtered.loc[max_epoch]['value'] 
    return (max_epoch, max_val)


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('logdir', type=str,
        help='Path to log-directory for relevant train-run.')
    parser.add_argument('metric_name', type=str,
        help='Name of relevant metric.')
    args = parser.parse_args()
    max_epoch, max_val = get_maxmetric_epoch(args.logdir, args.metric_name)
    print(f'Max. value for `{args.metric_name}` is {max_val} in epoch {max_epoch}')
