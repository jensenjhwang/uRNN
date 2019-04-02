CS 224N: Toy Examples
===
This folder contains our implementation of the uRNN for the various toy tasks.

## Data Generation
`run_generator.py` contains the script that generates data, which is formatted as numbers delimited by spaces.

Unfortunately, the data generation parameters are hardcoded in `run_generator.py` itself. However, one set of data is provided in `./data/` for length 100 strings with 100 different possible numbers. The task is to find the 10-th largest number within the sequence.

## Training
Commands:

Standard RNN: `python run.py train`
uRNN: `python run.py train --ortho`

On inputs with more types of numbers ("characters") in the sequence, you will have to manually set the `--n-char` option: e.g.
```python run.py train --ortho --n-char=200```

## All training options
```
Options:
    -h --help               Show this screen.
    --cuda                  If using gpu
    --ortho                 Orthogonal RNN (vs normal RNN)
    --seed=<int>            Seed values for randomness (for reproducibility) [Default: 0]
    --save-path=<str>       Relative path to save model to [Default: /models/saved_model.pt]
    --optimizer=<str>       Optimizer [Default: adam]
    --data-dir=<str>        Folder for data [Default: data]
    --reduced               Load reduced data set

    --batch-size=<int>      Batch-size [Default: 100]
    --train-data=<int>      Number of training samples [Default: 100000]
    --dev-data=<int>        Number of validation samples [Default: 10000]
    --long-val              Long validation (test for generalization) (INVALID)
    --print-matrices        Prints matrices during validation

    --n-char=<int>          Number of characters [Default: 100]
    --embed-size=<in>       Embedding dimensions [Default: 30]
    --hidden-size=<int>     Hidden dimensions [Default: 20]

    --lr=<float>            Learning rate [Default: 0.00005]
    --lr-decay=<float>      lr decay factor [Default: 0.5]

    --max-epoch=<int>       max number of epochs [Default: 200]
    --log-every=<int>       logs every N iterations [Default: 100]
    --val-every=<int>       validates every N iterations against test set [Default: 1000]
    --patience=<int>        lr decay if no best in P validations [Default: 5]
    --decay-limit=<int>     number of times to decay [Default: 5]

    --plot-graphs           plot graphs?

    --weightdrop=<float>    weight drop probability
```