import nussl
import matplotlib.pyplot as plt


def get_dataset():
    '''
       Download the dataset and get the training and validation splits.
    '''
    # Run this command to download 7-second clips from MUSDB18
    musdb = nussl.datasets.MUSDB18(download=True)

    musdb_train = nussl.datasets.MUSDB18(subsets=['train'])
    musdb_validation = nussl.datasets.MUSDB18(subsets=['test'])

    return musdb_train, musdb_validation

def show_sources(dataset, idx):
    '''
       Interesting waveplots for given idx from train/validation dataset. 
    '''
    item = dataset[idx]
    sources = item['sources']
    if isinstance(sources, list):
        sources = {f'Source {i}': s for i, s in enumerate(sources)}
    plt.figure(figsize=(20, 10))
    plt.subplot(211)
    nussl.core.utils.visualize_sources_as_waveform(sources)
    plt.subplot(212)
    nussl.core.utils.visualize_sources_as_masks(sources, db_cutoff=-80)
    plt.tight_layout()
    plt.show()