import pathlib
import os
import zipfile
import sys
import tarfile
from urllib.request import urlretrieve

DATA_PATH = pathlib.Path().parent / 'data'


def download(url):
    name = url.split('/')[-1]
    download_path = os.path.join(DATA_PATH, name)

    def _progress(count, block_size, total_size):
        sys.stdout.write('\rDownloading %s %.2f%%' % (name,
                                                      float(count * block_size) / float(total_size) * 100.0))
        sys.stdout.flush()

    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH)

    if not os.path.exists(download_path):
        urlretrieve(url, download_path, reporthook=_progress)
        sys.stdout.write('\n')

        if name.endswith('.tar.gz'):
            tarfile.open(download_path, 'r:gz').extractall(DATA_PATH)
        elif name.endswith('.zip'):
            zipfile.ZipFile(download_path, 'r').extractall(DATA_PATH)
        elif name.endswith('.gz'):
            pass
        else:
            sys.stderr.write("\nUnknown file type")