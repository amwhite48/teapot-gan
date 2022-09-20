"""teapot_dataset dataset."""
import tensorflow as tf
import tensorflow_datasets as tfds
from pathlib import Path
import numpy as np
import random

# TODO(teapot_dataset): Markdown description  that will appear on the catalog page.
_DESCRIPTION = """
Description is **formatted** as markdown.

It should also contain any processing which has been applied (if any),
(e.g. corrupted example skipped, images cropped,...):
"""


class TeapotDataset(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for teapot_dataset dataset."""

  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
  }

  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""
    return tfds.core.DatasetInfo(
        builder=self,
        description=_DESCRIPTION,
        features=tfds.features.FeaturesDict({
            # These are the features of your dataset like images, labels ...
            'voxels': tfds.features.Tensor(shape=(16, 16, 16), dtype=tf.float64),
        }),
        supervised_keys=('None', 'None'),  # Set to `None` to disable
    )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Returns SplitGenerators."""
    p = Path('..')
    return {
        'train': self._generate_examples(p),
    }

  def _generate_examples(self, path):
    """Yields examples."""
    # Yields (key, example) tuples from the dataset
    for f in path.glob('*.npy'):
      image_id = random.getrandbits(256)
      yield image_id, {
          'voxels': np.load(f),
      }
