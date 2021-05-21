import nobrainer
import tensorflow as tf

#Load sample Data--- inputs and labels 
csv_of_filepaths = nobrainer.utils.get_data()
filepaths = nobrainer.io.read_csv(csv_of_filepaths)
train_paths = filepaths[:9]
evaluate_paths = filepaths[9:]

#Convert medical images to TFRecords
invalid = nobrainer.io.verify_features_labels(train_paths, num_parallel_calls=2)
assert not invalid
invalid = nobrainer.io.verify_features_labels(evaluate_paths)
assert not invalid

nobrainer.tfrecord.write(
    features_labels=train_paths,
    filename_template='data/data-train_shard-{shard:03d}.tfrec',
    examples_per_shard=3)

nobrainer.tfrecord.write(
    features_labels=evaluate_paths,
    filename_template='data/data-evaluate_shard-{shard:03d}.tfrec',
    examples_per_shard=1)

# Set parameters
n_classes = 1
batch_size = 2
volume_shape = (256, 256, 256)
block_shape = (128, 128, 128)
n_epochs = None
augment = False
shuffle_buffer_size = 10
num_parallel_calls = 2

# Create and Load Datasets for training and validation
dataset_train = nobrainer.dataset.get_dataset(
    file_pattern="data/data-train_shard-*.tfrec",
    n_classes=n_classes,
    batch_size=batch_size,
    volume_shape=volume_shape,
    block_shape=block_shape,
    n_epochs=n_epochs,
    augment=augment,
    shuffle_buffer_size=shuffle_buffer_size,
    num_parallel_calls=num_parallel_calls,
)

dataset_evaluate = nobrainer.dataset.get_dataset(
    file_pattern="data/data-evaluate_shard-*.tfrec",
    n_classes=n_classes,
    batch_size=batch_size,
    volume_shape=volume_shape,
    block_shape=block_shape,
    n_epochs=1,
    augment=False,
    shuffle_buffer_size=None,
    num_parallel_calls=1,
)
