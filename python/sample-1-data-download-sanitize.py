# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import tarfile
from IPython.display import display, Image
from scipy import ndimage
from sklearn.linear_model import LogisticRegression
from six.moves.urllib.request import urlretrieve
from six.moves import cPickle as pickle

# Config the matplotlib backend as plotting inline in IPython
%matplotlib inline




url = 'http://commondatastorage.googleapis.com/books1000/'
last_percent_reported = None

def download_progress_hook(count, blockSize, totalSize):
  """A hook to report the progress of a download. This is mostly intended for users with
  slow internet connections. Reports every 5% change in download progress.
  """
  global last_percent_reported
  percent = int(count * blockSize * 100 / totalSize)

  if last_percent_reported != percent:
    if percent % 5 == 0:
      sys.stdout.write("%s%%" % percent)
      sys.stdout.flush()
    else:
      sys.stdout.write(".")
      sys.stdout.flush()

    last_percent_reported = percent

def maybe_download(filename, expected_bytes, force=False):
  """Download a file if not present, and make sure it's the right size."""
  if force or not os.path.exists(filename):
    print('Attempting to download:', filename)
    filename, _ = urlretrieve(url + filename, filename, reporthook=download_progress_hook)
    print('\nDownload Complete!')
  statinfo = os.stat(filename)
  if statinfo.st_size == expected_bytes:
    print('Found and verified', filename)
  else:
    raise Exception(
      'Failed to verify ' + filename + '. Can you get to it with a browser?')
  return filename

train_filename = maybe_download('notMNIST_large.tar.gz', 247336696)
test_filename = maybe_download('notMNIST_small.tar.gz', 8458043)




num_classes = 10
np.random.seed(133)

def maybe_extract(filename, force=False):
  root = os.path.splitext(os.path.splitext(filename)[0])[0]  # remove .tar.gz
  if os.path.isdir(root) and not force:
    # You may override by setting force=True.
    print('%s already present - Skipping extraction of %s.' % (root, filename))
  else:
    print('Extracting data for %s. This may take a while. Please wait.' % root)
    tar = tarfile.open(filename)
    sys.stdout.flush()
    tar.extractall()
    tar.close()
  data_folders = [
    os.path.join(root, d) for d in sorted(os.listdir(root))
    if os.path.isdir(os.path.join(root, d))]
  if len(data_folders) != num_classes:
    raise Exception(
      'Expected %d folders, one per class. Found %d instead.' % (
        num_classes, len(data_folders)))
  print(data_folders)
  return data_folders

train_folders = maybe_extract(train_filename)
test_folders = maybe_extract(test_filename)






image_size = 28  # Pixel width and height.
pixel_depth = 255.0  # Number of levels per pixel.

def load_letter(folder, min_num_images):
  """Load the data for a single letter label."""
  image_files = os.listdir(folder)
  dataset = np.ndarray(shape=(len(image_files), image_size, image_size),
                         dtype=np.float32)
  print(folder)
  num_images = 0
  for image in image_files:
    image_file = os.path.join(folder, image)
    try:
      image_data = (ndimage.imread(image_file).astype(float) -
                    pixel_depth / 2) / pixel_depth
      if image_data.shape != (image_size, image_size):
        raise Exception('Unexpected image shape: %s' % str(image_data.shape))
      dataset[num_images, :, :] = image_data
      num_images = num_images + 1
    except IOError as e:
      print('Could not read:', image_file, ':', e, '- it\'s ok, skipping.')

  dataset = dataset[0:num_images, :, :]
  if num_images < min_num_images:
    raise Exception('Many fewer images than expected: %d < %d' %
                    (num_images, min_num_images))

  print('Full dataset tensor:', dataset.shape)
  print('Mean:', np.mean(dataset))
  print('Standard deviation:', np.std(dataset))
  return dataset

def maybe_pickle(data_folders, min_num_images_per_class, force=False):
  dataset_names = []
  for folder in data_folders:
    set_filename = folder + '.pickle'
    dataset_names.append(set_filename)
    if os.path.exists(set_filename) and not force:
      # You may override by setting force=True.
      print('%s already present - Skipping pickling.' % set_filename)
    else:
      print('Pickling %s.' % set_filename)
      dataset = load_letter(folder, min_num_images_per_class)
      try:
        with open(set_filename, 'wb') as f:
          pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)
      except Exception as e:
        print('Unable to save data to', set_filename, ':', e)

  return dataset_names

train_datasets = maybe_pickle(train_folders, 45000)
test_datasets = maybe_pickle(test_folders, 1800)






def make_arrays(nb_rows, img_size):
  if nb_rows:
    dataset = np.ndarray((nb_rows, img_size, img_size), dtype=np.float32)
    labels = np.ndarray(nb_rows, dtype=np.int32)
  else:
    dataset, labels = None, None
  return dataset, labels

def merge_datasets(pickle_files, train_size, valid_size=0):
  num_classes = len(pickle_files)
  valid_dataset, valid_labels = make_arrays(valid_size, image_size)
  train_dataset, train_labels = make_arrays(train_size, image_size)
  vsize_per_class = valid_size // num_classes
  tsize_per_class = train_size // num_classes

  start_v, start_t = 0, 0
  end_v, end_t = vsize_per_class, tsize_per_class
  end_l = vsize_per_class+tsize_per_class
  for label, pickle_file in enumerate(pickle_files):
    try:
      with open(pickle_file, 'rb') as f:
        letter_set = pickle.load(f)
        # let's shuffle the letters to have random validation and training set
        np.random.shuffle(letter_set)
        if valid_dataset is not None:
          valid_letter = letter_set[:vsize_per_class, :, :]
          valid_dataset[start_v:end_v, :, :] = valid_letter
          valid_labels[start_v:end_v] = label
          start_v += vsize_per_class
          end_v += vsize_per_class

        train_letter = letter_set[vsize_per_class:end_l, :, :]
        train_dataset[start_t:end_t, :, :] = train_letter
        train_labels[start_t:end_t] = label
        start_t += tsize_per_class
        end_t += tsize_per_class
    except Exception as e:
      print('Unable to process data from', pickle_file, ':', e)
      raise

  return valid_dataset, valid_labels, train_dataset, train_labels


train_size = 200000
valid_size = 10000
test_size = 10000

valid_dataset, valid_labels, train_dataset, train_labels = merge_datasets(
  train_datasets, train_size, valid_size)
_, _, test_dataset, test_labels = merge_datasets(test_datasets, test_size)

print('Training:', train_dataset.shape, train_labels.shape)
print('Validation:', valid_dataset.shape, valid_labels.shape)
print('Testing:', test_dataset.shape, test_labels.shape)







def randomize(dataset, labels):
  permutation = np.random.permutation(labels.shape[0])
  shuffled_dataset = dataset[permutation,:,:]
  shuffled_labels = labels[permutation]
  return shuffled_dataset, shuffled_labels
train_dataset, train_labels = randomize(train_dataset, train_labels)
test_dataset, test_labels = randomize(test_dataset, test_labels)
valid_dataset, valid_labels = randomize(valid_dataset, valid_labels)







pickle_file = 'notMNIST.pickle'

try:
  f = open(pickle_file, 'wb')
  save = {
    'train_dataset': train_dataset,
    'train_labels': train_labels,
    'valid_dataset': valid_dataset,
    'valid_labels': valid_labels,
    'test_dataset': test_dataset,
    'test_labels': test_labels,
    }
  pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
  f.close()
except Exception as e:
  print('Unable to save data to', pickle_file, ':', e)
  raise




statinfo = os.stat(pickle_file)
print('Compressed pickle size:', statinfo.st_size)




train_r = train_dataset.reshape(train_dataset.shape[0],-1)
print(np.shape(train_r))
train_idx = np.lexsort(train_r.T)
print(np.shape(train_idx))
train_dataset_sanitized = train_dataset[train_idx][np.append(True,(np.diff(train_r[train_idx],axis=0)!=0).any(1))]
print(np.shape(train_dataset_sanitized))
train_labels_sanitized = train_labels[train_idx][np.append(True,(np.diff(train_r[train_idx],axis=0)!=0).any(1))]
print(np.shape(train_labels_sanitized))

valid_r = valid_dataset.reshape(valid_dataset.shape[0],-1)
valid_idx = np.lexsort(valid_r.T)
valid_dataset_sanitized = valid_dataset[valid_idx][np.append(True,(np.diff(valid_r[valid_idx],axis=0)!=0).any(1))]
valid_labels_sanitized = valid_labels[valid_idx][np.append(True,(np.diff(valid_r[valid_idx],axis=0)!=0).any(1))]

test_r = test_dataset.reshape(test_dataset.shape[0],-1)
test_idx = np.lexsort(test_r.T)
test_dataset_sanitized = test_dataset[test_idx][np.append(True,(np.diff(test_r[test_idx],axis=0)!=0).any(1))]
test_labels_sanitized = test_labels[test_idx][np.append(True,(np.diff(test_r[test_idx],axis=0)!=0).any(1))]

del train_r, valid_r, test_r

print('Training dataset has', train_dataset_sanitized.shape[0],'unique images.')
print('Validation dataset has', valid_dataset_sanitized.shape[0],'unique images.')
print('Test dataset has', test_dataset_sanitized.shape[0],'unique images.\n')

train_r = train_dataset_sanitized.reshape(train_dataset_sanitized.shape[0],-1)
valid_r = valid_dataset_sanitized.reshape(valid_dataset_sanitized.shape[0],-1)
test_r = test_dataset_sanitized.reshape(test_dataset_sanitized.shape[0],-1)

valid_dup = []
test_dup = []

train_r = {tuple(row):i for i,row in enumerate(train_r)}

for i,row in enumerate(valid_r):
    if tuple(row) in train_r:
        valid_dup.append(i)

for i,row in enumerate(test_r):
    if tuple(row) in train_r:
        test_dup.append(i)

print('Validation dataset has', len(valid_dup), 'duplicate images to training dataset.')
print('Test dataset has', len(test_dup), 'duplicate images to training dataset.\n')

valid_dataset_sanitized = np.delete(valid_dataset_sanitized, np.asarray(valid_dup), 0)
valid_labels_sanitized = np.delete(valid_labels_sanitized, np.asarray(valid_dup), 0)
test_dataset_sanitized = np.delete(test_dataset_sanitized, np.asarray(test_dup), 0)
test_labels_sanitized = np.delete(test_labels_sanitized, np.asarray(test_dup), 0)

print('Sanitized train dataset has', train_dataset_sanitized.shape[0],'images.')
print('Sanitized validation dataset has', valid_dataset_sanitized.shape[0],'images.')
print('Sanitized test dataset has', test_dataset_sanitized.shape[0],'images.')





pickle_file = 'notMNIST_sanitized.pickle'

try:
  f = open(pickle_file, 'wb')
  save = {
    'train_dataset': train_dataset_sanitized,
    'train_labels': train_labels_sanitized,
    'valid_dataset': valid_dataset_sanitized,
    'valid_labels': valid_labels_sanitized,
    'test_dataset': test_dataset_sanitized,
    'test_labels': test_labels_sanitized,
    }
  pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
  f.close()
  print('Sanitized data saved to', pickle_file);
except Exception as e:
  print('Unable to save data to', pickle_file, ':', e)
  raise




  from sklearn.metrics import classification_report, confusion_matrix

def train_predict(clf, n_data, train_data, train_label, test_data, test_label):
    clf.fit(train_data[:n_data,:,:].reshape(n_data,-1), train_label[:n_data])
    # Predict
    expected = test_label
    predicted = clf.predict(test_data.reshape(test_data.shape[0],-1))

    # Print Results
    print('Classification Report of',n_data,'training samples:\n', classification_report(expected, predicted))
    #print('Confusion Matrix of',n_data,'training samples:\n', confusion_matrix(expected, predicted))

# Create a Logistic Regression Classifier
clf = LogisticRegression(penalty='l2', tol=0.0001, C=1.0, random_state=133, solver='sag', max_iter=100, multi_class='ovr', verbose=0, n_jobs=4)

print('-------')
print(np.shape(train_dataset))
print(np.shape(train_labels))
print(np.shape(test_dataset))
print(np.shape(test_labels))
print(np.shape(valid_dataset))
print(np.shape(valid_labels))
print('-------_sanitized')
print(np.shape(train_dataset_sanitized))
print(np.shape(train_labels_sanitized))
print(np.shape(test_dataset_sanitized))
print(np.shape(test_labels_sanitized))
print(np.shape(valid_dataset_sanitized))
print(np.shape(valid_labels_sanitized))
print('-------')
