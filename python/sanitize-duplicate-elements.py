#This script does the following:
#   - Locates duplicate images on a given array
#   - Compares 3 arrays containing images looking for duplicates among them
#   - Removes those images so that there is no intersection between those arrays


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
