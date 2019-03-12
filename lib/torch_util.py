from sklearn.model_selection import train_test_split
import numpy as np
import torch

SOS_TOKEN = 0
EOS_TOKEN = 1
PAD_TOKEN = ord('*')
device = torch.device(
    'cuda' if torch.cuda.is_available() else 'cpu'
)

def apply_padding(_str, n_max_char, zero_char='*'):
  """
  apply padding to string until n_max_char fulfilled.
  """
  len_str = len(_str)
  if len_str >= n_max_char:
      return _str[:n_max_char]

  else:
    return _str + ''.join([zero_char for t in range(n_max_char - len_str)])
    
def apply_padding_ord(loi, n_max_char, zero_char=ord('*')):
  """
  apply padding to list of integer, gotten from function ord(char), until n_max_char fulfilled.
  """
  len_str = len(loi)
  if len_str >= n_max_char:
      return loi[:n_max_char]

  else:
      return loi + [zero_char for t in range(n_max_char - len_str)]
    
def replace_punctuations(x, replace_with='*'):
  """
  replace punctuation from string x to a character, by default replaced with *
  """
  return [str(e) if str(e).isalpha() else replace_with for e in x]

def replace_punctuations_ord(x, replace_with=42):
  """
  replace punctuation from list of integer x to a character, by default replaced with * (ordinal number 42)
  """
  return [e if str(e).isalpha() else replace_with for e in x]

def lord(x):
  """make string into list of integer integer"""
  return [SOS_TOKEN] + [ord(x1) for x1 in x] + [EOS_TOKEN]

def tchr(x):
  """
  from list of integer x -> string. Make list of ascii encoded integer to character and combine it all to 1 string
  """
  return ''.join([
    chr(y) for y in x[1:-2] 
    if (y != PAD_TOKEN and y != SOS_TOKEN and y != EOS_TOKEN)
  ])


def encoded_from_sentence(text, max_len):
  """
  make a string to list of integer
  """
  return apply_padding_ord(
    lord(replace_punctuations(text))
    , max_len
  )

def decoded_to_sentence(encoded):
  """
  make list of integer to string
  """
  return tchr(encoded)

def to_torch(x, device=device):
  """
  from numpy to pytorch's tensor. 
  Args
  ----
  x: numpy.array
  """
  return torch.from_numpy(np.array(x.values.tolist())).to(device)

def split_train_test_val(X, y, seed=0, test_size=0.3, val_size=0.3):
  """
    split feature and target into train, test, and validation.
    from 100% data, will be splitted first with test_size. we will take example if test_size = 30%, and val_size = 30%
    train_data = (100-test_size)% * len(X)
    train_data = 70% * len(X)

    intermediary_data = 30* * len(X)

    while the the other 30% remaining will be splitted into two parts:
    test_data = (100-val_size)% * len(intermediary_data)
    test_data = 70% * (30% * len(X))
    test_data = 21% * len(X)

    while validation data will be:
    val_data = len(intermediary_data) - len(test_data)
    val_data = (30% * len(X)) - (21% * len(X))
    val_data = 9% * len(X)


    all of these will be chosen randomly based on seed for reproducible purpose
  """

  np.random.seed(seed)
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
  X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=val_size)

  return X_train, X_test, X_val, y_train, y_test, y_val


def to_list_of_int(string_arr, max_len):
  """
    from string array into list on int. 
  """
  return string_arr.apply(
    replace_punctuations
  ).apply(lord).apply(
    apply_padding_ord
    , args=(max_len,)
  ).apply(list).apply(np.array)