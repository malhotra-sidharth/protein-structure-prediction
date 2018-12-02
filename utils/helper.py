import pandas as pd
import numpy as np
import os
from biopandas.pdb import PandasPdb as ppdb


class Extract:
  def __init__(self):
    self.refer_protein = {
      'ALA': 0,
      'GLY': 1,
      'ILE': 2,
      'LEU': 3,
      'PRO': 4,
      'VAL': 5,
      'PHE': 6,
      'TRP': 7,
      'TYR': 8,
      'ASP': 9,
      'GLU': 10,
      'ARG': 11,
      'HIS': 12,
      'LYS': 13,
      'SER': 14,
      'THR': 15,
      'CYS': 16,
      'MET': 17,
      'MSE': 17,
      'MSO': 17,
      'ASN': 18,
      'GLN': 19
    }


  def get_training_data(self, df_list):
    """
    Creates X and Y vector for training neural network
    from the given list of dataframes with a sliding window of
    10
    :param df_list: List of dataframes
    :return: X vector and Y vector <numpy arrays>
    """
    X = []
    Y = []
    for df in df_list:
      # get 10 rows at a time
      for i in range(0, len(df) - 9):
        ip = df[i:i + 10]
        label = ip['helix']
        input_vector = ip.drop(['helix'], axis=1)
        flattened_ip_vector = input_vector.values.flatten()
        X.append(flattened_ip_vector)
        Y.append(label.values)

    X = np.array(X)
    Y = np.array(Y)
    return X, Y


  def get_one_hot_encoding(self, df_list):
    """
    Creates one hot encoding for all the acids in each dataframe
    in the given list of the dataframes
    :param df_list: List of dataframes
    :return: List of dataframes with one hot encoding
    """
    # https://stackoverflow.com/a/37426982
    df2_list = []
    for df in df_list:
      df2 = df.drop(['acids'], axis=1)
      hot_encode = pd.get_dummies(df['acid_num'], dtype=float)
      hot_encode = hot_encode.T.reindex([i for i in range(0, 20)]).T.fillna(0)
      df2 = df2.drop(['acid_num'], axis=1)
      df2 = pd.concat([df2, hot_encode], axis=1)
      df2_list.append(df2)

    return df2_list


  def extract_data_from_pdb(self, dir, save_to_csv=False):
    """
    Extracts data from all the pdb files in the given directory and
    returns a list of dataframes
    :param dir: Name of the directory <string>
    :param save_to_csv: bool, true if dataframe to csv required, default False
    :return: list of dataframes for each pdb file in the given directory
    """
    files_list = [f for f in os.listdir('./data/' + dir)]
    df_list = []

    for f in files_list:
      df = self.extract_single_pdb('./data/' + dir + "/" + f, save_to_csv, dir)
      if df.empty:
        print("File " + f + " excluded")
      else:
        df_list.append(df)

    return df_list

  def extract_single_pdb(self, f, save_to_csv=False, dir=None):
    """
    Extracts data from single pdb file and displays in pandas dataframe
    :param f: name of the file <string>
    :param dir: Name of the directory <string>, Required when save_to_csv True
    :param save_to_csv: bool, true if dataframe to csv required, default False
    :return: pandas dataframe or False is any acid is not in given dict
    """
    data = ppdb().read_pdb(f)
    f = f[::-1]
    idx_dot = f.index('.')
    idx_slash = f.index('/')
    f = f[idx_dot+1:idx_slash]
    file_name = f[::-1]

    # read starting helix range
    dbref = data.df['OTHERS'][data.df['OTHERS']['record_name'] == 'DBREF']['entry']
    if len(dbref) == 0:
      dbref = data.df['OTHERS'][data.df['OTHERS']['record_name'] == 'DBREF2']['entry']
      if len(dbref) == 0:
        # empty df
        return pd.DataFrame({})

      start_range = dbref[dbref.first_valid_index()][39:49]
    else:
      start_range = dbref[dbref.first_valid_index()][49:54]

    # get the helic ranges
    helix_ranges = []
    for string in data.df['OTHERS'][data.df['OTHERS']['record_name'] == 'HELIX']['entry']:
      # Only get for model A
      if (string[13].strip() == 'A'):
        start = int(string[16:19]) - int(start_range)
        end = int(string[28:31]) - int(start_range)
        helix_ranges.append((start, end))

    # gets the amino acids sequences
    final_str = []
    for string in data.df['OTHERS'][data.df['OTHERS']['record_name'] == 'SEQRES']['entry']:
      # Only get for model A
      if (string[5].strip() == 'A'):
        final_str.extend(string[13:].split(sep=' '))

    # create the labels
    label = np.zeros(len(final_str))
    for st, end in helix_ranges:
      for i in range(st, end + 1):
        if i >= len(label) or i < (-1*len(label)):
          # empty df
          return pd.DataFrame({})

        label[i] = 1

    # create the dataframe
    df = pd.DataFrame({'acids': final_str, 'helix': label})

    def map_acid(x):
      if x not in self.refer_protein:
        return None
      else:
        return self.refer_protein[x]

    df['acid_num'] = df['acids'].apply(map_acid)

    if df['acid_num'].isnull().any():
      # empty df
      return pd.DataFrame({})

    # save to csv
    if save_to_csv:
      path = './data/'+dir+'/csv/'
      if not os.path.exists(path):
        os.mkdir(path)
      path += file_name+'.csv'
      df.to_csv(path, index=False)

    return df
