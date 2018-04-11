import os
import pandas as pd

"""Iterate over all the csv files in the directory,
load it in a DataFrame with pandas and create a list of 
dicts:
[{
	'name': str,
	'n_samples': int,
	'n_features': int,
	'n_classes': int,
	'data': pd.DataFrame
}]"""

def load_data():
	data = []
	for file in os.listdir('.'):
		try:
			if file.endswith('csv'):
				df = pd.read_csv(file)	
				if 'target' in df.columns:
					data.append({
						'name': file.split('.')[0],
						'n_samples': len(df),
						'n_features': len(df.columns) - 1,
						'n_classes': len(df['target']),
						'data': df
					})
		except:
			print('Loading {} failed...'.format(file))
	return data