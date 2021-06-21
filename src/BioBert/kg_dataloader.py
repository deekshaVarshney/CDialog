import os.path as osp

import torch
from torch_geometric.data import Dataset, download_url, Data, InMemoryDataset, DataLoader



class KgDataLoader(InMemoryDataset):
	def __init__(self, root, transform=None, pre_transform=None):
		super(KgDataLoader, self).__init__(root, transform, pre_transform)
		self.data, self.slices = torch.load(self.processed_paths[0])

	@property
	def raw_file_names(self):
		return ['kg_train_data.pkl', 'kg_test_data.pkl', 'kg_validate_data.pkl']

	@property
	def processed_file_names(self):
		return ['data_train.pt', 'data_test.pt', 'data_validate.pt']

	def download(self):
		pass
	    
	def process(self):
		i = 0
		for raw_path in self.raw_paths:
			# Read data from `raw_path`.
			print(raw_path)

			data_l = torch.load(raw_path)
			data_list = []    
			# print(len(data_l))
			for item in data_l:
				edge_index, feature_matrix = item
				node_features = torch.tensor(feature_matrix).cuda()
				edge_index = torch.tensor(edge_index).cuda()
				x = node_features
				data = Data(x=x, edge_index=edge_index)
				data_list.append(data)

			self.data, self.slices = self.collate(data_list)
			print(self.processed_paths[i])
			torch.save((self.data, self.slices), self.processed_paths[i])

			i += 1


if __name__ == '__main__':
	data = KgDataLoader('.')
	print(data[1])
	loader = DataLoader(data, batch_size=3, shuffle=True)
	for batch in loader:
	    print(batch)