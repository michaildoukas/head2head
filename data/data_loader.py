
def CreateDataLoader(opt, start_idx=0):
    from data.custom_dataset_data_loader import CustomDatasetDataLoader
    data_loader = CustomDatasetDataLoader()
    #print(data_loader.name())
    data_loader.initialize(opt, start_idx)
    return data_loader
