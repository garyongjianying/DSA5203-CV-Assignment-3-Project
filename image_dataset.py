import torch
from torch.utils.data import Dataset
from PIL import Image

class ImageDataset(Dataset):
    """
    Generate dataset from a list of image file paths

    """
    def __init__(self, labels_df, transform=None):
        """
        Initialize the relevant variables

        labels_df: pandas dataframe that contains the image path and its lable
        tranform: function that performs preprocessing on the dataset
        """
        self.labels_df = labels_df
        self.transform = transform

    # required to load into torch DataLoader()
    def __len__(self):
        """
        Get the number of samples of the dataset
        """
        return len(self.labels_df)

    def __getitem__(self, index):
        """
        Retrieve the corresponding sample & label pair when used in 
        model training/testing/validation.

        index: int() index of sample to retrieve
        :return: torch.tensor of image tensor and it's label
        """
        # need to use iloc for Dataloader to run
        img_path = self.labels_df.iloc[index, 0]
        # img = cv.imread(img_path)
        img = Image.open(img_path).convert('RGB')
        y_label = torch.tensor(float(self.labels_df.iloc[index, 1]))
        # need to convert to Long type to calculate CrossEntropy Loss
        y_label = y_label.type(torch.LongTensor)

        if self.transform is not None:
            img = self.transform(img)

        return (img, y_label)
