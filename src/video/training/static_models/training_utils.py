import torch


def calculate_class_weights(train_dataloader:torch.utils.data.DataLoader, num_classes:int)->torch.Tensor:
    """ Calculate class weights from the data presented by the train_dataloader. The labels should be provided as
    categories.

    Args:
        train_dataloader: torch.utils.data.DataLoader
                The dataloader for the training set.
        num_classes: int
                The number of classes in the dataset.

    Returns: torch.Tensor
        The class weights calculated from the provided data. THe weights are calculated as
        1 / (number of samples in class / total number of samples). (Inverse proportion)

    """
    class_weights = torch.zeros(num_classes)
    for _, labels in train_dataloader:
        one_hot_labels = labels[:,-1]
        one_hot_labels = torch.nn.functional.one_hot(one_hot_labels, num_classes=num_classes)
        class_weights += torch.sum(one_hot_labels, dim=0)
    class_weights = class_weights / torch.sum(class_weights)
    class_weights = 1. / class_weights

    return class_weights
