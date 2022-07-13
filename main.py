
from data.plantnet import PlantNet

if __name__ == "__main__":

    data = PlantNet()

    train_loader = data.train
    val_loader = data.val

