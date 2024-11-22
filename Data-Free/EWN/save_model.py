import torch

def save_model(save_path, epoch, optimizer, model):
    torch.save({'epoch': epoch,
                'optimizer_dict': optimizer.state_dict(),
                'model_dict': model.state_dict()},
                save_path)
    print("model save success")

def load_model(save_name, optimizer, model):
    model_data = torch.load(save_name)
    model.load_state_dict(model_data['model_dict'])
    optimizer.load_state_dict(model_data['optimizer_dict'])
    print("model load success")
