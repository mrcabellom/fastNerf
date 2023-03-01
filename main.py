import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from model import FastNerf, Cache
from utils.rays import render_rays
import time


@torch.no_grad()
def test(model, hn, hf, dataset, img_index=0, nb_bins=192, H=400, W=400):
    ray_origins = dataset[img_index * H * W: (img_index + 1) * H * W, :3]
    ray_directions = dataset[img_index * H * W: (img_index + 1) * H * W, 3:6]
    regenerated_px_values = render_rays(model, ray_origins.to(device), ray_directions.to(device), hn=hn, hf=hf,
                                        nb_bins=nb_bins)

    plt.figure()
    plt.imshow(regenerated_px_values.data.cpu().numpy().reshape(H, W, 3).clip(0, 1))
    plt.axis('off')
    plt.savefig(f'test/img_{img_index}.png', bbox_inches='tight')
    plt.close()


def train(nerf_model, optimizer, scheduler, data_loader, device='cpu', hn=0, hf=1, nb_epochs=int(1e5),
          nb_bins=192):
    training_loss = []
    for _ in (range(nb_epochs)):
        for ep, batch in enumerate(tqdm(data_loader)):

            ray_origins = batch[:, :3].to(device)
            ray_directions = batch[:, 3:6].to(device)
            ground_truth_px_values = batch[:, 6:].to(device)

            regenerated_px_values = render_rays(nerf_model, ray_origins, ray_directions, hn=hn, hf=hf, nb_bins=nb_bins)
            loss = ((ground_truth_px_values - regenerated_px_values) ** 2).sum()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            training_loss.append(loss.item())
        scheduler.step()
        torch.save(nerf_model.cpu(), 'nerf_model')
        nerf_model.to(device)
    return training_loss


if __name__ == '__main__':

    device = 'cuda'
    training_dataset = torch.from_numpy(np.load('data/training_data.pkl', allow_pickle=True))
    testing_dataset = torch.from_numpy(np.load('data/testing_data.pkl', allow_pickle=True))
    model = FastNerf().to(device)
    model_optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(model_optimizer, milestones=[2, 4, 8], gamma=0.5)
    data_loader = DataLoader(training_dataset, batch_size=1024, shuffle=True)
    train(model, model_optimizer, scheduler, data_loader, nb_epochs=16, device=device, hn=2, hf=6)

    cache = Cache(model, 2.2, device, 192, 128)

    for idx in range(200):
        start_time = time.time()
        test(cache, 2., 6., testing_dataset, img_index=idx, nb_bins=192, H=400, W=400)
        print("FPS: ", 1.0 / (time.time() - start_time))