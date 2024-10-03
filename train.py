import argparse
import os
import torch
from torch.utils import data
from dataset import Vox256, Taichi, TED, Mydataset256, Mydataset256_cross
import torchvision
import torchvision.transforms as transforms
from trainer import Trainer
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
import torch.multiprocessing as mp
import time
from tqdm import tqdm

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True


def data_sampler(dataset, shuffle):
    if shuffle:
        return data.RandomSampler(dataset)
    else:
        return data.SequentialSampler(dataset)

def sample_data(loader):
    while True:
        for batch in loader:
            yield batch

def display_img(idx, img, name, writer):
    img = img.clamp(-1, 1)
    img = ((img - img.min()) / (img.max() - img.min())).data
    writer.add_images(tag='%s' % (name), global_step=idx, img_tensor=img)

def ddp_setup(args, rank, world_size):
    os.environ['MASTER_ADDR'] = args.addr
    os.environ['MASTER_PORT'] = args.port
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def main(rank, world_size, args):
    

    ddp_setup(args, rank, world_size)
    torch.cuda.set_device(rank)
    device = torch.device("cuda")

    log_path = os.path.join(args.exp_path, args.exp_name + '/log')
    checkpoint_path = os.path.join(args.exp_path, args.exp_name + '/checkpoint')

    os.makedirs(log_path, exist_ok=True)
    os.makedirs(checkpoint_path, exist_ok=True)
    writer = SummaryWriter(log_path)

    print('==> preparing dataset')
    transform = torchvision.transforms.Compose([
        transforms.Resize((args.size, args.size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))]
    )


    dataset_same = Mydataset256(split='train',start=0,end=253, transform=transform, augmentation=False)
    dataset_cross = Mydataset256_cross(split='train',start=0,end=253, transform=transform, augmentation=False)
    dataset_test = Mydataset256_cross(split='train',start=0,end=253, transform=transform, augmentation=False)

    loader_same = data.DataLoader(
        dataset_same,
        num_workers=2,
        batch_size=args.batch_size // world_size,
        sampler=data.distributed.DistributedSampler(dataset_same, num_replicas=world_size, rank=rank, shuffle=True),
        pin_memory=True,
        drop_last=True,
        persistent_workers=True
    )

    loader_cross = data.DataLoader(
        dataset_cross,
        num_workers=2,
        batch_size=args.batch_size // world_size,
        sampler=data.distributed.DistributedSampler(dataset_cross, num_replicas=world_size, rank=rank, shuffle=True),
        pin_memory=True,
        drop_last=True,
        persistent_workers=True
    )

    loader_test = data.DataLoader(
        dataset_test,
        num_workers=1,
        batch_size=4,
        sampler=data.distributed.DistributedSampler(dataset_test, num_replicas=world_size, rank=rank, shuffle=True),
        pin_memory=False,
        drop_last=True,
        persistent_workers=True
    )

    loader_cross = sample_data(loader_cross)
    loader_same = sample_data(loader_same)
    loader_test = sample_data(loader_test)


    print('==> initializing trainer')
    # Trainer
    trainer = Trainer(args, device, rank)

    print('==> training')
    pbar = range(args.iter)

    id_triplet_loss = torch.tensor(0)
    id_loss = torch.tensor(0)
    r1_panelty = torch.tensor(0)

    for idx in pbar:
        i = idx + args.start_iter

        nerf_size = 64 + idx//100
        if nerf_size >= 128:
            nerf_size = 128

        # nerf_size = 64
        
        if idx % 2 == 0:
            img_source_0, img_target_256, camera_target, img_target_128, mask_target = next(loader_same)  
            vgg_loss_256, l1_loss_256, l1_loss_128, img_recon, id_loss_lia, exp_loss_lia, mask_loss, TVloss, id_consist_loss = trainer.gen_update_recon(img_source_0, img_target_256, camera_target, img_target_128, nerf_size, mask_target)
            vgg_loss_256, l1_loss_256, l1_loss_128, img_recon, id_loss_lia, exp_loss_lia, mask_loss, TVloss, id_consist_loss = trainer.gen_update_same_id(img_source_0, img_target_256, camera_target, img_target_128, nerf_size, mask_target)
        else:
            img_source_0, img_target_256, camera_target, img_target_128 = next(loader_cross)
            img_recon, id_loss_lia, exp_loss_lia, TVloss, id_triplet_loss, id_loss = trainer.gen_update_cross_id(img_source_0, img_target_256, camera_target, img_target_128, nerf_size)
            

        if rank == 0:
            # write to log
            write_loss(idx, vgg_loss_256, l1_loss_256, l1_loss_128, id_loss_lia, exp_loss_lia, mask_loss, TVloss, id_consist_loss,id_triplet_loss, id_loss, r1_panelty,writer)

        # display
        if i % args.display_freq == 0 and rank == 0:
            print("[Iter %d/%d] [vgg 256: %f] [l1 256: %f] [l1 128: %f] [id_lia: %f] [exp_lia: %f] [mask: %f] [TVloss: %f] [id_consist_loss: %f] [id_triplet: %f] [id_deca: %f] [r1_panelty: %f] "
                  % (i, args.iter, vgg_loss_256.item(), l1_loss_256.item(), l1_loss_128.item(),  id_loss_lia.item(),exp_loss_lia.item(), 
                  mask_loss.item(), TVloss.item(), id_consist_loss.item(), id_triplet_loss.item(),id_loss.item(), r1_panelty.item()))

            if rank == 0:
                torch.cuda.empty_cache()
                img_source_0, img_target_256, camera_target, img_target_128 = next(loader_test)
                img_d, img_d_256 = trainer.sample(img_source_0, img_target_256, camera_target, img_target_128, nerf_size)
                display_img(i, img_source_0, 'source', writer)
                display_img(i, img_target_256, 'target', writer)
                display_img(i, img_d['image_128'], 'gen_128', writer)
                display_img(i, img_d_256, 'gen_256', writer)
                # display_img(i, img_d[1].unsqueeze(0), 'driven', writer)
                writer.flush()
        # save model
        if i % args.save_freq == 0 and rank == 0:
            trainer.save(i, checkpoint_path)
    return


if __name__ == "__main__":
    # training params
    parser = argparse.ArgumentParser()
    parser.add_argument("--iter", type=int, default=8000000)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--d_reg_every", type=int, default=2)
    parser.add_argument("--g_reg_every", type=int, default=3)
    parser.add_argument("--resume_ckpt", type=str, default='.../020000.pt')
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--channel_multiplier", type=int, default=1)
    parser.add_argument("--start_iter", type=int, default=0)
    parser.add_argument("--display_freq", type=int, default=100) 
    parser.add_argument("--save_freq", type=int, default=1000) 
    parser.add_argument("--latent_dim_style", type=int, default=512)
    parser.add_argument("--latent_dim_motion", type=int, default=20) 
    parser.add_argument("--dataset", type=str, default='vox')
    parser.add_argument("--exp_path", type=str, default='./exps/')
    parser.add_argument("--exp_name", type=str, default='v23_5')
    parser.add_argument("--addr", type=str, default='localhost')
    parser.add_argument("--port", type=str, default='12140')
    opts = parser.parse_args()

    n_gpus = torch.cuda.device_count()
    assert n_gpus >= 2

    def force_cudnn_initialization():
        s = 32
        dev = torch.device('cuda')
        torch.nn.functional.conv2d(torch.zeros(s, s, s, s, device=dev), torch.zeros(s, s, s, s, device=dev))
    
    force_cudnn_initialization()

    world_size = n_gpus
    print('==> training on %d gpus' % n_gpus)
    mp.spawn(main, args=(world_size, opts,), nprocs=world_size, join=True)