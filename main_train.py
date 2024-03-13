from get_opts import *

from nerf.provider import NeRFDataset
from nerf.gui import NeRFGUI
from nerf.trainer import *
import os
pienerf_dir = os.path.dirname(os.path.abspath(__file__))

#torch.autograd.set_detect_anomaly(True)

os.environ['KMP_DUPLICATE_LIB_OK'] = '1'

if __name__ == '__main__':
    if not os.path.exists(pienerf_dir + "/model"):
        os.mkdir(pienerf_dir + "/model")

    parser = argparse.ArgumentParser()
    opt = get_shared_opts(parser)
    opt.workspace = pienerf_dir + "/model/" + opt.path.split("/")[-1]

    if opt.ff:
        opt.fp16 = True
        assert opt.bg_radius <= 0, "background model is not implemented for --ff"
        from nerf.network_ff import NeRFNetwork
    elif opt.tcnn:
        opt.fp16 = True
        assert opt.bg_radius <= 0, "background model is not implemented for --tcnn"
        from nerf.network_tcnn import NeRFNetwork
    else:
        from nerf.network import NeRFNetwork

    seed_everything(opt.seed)

    model = NeRFNetwork( # NeRFNetwork inherits NeRFRenderer
        encoding="hashgrid",
        bound=opt.bound,
        cuda_ray=opt.cuda_ray,
        density_scale=1,
        min_near=opt.min_near,
        density_thresh=opt.density_thresh,
        bg_radius=opt.bg_radius,
    )
    
    print(model)

    criterion = torch.nn.MSELoss(reduction='none')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if opt.test:
        metrics = [PSNRMeter(), LPIPSMeter(device=device)]
        trainer = Trainer('ngp', opt, model, device=device, workspace=opt.workspace, criterion=criterion, fp16=opt.fp16, metrics=metrics, use_checkpoint=opt.ckpt)

        if opt.gui:
            gui = NeRFGUI(opt, trainer)
            gui.render()
        
        else:
            test_loader = NeRFDataset(opt, device=device, type='test').dataloader()

            if test_loader.has_gt:
                trainer.evaluate(test_loader) # blender has gt, so evaluate it.
    
            # trainer.test(test_loader, write_video=True) # test and save video
            
            trainer.save_mesh(resolution=256, threshold=10)

    else:
        optimizer = lambda model: torch.optim.Adam(model.get_params(opt.lr), betas=(0.9, 0.99), eps=1e-15)

        train_loader = NeRFDataset(opt, device=device, type='train').dataloader()

        # decay to 0.1 * init_lr at last iter step
        scheduler = lambda optimizer: optim.lr_scheduler.LambdaLR(optimizer, lambda iter: 0.1 ** min(iter / opt.iters, 1))

        metrics = [PSNRMeter(), LPIPSMeter(device=device)]
        trainer = Trainer('ngp', opt, model, device=device, workspace=opt.workspace, optimizer=optimizer,
                          criterion=criterion, ema_decay=0.95, fp16=opt.fp16, lr_scheduler=scheduler,
                          scheduler_update_every_step=True, metrics=metrics, use_checkpoint=opt.ckpt, eval_interval=50)

        if opt.gui:
            gui = NeRFGUI(opt, trainer, train_loader)
            gui.render()
        
        else:
            valid_loader = NeRFDataset(opt, device=device, type='val', downscale=1).dataloader()

            max_epoch = np.ceil(opt.iters / len(train_loader)).astype(np.int32)
            trainer.train(train_loader, valid_loader, max_epoch)####

            # also test
            test_loader = NeRFDataset(opt, device=device, type='test').dataloader()
            
            if test_loader.has_gt:
                trainer.evaluate(test_loader) # blender has gt, so evaluate it.
            
            # trainer.test(test_loader, write_video=True) # test and save video
            
            # trainer.save_mesh(resolution=256, threshold=10)

            trainer.save_point_cloud(resolution=256)
