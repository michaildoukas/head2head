import time
import os
import numpy as np
import torch
from torch.autograd import Variable
from collections import OrderedDict
import fractions
from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
import util.util as util
from util.visualizer import Visualizer

opt = TrainOptions().parse()
n_frames_G, n_frames_D = opt.n_frames_G, opt.n_frames_D
t_scales = opt.n_scales_temporal
input_nc = opt.input_nc
output_nc = opt.output_nc

visualizer = Visualizer(opt)

### initialize dataset
data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
dataset_size = len(data_loader)

### initialize models
modelG, modelD, flowNet = create_model(opt)

iter_path = os.path.join(opt.checkpoints_dir, opt.name, 'iter.txt')
if opt.continue_train:
    try:
        start_epoch, epoch_iter = np.loadtxt(iter_path , delimiter=',', dtype=int)
    except:
        start_epoch, epoch_iter = 1, 0
    if epoch_iter > 0:
        ### initialize dataset again
        if opt.serial_batches:
            data_loader = CreateDataLoader(opt, epoch_iter)
            dataset = data_loader.load_data()
    visualizer.vis_print('Resuming from epoch %d at iteration %d' % (start_epoch, epoch_iter))
    if start_epoch > opt.niter:
        modelG.module.update_learning_rate(start_epoch-1)
        modelD.module.update_learning_rate(start_epoch-1)
    if start_epoch > opt.niter_step:
        data_loader.dataset.update_sequence_length((start_epoch-1)//opt.niter_step)
else:
    start_epoch, epoch_iter = 1, 0

total_steps = (start_epoch-1) * dataset_size + epoch_iter
for epoch in range(start_epoch, opt.niter + opt.niter_decay + 1):
    epoch_start_time = time.time()
    for idx, data in enumerate(dataset):
        if total_steps % opt.print_freq == 0:
            iter_start_time = time.time()
        save_fake = total_steps % opt.display_freq == 0

        _, n_frames_total, height, width = data['rgb_video'].size()
        n_frames_total = n_frames_total // opt.output_nc
        n_frames_load = opt.max_frames_per_gpu
        n_frames_load = min(n_frames_load, n_frames_total - n_frames_G + 1)
        t_len = n_frames_load + n_frames_G - 1
        fake_B_last = None
        real_B_all, fake_B_all, flow_ref_all, conf_ref_all = None, None, None, None
        real_B_skipped, fake_B_skipped = [None]*t_scales, [None]*t_scales
        flow_ref_skipped, conf_ref_skipped = [None]*t_scales, [None]*t_scales

        for i in range(0, n_frames_total-t_len+1, n_frames_load):
            A_paths = data['A_paths'][i]
            nmfc_video = Variable(data['nmfc_video'][:, i*3:(i+t_len)*3, ...]).view(-1, t_len, 3, height, width) # nmfc_video have 3 channels
            input_A = nmfc_video
            input_B = Variable(data['rgb_video'][:, i*3:(i+t_len)*3, ...]).view(-1, t_len, 3, height, width) # rgb_video has 3 channels
            mouth_centers = Variable(data['mouth_centers'][:, i:i+t_len, ...]).view(-1, t_len, 2) if not opt.no_mouth_D else None
            eyes_centers = Variable(data['eyes_centers'][:, i:i+t_len, ...]).view(-1, t_len, 2) if opt.use_eyes_D else None

            if not opt.no_eye_gaze:
                eye_gaze_video = Variable(data['eye_video'][:, i*3:(i+t_len)*3, ...]).view(-1, t_len, 3, height, width) # eye_gaze_video has 3 channels
                input_A = torch.cat([nmfc_video, eye_gaze_video], dim=2)

            ############## Forward Pass ######################
            # Identity Embedder and Generator
            fake_B, real_A, real_Bp, fake_B_last = modelG(input_A, input_B, fake_B_last)

            if i == 0:
                fake_B_first = fake_B[0, 0]
            real_B_prev, real_B = real_Bp[:, :-1], real_Bp[:, 1:]
            flow_ref, conf_ref = flowNet(real_B, real_B_prev)  # reference flows and confidences
            fake_B_prev = real_B_prev[:, 0:1] if fake_B_last is None else fake_B_last[:, -1:]
            if fake_B.size()[1] > 1:
                fake_B_prev = torch.cat([fake_B_prev, fake_B[:, :-1].detach()], dim=1)
            if mouth_centers is not None:
                mouth_centers = mouth_centers.contiguous().view(-1, 2)[n_frames_G-1:,:]
            if eyes_centers is not None:
                eyes_centers = eyes_centers.contiguous().view(-1, 2)[n_frames_G-1:,:]
            tensor_list = util.reshape([real_B, fake_B, real_A, real_B_prev, fake_B_prev, flow_ref, conf_ref])

            # Image and Mouth, Eyes Discriminators
            losses = modelD(0, tensor_list, mouth_centers, eyes_centers)
            losses = [ torch.mean(x) if x is not None else 0 for x in losses ]
            loss_dict = dict(zip(modelD.module.loss_names, losses))

            # Dynamics Discriminator
            loss_dict_T = []
            if t_scales > 0:
                real_B_all, real_B_skipped = util.get_skipped_frames(real_B_all, real_B, t_scales, n_frames_D)
                fake_B_all, fake_B_skipped = util.get_skipped_frames(fake_B_all, fake_B, t_scales, n_frames_D)
                flow_ref_all, conf_ref_all, flow_ref_skipped, conf_ref_skipped = util.get_skipped_flows(flowNet, flow_ref_all, conf_ref_all, real_B_skipped,
                                                                                                   flow_ref, conf_ref, t_scales, n_frames_D)
            for s in range(t_scales):
                if real_B_skipped[s] is not None:
                    losses = modelD(s+1, [real_B_skipped[s], fake_B_skipped[s], flow_ref_skipped[s], conf_ref_skipped[s]])
                    losses = [ torch.mean(x) if not isinstance(x, int) else x for x in losses ]
                    loss_dict_T.append(dict(zip(modelD.module.loss_names_T, losses)))

            # Losses
            loss_D = (loss_dict['D_fake'] + loss_dict['D_real']) * 0.5
            loss_G = loss_dict['G_GAN'] + loss_dict['G_GAN_Feat'] + loss_dict['G_VGG'] + loss_dict['G_Warp']
            if not opt.no_mouth_D:
                loss_G += loss_dict['Gm_GAN'] + loss_dict['Gm_GAN_Feat']
                loss_D += (loss_dict['Dm_fake'] + loss_dict['Dm_real']) * 0.5
            if opt.use_eyes_D:
                loss_G += loss_dict['Ge_GAN'] + loss_dict['Ge_GAN_Feat']
                loss_D += (loss_dict['De_fake'] + loss_dict['De_real']) * 0.5
            loss_D_T = []
            actual_t_scales = min(t_scales, len(loss_dict_T))
            for s in range(actual_t_scales):
                loss_G += loss_dict_T[s]['G_T_GAN'] + loss_dict_T[s]['G_T_GAN_Feat']
                loss_D_T.append((loss_dict_T[s]['D_T_fake'] + loss_dict_T[s]['D_T_real']) * 0.5)

            ############### Backward Pass ####################
            optimizer_G = modelG.module.optimizer_G
            optimizer_D = modelD.module.optimizer_D
            optimizer_G.zero_grad()
            loss_G.backward()
            optimizer_G.step()
            optimizer_D.zero_grad()
            loss_D.backward()
            optimizer_D.step()
            for s in range(actual_t_scales):
                optimizer_D_T = getattr(modelD.module, 'optimizer_D_T'+str(s))
                optimizer_D_T.zero_grad()
                loss_D_T[s].backward()
                optimizer_D_T.step()

        visualizer.vis_print('Video path: ' + A_paths[0])
        total_steps += opt.batchSize
        epoch_iter += opt.batchSize

        ############## Display results and errors ##########
        ### print out errors
        if total_steps % opt.print_freq == 0:
            t = (time.time() - iter_start_time) / opt.print_freq
            errors = {k: v.data.item() if not isinstance(v, int) else v for k, v in loss_dict.items()}
            for s in range(len(loss_dict_T)):
                errors.update({k+str(s): v.data.item() if not isinstance(v, int) else v for k, v in loss_dict_T[s].items()})
            visualizer.print_current_errors(epoch, epoch_iter, errors, t)
            visualizer.plot_current_errors(errors, total_steps)

        ### display output images
        if save_fake:
            visual_dict = [('input_nmfc_image', util.tensor2im(nmfc_video[0, -1], normalize=False)),
                           ('fake_image', util.tensor2im(fake_B[0, -1])),
                           ('fake_first_image', util.tensor2im(fake_B_first)),
                           ('real_image', util.tensor2im(real_B[0, -1])),
                           ('flow_ref', util.tensor2flow(flow_ref[0, -1])),
                           ('conf_ref', util.tensor2im(conf_ref[0, -1], normalize=False))]
            if not opt.no_eye_gaze:
                visual_dict += [('input_eye_gaze_image', util.tensor2im(eye_gaze_video[0, -1], normalize=False))]
            if not opt.no_mouth_D:
                mc = util.fit_ROI_in_frame(mouth_centers[-1], opt)
                fake_B_mouth = util.tensor2im(util.crop_ROI(fake_B[0, -1], mc, opt.ROI_size))
                visual_dict += [('fake_image_mouth', fake_B_mouth)]
            if opt.use_eyes_D:
                mc = util.fit_ROI_in_frame(eyes_centers[-1], opt)
                fake_B_eyes = util.tensor2im(util.crop_ROI(fake_B[0, -1], mc, opt.ROI_size))
                visual_dict += [('fake_image_eyes', fake_B_eyes)]
            visuals = OrderedDict(visual_dict)
            visualizer.display_current_results(visuals, epoch, total_steps)

        ### save latest model
        if total_steps % opt.save_latest_freq == 0:
            visualizer.vis_print('saving the latest model (epoch %d, total_steps %d)' % (epoch, total_steps))
            modelG.module.save('latest')
            modelD.module.save('latest')
            np.savetxt(iter_path, (epoch, epoch_iter), delimiter=',', fmt='%d')
            visualizer.vis_print('Saved the latest model (epoch %d, epoch iterations %d)' % (epoch, epoch_iter))

        if epoch_iter > dataset_size - opt.batchSize:
            epoch_iter = 0
            break

    # end of epoch
    iter_end_time = time.time()
    visualizer.vis_print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))

    ### save model for this epoch, as latest
    visualizer.vis_print('saving as latest the model at the end of epoch %d, total_steps %d' % (epoch, total_steps))
    modelG.module.save('latest')
    modelD.module.save('latest')
    np.savetxt(iter_path, (epoch+1, 0), delimiter=',', fmt='%d')
    visualizer.vis_print('Saved the latest the model at the end of epoch %d, total_steps %d' % (epoch, total_steps))

    if epoch % opt.save_epoch_freq == 0:
        visualizer.vis_print('saving the model at the end of epoch %d, iters %d' % (epoch, total_steps))
        modelG.module.save(epoch)
        modelD.module.save(epoch)

    ### linearly decay learning rate after certain iterations
    if epoch > opt.niter:
        modelG.module.update_learning_rate(epoch)
        modelD.module.update_learning_rate(epoch)

    ### grow training sequence length
    data_loader.dataset.update_sequence_length(epoch//opt.niter_step)
