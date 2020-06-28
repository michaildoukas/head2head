import time
import os
import numpy as np
import torch
import torchvision
import cv2
import dlib
from torch.autograd import Variable
from collections import OrderedDict
from PIL import Image
from multiprocessing import Process, Queue
from torch.multiprocessing import Process as torchProcess
from torch.multiprocessing import Queue as torchQueue
import queue
from facenet_pytorch import MTCNN, extract_face
import util.util as util
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from preprocessing.reconstruction import NMFCRenderer
from preprocessing.reenact import read_params, read_eye_landmarks, search_eye_centres
from preprocessing.reenact import compute_eye_landmarks_ratio, adapt_eye_landmarks
from preprocessing.detect import tensor2npimage
from preprocessing.detect_landmarks70 import add_eye_pupils_landmarks
from data.landmarks_to_image import create_eyes_image
from data.base_dataset import get_transform, get_params


def make_frame_square(frame):
    h, w = frame.shape[:2]
    diff = abs(h - w)
    if h > w:
        frame = frame[diff//2:diff//2+w,:,:]
    else:
        frame = frame[:,diff//2:diff//2+h,:]
    return frame

def detect_box(detector, frame):
    # Detect face
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    imgs_pil = [Image.fromarray(frame)]
    boxes, _, _ = detector.detect(imgs_pil, landmarks=True)
    if boxes[0] is None:
        return None
    box = boxes[0][0]
    # Make box square.
    offset_w = box[2] - box[0]
    offset_h = box[3] - box[1]
    offset_dif = (offset_h - offset_w) / 2
    # width
    box[0] = box[2] - offset_w - offset_dif
    box[2] = box[2] + offset_dif
    return box

def compute_eye_landmarks(detector, predictor, eye_landmarks_source_queue, landmarks_success_queue, frames_queue):
    while True:
        frame = frames_queue.get()
        dets = detector(frame, 1)
        if len(dets) > 0:
            shape = predictor(frame, dets[0])
            points = np.empty([70, 2], dtype=int)
            for b in range(68):
                points[b,0] = shape.part(b).x
                points[b,1] = shape.part(b).y
            points = add_eye_pupils_landmarks(points, frame)
            left_eye = np.concatenate([points[36:42], points[68:69]], axis=0)
            right_eye = np.concatenate([points[42:48], points[69:70]], axis=0)
            eye_landmarks_source_queue.put((left_eye, right_eye))
            landmarks_success_queue.put(True)
        else:
            print('No face detected from landmarks extractor')
            landmarks_success_queue.put(False)

def compute_reconstruction(renderer, id_params, t_cam_params, s_cam_params, adapted_cam_params, frame):
    n_frames_source_memory = 500 # Hardcoded
    success, exp_params, cam_params = renderer.get_expression_and_pose(frame)
    if success:
        s_cam_params.append(cam_params)
        if len(s_cam_params) > n_frames_source_memory:
            s_cam_params = s_cam_params[1:]
        # Adapt camera parameters to target
        ad_cam_params = adapt_cam_params(s_cam_params, t_cam_params)
        adapted_cam_params.append(ad_cam_params)
        if len(adapted_cam_params) > n_frames_source_memory:
            adapted_cam_params = adapted_cam_params[1:]
        # Compute NMFC
        nmfc = renderer.computeNMFC(ad_cam_params, id_params, exp_params)
        return True, s_cam_params, adapted_cam_params, nmfc
    else:
        print('No face detected from NMFC renderer')
        return False, s_cam_params, adapted_cam_params, None

def compute_fake_video(input_queue, output_queue, modelG, opt):
    input_A_all = None
    while True:
        # Read input.
        conditional_input = input_queue.get()
        nmfc, eye_landmarks, real_frame = conditional_input
        width, height = nmfc.shape[0:2]
        # Create tensors
        params = get_params(opt, (width, height))
        transform_scale_nmfc_video = get_transform(opt, params, normalize=False, augment=False)
        nmfc = transform_scale_nmfc_video(Image.fromarray(nmfc))
        transform_scale_eye_gaze_video = get_transform(opt, params, normalize=False)
        eye_gaze = create_eyes_image(None, (width, height), transform_scale_eye_gaze_video,
                                     add_noise=False, pts=eye_landmarks)
        # Concat conditional inputs.
        input_A = torch.cat([nmfc, eye_gaze], dim=0)
        if input_A_all is None:
            # If no previously generated frames available, pad zeros
            input_A_all = torch.cat([torch.zeros((opt.n_frames_G-1) * opt.input_nc,
                                                 width, height), input_A], dim=0)
        else:
            # Discard oldest conditional input and append new one.
            input_A_all = torch.cat([input_A_all[opt.input_nc:,:,:], input_A], dim=0)
        input_A_final = input_A_all.view(1, -1, opt.input_nc, width, height)
        # Forward pass through Generator.
        generated = modelG.inference(input_A_final, None)
        fake_frame = util.tensor2im(generated[0].data[0])
        # Write results to Queue.
        output_queue.put((fake_frame, real_frame))

def adapt_cam_params(s_cam_params, t_cam_params):
    # Re-scale
    mean_S_target = np.mean([params[0] for params in t_cam_params])
    mean_S_source = np.mean([params[0] for params in s_cam_params])
    S = s_cam_params[-1][0] * (mean_S_target / mean_S_source)
    # Compute normalised translation params for source and target.
    nT_target = [params[2] / params[0] for params in t_cam_params]
    nT_source = [params[2] / params[0] for params in s_cam_params]
    # Get statistics.
    mean_nT_target = np.mean(nT_target, axis=0)
    mean_nT_source = np.mean(nT_source, axis=0)
    std_nT_target = np.std(nT_target, axis=0)
    # Allow camera translation two standard deviation away from the one on target video.
    upper_limit = mean_nT_target + std_nT_target * 2
    lower_limit = mean_nT_target - std_nT_target * 2
    nT = np.maximum(np.minimum(nT_source[-1] - mean_nT_source + mean_nT_target,
                               upper_limit), lower_limit)
    cam_params = (S, s_cam_params[-1][1], S * nT)
    return cam_params

def main():
    # Read options
    opt = TestOptions().parse(save=False)
    # If demo directory to save generated frames is given
    if opt.demo_dir is not None and not os.path.exists(opt.demo_dir):
        os.makedirs(opt.demo_dir)

    # hardcoded constant values
    opt.nThreads = 0
    opt.batchSize = 1
    opt.serial_batches = True
    # GPU id to be used for mxnet/reconstructor
    opt.gpu_id = opt.gpu_ids[-1]
    # Device to be used for MTCNN face detector
    detector_device = 'cpu'
    # Face bounding box margin
    margin = 120
    # How many frames from the target's training video
    # to consider when gathering head pose and eye size statistics
    n_frames_target_used = 1000
    # How many of the first source frames to consider for eye size adaptation
    # between source and target.
    n_frames_init = 25
    # For cuda initialization errors.
    torch.multiprocessing.set_start_method('spawn', force=True)

    # Initialize video renderer.
    modelG = create_model(opt)
    # Initialize NMFC renderer.
    renderer = NMFCRenderer(opt)
    # Initialize face detector.
    detector = MTCNN(image_size=opt.loadSize, margin=margin,
                     post_process=False, device=detector_device)
    # Initialize landmark extractor.
    dlib_detector = dlib.get_frontal_face_detector()
    dlib_predictor = dlib.shape_predictor('preprocessing/files/shape_predictor_68_face_landmarks.dat')

    # Read the identity parameters from the target person.
    id_params, _ = read_params('id', os.path.join(opt.dataroot,
                               'train', 'id_coeffs'), opt.target_name)
    # Read camera parameters from target
    t_cam_params, _ = read_params('cam', os.path.join(opt.dataroot,
                                  'train', 'misc'), opt.target_name)
    t_cam_params = t_cam_params[:n_frames_target_used]
    # Read eye landmarks from target's video.
    eye_landmarks_target = read_eye_landmarks(os.path.join(opt.dataroot,
                            'train', 'landmarks70'), opt.target_name)
    eye_landmarks_target[0] = eye_landmarks_target[0][:n_frames_target_used]
    eye_landmarks_target[1] = eye_landmarks_target[1][:n_frames_target_used]

    # Setup camera capturing
    window_name = 'Hea2Head Demo'
    video_capture = cv2.VideoCapture(0)
    video_capture.set(cv2.CAP_PROP_BUFFERSIZE, 2) # set double buffer for capture
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    print("Video capture at {} fps.".format(fps))

    proccesses = []

    # Face tracker / detector
    box = None # Face bounding box, calculated by first frame

    # Face reconstructor / NMFC renderer
    nmfc = None # Current nmfc image
    s_cam_params = [] # camera parameters of source video.
    adapted_cam_params = [] # camera parameters of source video, adapted to target.

    # Facial (eyes) landmarks detector
    prev_eye_centres = None # Eye centres in previous frame
    eye_landmarks = None # Final eye landmarks, send to video renderer.
    eye_landmarks_source = [[], []] # Eye landmarks from n_frames_init first frames of source video.
    eye_landmarks_source_queue = Queue() # Queue to write extracted eye landmarks from source video.
    landmarks_success_queue = Queue() # Queue to write whether eye landmark detection was successful
    frames_queue = Queue() # Queue for writing video frames, read by the landmark detector process.
    # Process for running 68 + 2 landmark detection in parallel with Face reconstruction / NMFC renderering
    proccess_eye_landmarks = Process(target=compute_eye_landmarks,
            args=(dlib_detector, dlib_predictor, eye_landmarks_source_queue,
                  landmarks_success_queue, frames_queue))
    proccess_eye_landmarks.start()
    proccesses.append(proccess_eye_landmarks)
    print('Launced landmark extractor!')

    # Video renderer (GAN).
    input_queue = torchQueue() # Queue of GAN's input
    output_queue = torchQueue() # Queue of GAN's output
    # Process for running the video renderer without waiting NMFC + eye lands creation.
    proccess_video_renderer = torchProcess(target=compute_fake_video, args=(input_queue, output_queue, modelG, opt))
    proccess_video_renderer.start()
    proccesses.append(proccess_video_renderer)
    print('Launced video renderer!')

    iter = 0
    # Start main Process (Face reconstruction / NMFC renderering)
    while True:
        t0 = time.perf_counter()
        try: # Read generated frames from video renderer's output Queue.
            # Non-blocking
            fake_frame, real_frame = output_queue.get_nowait()
            result = np.concatenate([real_frame, fake_frame[..., ::-1]], axis=1)
            # If output directory is specified save frames there.
            if opt.demo_dir is not None:
                result_path = os.path.join(opt.demo_dir, "{:06d}".format(iter) + '.png')
                cv2.imwrite(result_path, result)
            else:
                cv2.imshow(window_name, result)
        except queue.Empty: # If empty queue continue.
            pass
        # Read next frame
        _, frame = video_capture.read()
        # Crop the larger dimension of frame to make it square
        frame = make_frame_square(frame)
        # If no bounding box has been detected yet, run MTCNN (once in first frame)
        if box is None:
            box = detect_box(detector, frame)
        # If no face detected exit.
        if box is None:
            break
        # Crop frame at the point were the face was seen in the first frame.
        frame = extract_face(frame, box, opt.loadSize, margin)
        frame = tensor2npimage(frame)
        frame = np.transpose(frame, (1, 2, 0))
        # Send ROI frame to landmark detector, while the main Process performs face reconstruction.
        frames_queue.put(frame)
        # Get expression and pose, adapt pose and identity to target and render NMFC.
        success, s_cam_params, adapted_cam_params, new_nmfc = \
            compute_reconstruction(renderer, id_params, t_cam_params, s_cam_params,
                                   adapted_cam_params, frame)
        # Update the current NMFC if reconstruction was successful
        if success:
            nmfc = new_nmfc
        # If not, use previous nmfc. If it does not exist, exit.
        if not success and nmfc is None:
            break
        # Find eye centres using nmfc image.
        eye_centres, prev_eye_centres = search_eye_centres([nmfc[:,:,::-1]], prev_eye_centres)
        # Read Queue to get eye landmarks, if detection was successful.
        if landmarks_success_queue.get():
            eye_landmarks = eye_landmarks_source_queue.get()
        # If not, use previous eye landmarks. If they do not exist, exit.
        if eye_landmarks is None:
            break
        # If in first frames, determine the source-target eye size (height) ratio.
        if iter < n_frames_init:
            eye_landmarks_source[0].append(eye_landmarks[0])
            eye_landmarks_source[1].append(eye_landmarks[1])
            eye_ratios = compute_eye_landmarks_ratio(eye_landmarks_source,
                                                     eye_landmarks_target)
        # Adapt the eye landmarks to the target face, by placing to the eyes centre
        # and re-scaling their size to match the NMFC size and target eyes mean height (top-down distance).
        eye_lands = adapt_eye_landmarks([[eye_landmarks[0]], [eye_landmarks[1]]], eye_centres, eye_ratios,
                                        s_cam_params[-1:], adapted_cam_params[-1:])
        # Send the conditional input to video renderer
        input_queue.put((nmfc, eye_lands[0], frame))
        iter += 1
        # Show frame rate.
        t1 = time.perf_counter()
        dt = t1 - t0
        print('fps: %0.2f' % (1/dt))

    # Terminate proccesses and join
    for process in proccesses:
        process.terminate()
        process.join()

    renderer.clear()
    print('Main process exiting')

if __name__ == '__main__':
    main()
