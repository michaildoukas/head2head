import cv2
import os
import numpy as np
import pickle
import transform
import sys
import scipy.io as io
import glob
from scipy import optimize
from tqdm import tqdm
from multiface import fc_predictor
from avatars import serialize
from hephaestus import hephaestus_bindings as hephaestus

def _procrustes(X, Y, scaling=True, reflection='best'):
    """
    Procrustes analysis determines a linear transformation (translation,
    reflection, orthogonal rotation and scaling) of the points in Y to best
    conform them to the points in matrix X, using the sum of squared errors
    as the goodness of fit criterion.

        d, Z, [tform] = procrustes(X, Y)

    Inputs:
    ------------
    X, Y
        matrices of target and input coordinates. they must have equal
        numbers of  points (rows), but Y may have fewer dimensions
        (columns) than X.

    scaling
        if False, the scaling component of the transformation is forced
        to 1

    reflection
        if 'best' (default), the transformation solution may or may not
        include a reflection component, depending on which fits the data
        best. setting reflection to True or False forces a solution with
        reflection or no reflection respectively.

    Outputs
    ------------
    d
        the residual sum of squared errors, normalized according to a
        measure of the scale of X, ((X - X.mean(0))**2).sum()

    Z
        the matrix of transformed Y-values

    tform
        a dict specifying the rotation, translation and scaling that
        maps X --> Y

    """
    n, m = X.shape
    ny, my = Y.shape

    muX = X.mean(0)
    muY = Y.mean(0)

    X0 = X - muX
    Y0 = Y - muY

    ssX = (X0**2.).sum()
    ssY = (Y0**2.).sum()

    # centred Frobenius norm
    normX = np.sqrt(ssX)
    normY = np.sqrt(ssY)

    # scale to equal (unit) norm
    X0 /= normX
    Y0 /= normY

    if my < m:
        Y0 = np.concatenate((Y0, np.zeros(n, m-my)),0)

    # optimum rotation matrix of Y
    A = np.dot(X0.T, Y0)
    U, s, Vt = np.linalg.svd(A,full_matrices=False)
    V = Vt.T
    T = np.dot(V, U.T)

    if reflection is not 'best':

        # does the current solution use a reflection?
        have_reflection = np.linalg.det(T) < 0

        # if that's not what was specified, force another reflection
        if reflection != have_reflection:
            V[:,-1] *= -1
            s[-1] *= -1
            T = np.dot(V, U.T)

    traceTA = s.sum()

    if scaling:

        # optimum scaling of Y
        b = traceTA * normX / normY

        # standarised distance between X and b*Y*T + c
        d = 1 - traceTA**2

        # transformed coords
        Z = normX*traceTA*np.dot(Y0, T) + muX

    else:
        b = 1
        d = 1 + ssY/ssX - 2 * traceTA * normY / normX
        Z = normY*np.dot(Y0, T) + muX

    # transformation matrix
    if my < m:
        T = T[:my,:]
    c = muX - b*np.dot(muY, T)

    #transformation values
    tform = {'rotation':T, 'scale':b, 'translation':c}

    return d, Z, tform


def fit_3Dmodel_to_shape(Sinp, M, Bas, ver_idx, eta, Bcoefs):
    M_fit = M[:, ver_idx]
    Bas_use_3Darr = Bas.T.reshape((Bas.shape[1], -1, 3)).transpose((2, 1, 0))
    Bas_fit = Bas_use_3Darr[:, ver_idx, :]
    Bas_fit = Bas_fit.transpose((2, 1, 0)).reshape((Bas.shape[1], -1)).T
    _, Sinp_use, tform = _procrustes(M_fit.T, Sinp, reflection=False)
    A = Sinp_use.ravel()- M_fit.T.ravel()
    if(eta==0):
        coefs = optimize.lsq_linear(Bas_fit, A, bounds=(Bcoefs[:, 0], Bcoefs[:, 1]), method='trf',
                                    tol=1e-10, lsq_solver=None, lsmr_tol=None, max_iter=None, verbose=0)
    return coefs['x']

def est_id_exp(fmod, points, model_path, idxs, Wbound_Cid=.8, Wbound_Cexp=1.5):

    num_id = fmod['id_basis'].shape[1]
    num_exp = fmod['exp_basis'].shape[1]
    Bas = np.concatenate((fmod['id_basis'], fmod['exp_basis']), 1)
    eta = 0
    UBcoefs = np.vstack((Wbound_Cid*np.ones((num_id, 1)), Wbound_Cexp*np.ones((num_exp,1))) )
    #print(UBcoefs.shape, points.shape, Bas.shape)
    coefs = fit_3Dmodel_to_shape(points, fmod['mean'].reshape((-1, 3)).T, Bas, idxs, eta, np.hstack((-UBcoefs,UBcoefs)) )
    return coefs

class NMFCRenderer:
    def __init__(self, args):
        self.args = args
        self.MM_inpath ='preprocessing/files/all_all_all.mat'

        model_prefix = ['preprocessing/models/A3'] #NME 2.2 # without edge loss
        # model_prefix = ['./data/model/E2'] #NME 2.2 # with edge loss
        model_epoch = 70

        self.handler = fc_predictor.Handler(model_prefix, model_epoch, 192, args.gpu_id)

        # load sampling indices
        with open('preprocessing/files/sampler1035ver.pkl','rb') as f:
            sampler = pickle.load(f)
        self.idxs = sampler['idxs']

        with open('preprocessing/files/ver1103tri2110.pkl','rb') as f:
            sampler__ = pickle.load(f)
        trilist = sampler__['trilist']

        with open('preprocessing/files/lsfm_exp_30.dat','rb') as f:
            lsfm = serialize.deserialize_binary_to_morphable_model(f.read())

        # load face model data for estimating weights
        num_components = lsfm['components'].shape[0]
        c_fit = lsfm['components'].T.reshape((-1, 3, num_components))[self.idxs].reshape((-1, num_components))
        self.m_fit = lsfm['mean_points'].reshape((-1,3))[self.idxs].astype(np.float32)
        exp_basis = lsfm['components'].astype(np.float32)

        temp = io.loadmat(self.MM_inpath)['fmod']
        self.fmod = {'mean': temp['mean'][0][0], 'faces': temp['faces'][0][0],
                        'id_basis':temp['id_basis'][0][0],
                        'exp_basis': temp['exp_basis'][0][0],
                        'IDlands': (temp['IDlands'][0][0]-1).flatten()}

        # load the id basis of the LSFM model
        id_basis = io.loadmat(self.MM_inpath)['fmod']['id_basis'][0][0]

        # compute face ID from ID model weights
        faceID = lsfm['mean_points'].astype(np.float32)

        # initialize hephaestus renderer
        self.width = 256     # NMFC width hardcoded
        self.height = 256    # NMFC height hardcoded
        shaderDir = 'preprocessing/shaders'   # needs to point to the directory with the hephaestus shaders
        hephaestus.init_system(self.width, self.height, shaderDir)
        hephaestus.set_clear_color(0, 0, 0, 0)

        # create a model from the mean mesh of the LSFM
        self.model = hephaestus.create_NMFC(lsfm['mean_points'], lsfm['mean_indices'])
        hephaestus.setup_model(self.model)

    def reconstruct(self, image_pths):
        # Values to return
        cam_params, id_params, exp_params, landmarks5 = ([] for i in range(4))
        success = True
        # Perform 3D face reconstruction for each given frame.
        print('Performing face reconstruction')
        for image_pth in tqdm(image_pths):
            # Read current frame
            frame = cv2.imread(image_pth)
            # Check if frame was successfully read.
            if frame is None:
                print('Failed to read %s' % image_pth)
                success = False
                break
            # Check if dense landmarks were found and only one face exists in the image.
            handler_ret = self.handler.get(frame)
            if len(handler_ret) == 2:
                landmarks, lands5 = handler_ret[0], handler_ret[1]
            else:
                print('Failed to find a face in %s' % image_pth)
                success = False
                break
            if len(landmarks) != 1 or len(lands5) != 1:
                print('None or more than one faces were found in %s' % image_pth)
                success = False
                break
            # Perform fitting.
            pos_lms = landmarks[0][:-68].astype(np.float32)
            shape = pos_lms.copy() * np.array([1, -1, -1], dtype=np.float32) # landmark mesh is in left-handed system
            coefs = est_id_exp(self.fmod, shape, self.MM_inpath, self.idxs)
            Pinv = transform.estimate_affine_matrix_3d23d(self.m_fit, pos_lms).astype(np.float32)
            # Gather results
            cam_params.append(transform.P2sRt(Pinv)) # Scale, Rotation, Translation
            id_params.append(coefs[:157])            # Identity coefficients
            exp_params.append(coefs[157:])           # Expression coefficients
            landmarks5.append(lands5[0])             # Five facial landmarks
        # Return
        return success, (cam_params, id_params, exp_params, landmarks5)

    def computeNMFCs(self, cam_params, id_params, exp_params):
        nmfcs = []
        # Compute NMFCs from reconstruction parameters of frames.
        print('Computing NMFCs')
        for cam_param, id_param, exp_param in tqdm(zip(cam_params, id_params, exp_params), total=len(cam_params)):
            # Get Scale, Rotation, Translation
            S, R, T = cam_param
            # Compute face without pose.
            faceAll = self.fmod['mean'].ravel() + np.matmul(self.fmod['id_basis'],
                      id_param).ravel() + exp_param.dot(self.fmod['exp_basis'].T)
            # Compute face with pose.
            T = (T / S).reshape(3,1)
            posed_face3d = R.dot(faceAll.reshape(-1, 3).T) + T
            # Use hephaestus to generate the NMFC image.
            hephaestus.update_positions(self.model, posed_face3d.astype(np.float32).T.ravel())
            # setup orthographic projection and place the camera
            viewportWidth = self.width / S
            viewportHeight = self.height / S
            # seems the viewport is inverted for Vulkan, handle this by inverting the ortho projection
            hephaestus.set_orthographics_projection(self.model, viewportWidth * 0.5, -viewportWidth * 0.5,
                                                    -viewportHeight * 0.5, viewportHeight * 0.5, -10, 10)
            # set the cameara to look at the center of the mesh
            target = hephaestus.vec4(viewportWidth * 0.5, viewportHeight * 0.5, 0, 1)
            camera = hephaestus.vec4(viewportWidth * 0.5, viewportHeight * 0.5, -3, 1)
            hephaestus.set_camera_lookat(self.model, camera, target)
            data, channels, width, height = hephaestus.render_NMFC(self.model)
            data3D = data.reshape((height, width, channels))
            data3D = data3D[:,:,0:3]
            nmfcs.append(data3D[..., ::-1])
        return nmfcs

    def clear(self):
        # clean up
        hephaestus.clear_system()
