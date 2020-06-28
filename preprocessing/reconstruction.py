import cv2
import os
import numpy as np
import pickle
from preprocessing import transform
import sys
import scipy.io as io
import glob
from scipy import optimize
from tqdm import tqdm
from preprocessing.multiface import fc_predictor
from preprocessing.avatars import serialize
from preprocessing.hephaestus import hephaestus_bindings as hephaestus

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

        with open('preprocessing/files/lsfm_exp_30.dat','rb') as f:
            lsfm = serialize.deserialize_binary_to_morphable_model(f.read())

        # load face model data for estimating weights
        num_components = lsfm['components'].shape[0]
        self.c_fit = lsfm['components'].T.reshape((-1, 3, num_components))[self.idxs].reshape((-1, num_components))
        self.m_fit = lsfm['mean_points'].reshape((-1,3))[self.idxs].astype(np.float32)
        self.faceID = lsfm['mean_points'].astype(np.float32)
        self.exp_basis = lsfm['components'].astype(np.float32)
        self.stds = lsfm['weights']

        self.id_basis = io.loadmat(self.MM_inpath)['fmod']['id_basis'][0][0]

        temp = io.loadmat(self.MM_inpath)['fmod']
        self.fmod = {'mean': temp['mean'][0][0], 'faces': temp['faces'][0][0],
                        'id_basis':temp['id_basis'][0][0],
                        'exp_basis': temp['exp_basis'][0][0],
                        'IDlands': (temp['IDlands'][0][0]-1).flatten()}

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
        n_consecutive_fails_threshold = 5 # hardcoded
        # Values to return
        cam_params, id_params, exp_params, landmarks5 = ([] for i in range(4))
        success = True
        handler_ret_prev = None
        n_consecutive_fails = 0
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
                # Face(s) found in frame.
                n_consecutive_fails = 0
                landmarks, lands5 = handler_ret[0], handler_ret[1]
                if len(landmarks) > 1:
                    print('More than one faces were found in %s' % image_pth)
                    landmarks, lands5 = landmarks[0:1], lands5[0:1]
            else:
                # Face not found in frame.
                n_consecutive_fails += 1
                print('Failed to find a face in %s (%d times in a row)' % (image_pth, n_consecutive_fails))
                if handler_ret_prev is None or n_consecutive_fails > n_consecutive_fails_threshold:
                    success = False
                    break
                else:
                    # Recover using previous landmarks
                    handler_ret = handler_ret_prev
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
            handler_ret_prev = handler_ret
        # Return
        return success, (cam_params, id_params, exp_params, landmarks5)

    def get_expression_and_pose(self, image):
        # Perform 3D face reconstruction for given frame.
        handler_ret = self.handler.get(image)
        if len(handler_ret) == 2:
            success = True
            # Face(s) found in frame.
            landmarks, lands5 = handler_ret[0], handler_ret[1]
            if len(landmarks) > 1:
                print('More than one faces were found in image')
                landmarks, lands5 = landmarks[0:1], lands5[0:1]
        else:
            # Face not found in frame.
            success = False
            print('Failed to find a face in image')
            return success, None, None
        # Perform fitting.
        pos_lms = landmarks[0][:-68].astype(np.float32)
        shape = pos_lms.copy() * np.array([1, -1, -1], dtype=np.float32) # landmark mesh is in left-handed system

        P = transform.estimate_affine_matrix_3d23d(shape, self.m_fit).astype(np.float32)
        position_model_space = np.concatenate((shape, np.ones((shape.shape[0], 1), dtype=np.float32)), axis=1) @ P.T
        position_model_space -= self.m_fit
        singular_value_cut_ratio=0.3
        result = np.linalg.lstsq(self.c_fit, position_model_space.reshape(-1), rcond=singular_value_cut_ratio)
        weights = result[0]
        exp_params = weights / self.stds
        Pinv = transform.estimate_affine_matrix_3d23d(self.m_fit, pos_lms).astype(np.float32)
        cam_params = transform.P2sRt(Pinv)
        # Return
        return success, exp_params, cam_params

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
            nmfcs.append(data3D[..., ::-1]) # return BGR
        return nmfcs

    def computeNMFC(self, cam_param, id_param, exp_param):
        # Compute NMFCs from frame. Used for demo.
        # Get Scale, Rotation, Translation
        S, R, T = cam_param
        faceAll = self.faceID + np.matmul(self.id_basis, id_param).ravel() + exp_param.dot(self.exp_basis)
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
        return data3D # return RGB

    def clear(self):
        # clean up
        hephaestus.clear_system()
