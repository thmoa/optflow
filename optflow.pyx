# -*- coding: utf-8 -*-

import numpy as np  # Import Python functions, attributes, submodules of numpy
cimport numpy as np  # Import numpy C/C++ API
from libcpp cimport bool
from cpython.ref cimport PyObject
from libc.stdlib cimport malloc, free

np.import_array()

cdef extern from 'src/pyopencv_converter.cpp':
    #mrc689 April 20,2017
    cdef PyObject*pyopencv_from(const Mat& m)
    cdef bool pyopencv_to(PyObject*o, Mat& m)

cdef extern from 'opencv2/core/core.hpp':
    cdef int CV_8UC1
    cdef int CV_32FC1
    cdef int CV_32FC2
    cdef int CV_32FC3

cdef extern from 'opencv2/core/core.hpp' namespace 'cv':
    cdef cppclass Mat:
        Mat() except +
        Mat(int, int, int) except +
        int cols
        int rows

cdef extern from 'opencv2/core/cvstd.hpp' namespace 'cv':
    cdef cppclass Ptr[T]:
        Ptr() except +
        T*get()

cdef extern from 'opencv2/core/cuda.hpp' namespace 'cv::cuda':
    cdef cppclass GpuMat:
        GpuMat() except +
        GpuMat(int, int, int) except +
        void upload(Mat arr) except +
        void download(Mat dst) const
        int cols
        int rows

cdef extern from 'opencv2/cudaoptflow.hpp' namespace 'cv::cuda':
    cdef cppclass BroxOpticalFlow:
        @ staticmethod
        Ptr[BroxOpticalFlow] create(double, double, double, int, int, int)

        void calc(GpuMat, GpuMat, GpuMat) except +

cdef extern from 'EPPM/bao_flow_patchmatch_multiscale_cuda.h':
    cdef cppclass bao_flow_patchmatch_multiscale_cuda:
        void init(int h, int w)
        void init(unsigned char** *img1, unsigned char** *img, int h, int w)
        void set_data(unsigned char** *img1, unsigned char** *img)
        void compute_flow(float** disp1_x, float** disp1_y) except +


cpdef eppm(np.ndarray[np.float64_t, ndim=3, mode="c"] im0, np.ndarray[np.float64_t, ndim=3, mode="c"] im1):
    """
    Calculates the optical flow between images im0 and im1 using the CUDA accelerated code of
    
    Bao, Linchao, Qingxiong Yang, and Hailin Jin. Fast edge-preserving patchmatch for large displacement optical flow.
    IEEE Conference on Computer Vision and Pattern Recognition. 2014.
    
    :param im0: Image 0 (NxMx3)
    :param im1: Image 1 (NxMx3)
    :return: Flow field (NxMx2)
    """

    assert im0.shape[0] == im1.shape[0] and im0.shape[1] == im1.shape[1]

    cdef np.ndarray[np.uint8_t, ndim=3] im0_u = (im0 * 255).astype(np.uint8)
    cdef np.ndarray[np.uint8_t, ndim=3] im1_u = (im1 * 255).astype(np.uint8)

    cdef unsigned int h = im0.shape[0]
    cdef unsigned int w = im0.shape[1]

    cdef np.ndarray[np.float32_t, ndim=2] u = np.zeros([h, w], dtype=np.float32)
    cdef np.ndarray[np.float32_t, ndim=2] v = np.zeros([h, w], dtype=np.float32)

    cdef bao_flow_patchmatch_multiscale_cuda eppm

    cdef float *u_data = <float *> u.data
    cdef float *v_data = <float *> v.data

    cdef float ** u_p
    cdef float ** v_p

    cdef unsigned char *im0_data = <unsigned char *> im0_u.data
    cdef unsigned char *im1_data = <unsigned char *> im1_u.data

    cdef unsigned char ** im0_p
    cdef unsigned char ** im1_p
    cdef unsigned char ** *im0_pp
    cdef unsigned char ** *im1_pp

    im0_p = <unsigned char **> malloc(h * w * sizeof(unsigned char *))
    im1_p = <unsigned char **> malloc(h * w * sizeof(unsigned char *))
    im0_pp = <unsigned char ** *> malloc(h * sizeof(unsigned char **))
    im1_pp = <unsigned char ** *> malloc(h * sizeof(unsigned char **))
    u_p = <float **> malloc(h * sizeof(float *))
    v_p = <float **> malloc(h * sizeof(float *))

    try:
        for i in range(h):
            for j in range(w):
                im0_p[i * w + j] = &im0_data[i * w * 3 + j * 3]
                im1_p[i * w + j] = &im1_data[i * w * 3 + j * 3]

            im0_pp[i] = &im0_p[i * w]
            im1_pp[i] = &im1_p[i * w]
            u_p[i] = &u_data[i * w]
            v_p[i] = &v_data[i * w]

        eppm.init(h, w)
        eppm.set_data(im0_pp, im1_pp)
        eppm.compute_flow(u_p, v_p)

    finally:
        free(im0_p)
        free(im1_p)
        free(im0_pp)
        free(im1_pp)
        free(u_p)
        free(v_p)

    return np.dstack((u, v))

cpdef brox(np.ndarray[np.float64_t, ndim=2, mode="c"] im0_d, np.ndarray[np.float64_t, ndim=2, mode="c"] im1_d,
           alpha = 0.197, gamma = 50.0, scale_factor = 0.8, inner_iterations = 5, outer_iterations = 150,
           solver_iterations = 10):
    """
    Calculates the optical flow between images im0 and im1 using the CUDA accelerated OpenCV code of
    
    Brox, Thomas, et al. High accuracy optical flow estimation based on a theory for warping.
    European conference on computer vision. 2004.
    
    :param im0: Image 0 (NxM)
    :param im1: Image 1 (NxM)
    :param alpha: alpha = 0.197
    :param gamma: gamma = 50.0
    :param scale_factor: scale_factor = 0.8
    :param inner_iterations: inner_iterations = 5
    :param outer_iterations: outer_iterations = 150
    :param solver_iterations: solver_iterations = 10
    :return: Flow field (NxMx2)
    """
    cdef np.ndarray[np.float32_t, ndim=2] im0 = im0_d.astype(np.float32)
    cdef np.ndarray[np.float32_t, ndim=2] im1 = im1_d.astype(np.float32)

    assert im0.shape[0] == im1.shape[0] and im0.shape[1] == im1.shape[1]

    cdef unsigned int h = im0.shape[0]
    cdef unsigned int w = im0.shape[1]

    cdef Mat m_im0
    cdef GpuMat g_im0
    cdef Mat m_im1
    cdef GpuMat g_im1

    cdef Mat flow = Mat(h, w, CV_32FC2)
    cdef GpuMat g_flow = GpuMat(h, w, CV_32FC2)

    pyopencv_to(<PyObject*> im0, m_im0)
    g_im0.upload(m_im0)
    pyopencv_to(<PyObject*> im1, m_im1)
    g_im1.upload(m_im1)

    cdef Ptr[BroxOpticalFlow] brox = BroxOpticalFlow.create(
        alpha, gamma, scale_factor, inner_iterations, outer_iterations, solver_iterations)

    brox.get().calc(g_im0, g_im1, g_flow)

    g_flow.download(flow)
    return <object> pyopencv_from(flow)
