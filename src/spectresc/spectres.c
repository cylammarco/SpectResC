#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <Python.h>
#include <numpy/arrayobject.h>

double *make_bins(double *wavs, int wavs_len)
{
    double *edges = (double *)malloc(sizeof(double) * (wavs_len + 1));

    edges[0] = wavs[0] - (wavs[1] - wavs[0]) / 2.0;
    edges[wavs_len] = wavs[wavs_len - 1] + (wavs[wavs_len - 1] - wavs[wavs_len - 2]) / 2.0;

    for (int i = 1; i < wavs_len; i++)
    {
        edges[i] = (wavs[i] + wavs[i - 1]) / 2.0;
    }

    return edges;
}

// Define the spectres function
static PyObject *spectres(PyObject *self, PyObject *args, PyObject *kwargs)
{

    PyObject *new_wavs_obj, *spec_wavs_obj, *spec_fluxes_obj, *spec_errs_obj = NULL;
    double fill = NAN;
    int verbose = 1;

    static char *kwlist[] = {"new_wavs", "spec_wavs", "spec_fluxes", "spec_errs", "fill", "verbose", NULL};

    /* Parse the input tuple */
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OOO|Odp:spectres", kwlist, &new_wavs_obj, &spec_wavs_obj, &spec_fluxes_obj, &spec_errs_obj,
                                     &fill, &verbose))
    {
        return NULL;
    }

    // Convert input object to NumPy array
    PyArrayObject *_new_wavs_array = (PyArrayObject *)PyArray_FROM_OTF(new_wavs_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *_spec_wavs_array = (PyArrayObject *)PyArray_FROM_OTF(spec_wavs_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *_spec_fluxes_array = (PyArrayObject *)PyArray_FROM_OTF(spec_fluxes_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);

    // Check if input array is contiguous
    npy_intp *new_wavs_strides = PyArray_STRIDES(_new_wavs_array);
    npy_intp *spec_wavs_strides = PyArray_STRIDES(_spec_wavs_array);
    npy_intp *spec_fluxes_strides = PyArray_STRIDES(_spec_fluxes_array);
    int new_wavs_is_contiguous = (new_wavs_strides[PyArray_NDIM(_new_wavs_array) - 1] == sizeof(double));
    int spec_wavs_is_contiguous = (spec_wavs_strides[PyArray_NDIM(_spec_wavs_array) - 1] == sizeof(double));
    int spec_fluxes_is_contiguous = (spec_fluxes_strides[PyArray_NDIM(_spec_fluxes_array) - 1] == sizeof(double));

    // Create new array if input array is not contiguous
    PyArrayObject *new_wavs_array;
    PyArrayObject *spec_wavs_array;
    PyArrayObject *spec_fluxes_array;

    // handle the new_wavs_array
    if (!new_wavs_is_contiguous)
    {
        npy_intp *new_wavs_shape = PyArray_SHAPE(_new_wavs_array);
        new_wavs_array = (PyArrayObject *)PyArray_SimpleNew(PyArray_NDIM(_new_wavs_array), new_wavs_shape, NPY_DOUBLE);
        if (new_wavs_array == NULL)
        {
            Py_DECREF(_new_wavs_array);
            return NULL;
        }

        // Copy data from input array into output array
        char *src = (char *)PyArray_DATA(_new_wavs_array);
        char *dst = (char *)PyArray_DATA(new_wavs_array);
        npy_intp size = PyArray_SIZE(_new_wavs_array) * sizeof(short);
        memcpy(dst, src, size);
    }
    else
    {
        // Use input array directly
        Py_INCREF(_new_wavs_array);
        new_wavs_array = _new_wavs_array;
    }

    // handle the spec_wavs_array
    if (!spec_wavs_is_contiguous)
    {
        npy_intp *spec_wavs_shape = PyArray_SHAPE(_spec_wavs_array);
        spec_wavs_array = (PyArrayObject *)PyArray_SimpleNew(PyArray_NDIM(_spec_wavs_array), spec_wavs_shape, NPY_DOUBLE);
        if (spec_wavs_array == NULL)
        {
            Py_DECREF(_spec_wavs_array);
            return NULL;
        }

        // Copy data from input array into output array
        char *src = (char *)PyArray_DATA(_spec_wavs_array);
        char *dst = (char *)PyArray_DATA(spec_wavs_array);
        npy_intp size = PyArray_SIZE(_spec_wavs_array) * sizeof(short);
        memcpy(dst, src, size);
    }
    else
    {
        // Use input array directly
        Py_INCREF(_spec_wavs_array);
        spec_wavs_array = _spec_wavs_array;
    }

    // handle the spec_fluxes_array
    if (!spec_fluxes_is_contiguous)
    {
        npy_intp *spec_fluxes_shape = PyArray_SHAPE(_spec_fluxes_array);
        spec_fluxes_array = (PyArrayObject *)PyArray_SimpleNew(PyArray_NDIM(_spec_fluxes_array), spec_fluxes_shape, NPY_DOUBLE);
        if (spec_fluxes_array == NULL)
        {
            Py_DECREF(_spec_fluxes_array);
            return NULL;
        }

        // Copy data from input array into output array
        char *src = (char *)PyArray_DATA(_spec_fluxes_array);
        char *dst = (char *)PyArray_DATA(spec_fluxes_array);
        npy_intp size = PyArray_SIZE(_spec_fluxes_array) * sizeof(short);
        memcpy(dst, src, size);
    }
    else
    {
        // Use input array directly
        Py_INCREF(_spec_fluxes_array);
        spec_fluxes_array = _spec_fluxes_array;
    }

    double *new_wavs = (double *)PyArray_DATA(new_wavs_array);
    double *spec_wavs = (double *)PyArray_DATA(spec_wavs_array);
    double *spec_fluxes = (double *)PyArray_DATA(spec_fluxes_array);

    // Get the length of the input arrays
    int new_wavs_len = (int)PyArray_DIM(new_wavs_array, 0);
    int spec_wavs_len = (int)PyArray_DIM(spec_wavs_array, 0);

    double *spec_edges, *new_edges, *spec_widths;
    spec_edges = make_bins(spec_wavs, spec_wavs_len);
    new_edges = make_bins(new_wavs, new_wavs_len);
    spec_widths = (double *)malloc(sizeof(double) * spec_wavs_len);
    for (int i = 0; i < spec_wavs_len; i++)
    {
        spec_widths[i] = spec_edges[i + 1] - spec_edges[i];
    }

    // Create empty arrays for populating resampled flux and error
    double *new_fluxes = (double *)malloc(sizeof(double) * new_wavs_len);

    /* Get pointers to the data as C-types. */
    PyObject *spec_errs_array = NULL;
    double *spec_errs = NULL;
    double *new_errs = NULL;

    if (spec_errs_obj != NULL)
    {
        PyArrayObject *_spec_errs_array = (PyArrayObject *)PyArray_FROM_OTF(spec_errs_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);

        // Check if input array is contiguous
        npy_intp *spec_errs_strides = PyArray_STRIDES(_spec_errs_array);
        int spec_errs_is_contiguous = (spec_errs_strides[PyArray_NDIM(_spec_errs_array) - 1] == sizeof(double));

        // Create new array if input array is not contiguous
        PyArrayObject *spec_errs_array;
        if (!spec_errs_is_contiguous)
        {
            npy_intp *spec_errs_shape = PyArray_SHAPE(_spec_errs_array);
            spec_errs_array = (PyArrayObject *)PyArray_SimpleNew(PyArray_NDIM(_spec_errs_array), spec_errs_shape, NPY_DOUBLE);
            if (spec_errs_array == NULL)
            {
                Py_DECREF(_spec_errs_array);
                return NULL;
            }

            // Copy data from input array into output array
            char *src = (char *)PyArray_DATA(_spec_errs_array);
            char *dst = (char *)PyArray_DATA(spec_errs_array);
            npy_intp size = PyArray_SIZE(_spec_errs_array) * sizeof(short);
            memcpy(dst, src, size);
        }
        else
        {
            // Use input array directly
            Py_INCREF(_spec_errs_array);
            spec_errs_array = _spec_errs_array;
        }
        spec_errs = (double *)PyArray_DATA(spec_errs_array);

        new_errs = (double *)malloc(sizeof(double) * new_wavs_len);
    }

    int start = 0, stop = 0, warned = 0;
    for (int i = 0; i < new_wavs_len; i++)
    {
        if (new_edges[i] < spec_edges[0] || new_edges[i + 1] > spec_edges[spec_wavs_len])
        {
            new_fluxes[i] = fill;
            if (spec_errs != NULL)
            {
                new_errs[i] = fill;
            }
            if ((i == 0 || i == new_wavs_len - 1) && verbose && !warned)
            {
                warned = 1;
                printf("SpectResC: new_wavs contains values outside the range "
                       "in spec_wavs, new_fluxes and new_errs will be filled "
                       "with the value set in the 'fill' keyword argument.\n");
            }
        }
        else
        {
            while (spec_edges[start + 1] <= new_edges[i])
            {
                start += 1;
            }

            while (spec_edges[stop + 1] < new_edges[i + 1])
            {
                stop += 1;
            }

            if (stop == start)
            {
                new_fluxes[i] = spec_fluxes[start];
                if (spec_errs != NULL)
                {
                    new_errs[i] = spec_errs[start];
                }
            }
            else
            {
                double start_factor = ((spec_edges[start + 1] - new_edges[i]) / (spec_edges[start + 1] - spec_edges[start]));
                double end_factor = ((new_edges[i + 1] - spec_edges[stop]) / (spec_edges[stop + 1] - spec_edges[stop]));

                spec_widths[start] *= start_factor;
                spec_widths[stop] *= end_factor;

                // Populate new_fluxes spectrum and uncertainty arrays
                double f_widths_sum = 0.0;
                double spec_widths_sum = 0.0;
                double e_wid_sum = 0.0;
                for (int j = start; j <= stop; j++)
                {
                    f_widths_sum += spec_widths[j] * spec_fluxes[j];
                    spec_widths_sum += spec_widths[j];
                    if (spec_errs != NULL)
                    {
                        e_wid_sum += pow(spec_widths[j] * spec_errs[j], 2);
                    }
                }

                new_fluxes[i] = f_widths_sum / spec_widths_sum;

                if (spec_errs != NULL)
                {
                    new_errs[i] = sqrt(e_wid_sum) / spec_widths_sum;
                }

                // Put back the old bin widths to their initial values
                spec_widths[start] /= start_factor;
                spec_widths[stop] /= end_factor;
            }
        }
    }

    // Free the memory
    free(spec_edges);
    free(new_edges);
    free(spec_widths);

    // Create NumPy arrays to return the data to Python
    npy_intp dims[1] = {new_wavs_len};
    PyObject *new_fluxes_array = PyArray_SimpleNewFromData(1, dims, NPY_DOUBLE, new_fluxes);
    PyArray_ENABLEFLAGS((PyArrayObject *)new_fluxes_array, NPY_ARRAY_OWNDATA);

    // Set the base object for the arrays to NULL to indicate that the arrays are not owned by Python
    // and reate a tuple to return the array(s)
    if (spec_errs != NULL)
    {
        PyObject *new_errs_array = NULL;
        PyObject *result_list = PyList_New(0);
        PyList_Append(result_list, PyArray_Return(new_fluxes_array));
        new_errs_array = PyArray_SimpleNewFromData(1, dims, NPY_DOUBLE, new_errs);
        PyArray_ENABLEFLAGS((PyArrayObject *)new_errs_array, NPY_ARRAY_OWNDATA);
        PyList_Append(result_list, PyArray_Return(new_errs_array));
        return result_list;
    }
    else
    {
        return new_fluxes_array;
    }
}

// Define the module methods
static PyMethodDef SpectrescMethods[] = {
    {"spectres", spectres, METH_VARARGS | METH_KEYWORDS, "Resample a spectrum onto a new wavelength grid."},
    {NULL, NULL, 0, NULL}};

// Define the module structure
static struct PyModuleDef spectrescmodule = {
    PyModuleDef_HEAD_INIT,
    "spectresc",                                              // Module name
    "Python extension module for the spectres function in C", // Module description
    -1,
    SpectrescMethods};

// Define the module initialization function
PyMODINIT_FUNC PyInit_spectresc(void)
{
    import_array(); // Initialize NumPy
    return PyModule_Create(&spectrescmodule);
}