#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <Python.h>
#include <numpy/arrayobject.h>

void make_bins(double *wavs, int wavs_len, double *edges, double *widths)
{
    edges[0] = wavs[0] - (wavs[1] - wavs[0]) / 2.0;
    edges[wavs_len] = wavs[wavs_len - 1] + (wavs[wavs_len - 1] - wavs[wavs_len - 2]) / 2.0;

    for (int i = 1; i < wavs_len; i++)
    {
        edges[i] = (wavs[i] + wavs[i - 1]) / 2.0;
    }

    for (int i = 0; i < wavs_len - 1; i++)
    {
        widths[i] = edges[i + 1] - edges[i];
    }
    widths[wavs_len - 1] = edges[wavs_len] - edges[wavs_len - 1];
}

// Define the spectres function
static PyObject *spectres(PyObject *self, PyObject *args, PyObject *kwargs)
{
    PyObject *new_wavs_obj, *spec_wavs_obj, *spec_fluxes_obj, *spec_errs_obj = NULL;
    double fill = NAN;
    int verbose = 1;

    static char *kwlist[] = {"new_wavs", "spec_wavs", "spec_fluxes", "spec_errs", "fill", "verbose", NULL};

    /* Parse the input tuple */
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OOO|Odp:spectres", kwlist, &new_wavs_obj, &spec_wavs_obj, &spec_fluxes_obj, &spec_errs_obj, &fill, &verbose))
    {
        return NULL;
    }

    // Convert input object to NumPy array
    PyArrayObject *new_wavs_array = (PyArrayObject *)PyArray_FROM_OTF(new_wavs_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *spec_wavs_array = (PyArrayObject *)PyArray_FROM_OTF(spec_wavs_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *spec_fluxes_array = (PyArrayObject *)PyArray_FROM_OTF(spec_fluxes_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);

    if (!new_wavs_array || !spec_wavs_array || !spec_fluxes_array)
    {
        Py_XDECREF(new_wavs_array);
        Py_XDECREF(spec_wavs_array);
        Py_XDECREF(spec_fluxes_array);
        return NULL;
    }

    double *new_wavs = (double *)PyArray_DATA(new_wavs_array);
    double *spec_wavs = (double *)PyArray_DATA(spec_wavs_array);
    double *spec_fluxes = (double *)PyArray_DATA(spec_fluxes_array);

    int new_wavs_len = (int)PyArray_DIM(new_wavs_array, 0);
    int spec_wavs_len = (int)PyArray_DIM(spec_wavs_array, 0);

    /* --- Determine flux array dimensions --- */
    int ndim = PyArray_NDIM(spec_fluxes_array);
    npy_intp *flux_shape = PyArray_DIMS(spec_fluxes_array);

    if (flux_shape[ndim - 1] != spec_wavs_len)
    {
        PyErr_SetString(PyExc_ValueError, "Last dimension of spec_fluxes must match length of spec_wavs.");
        return NULL;
    }

    int num_spectra = 1;
    for (int i = 0; i < ndim - 1; i++)
    {
        num_spectra *= flux_shape[i];
    }

    /* --- Make bins --- */
    double *spec_edges = malloc((spec_wavs_len + 1) * sizeof(double));
    double *spec_widths = malloc(spec_wavs_len * sizeof(double));
    make_bins(spec_wavs, spec_wavs_len, spec_edges, spec_widths);

    double *new_edges = malloc((new_wavs_len + 1) * sizeof(double));
    double *new_widths = malloc(new_wavs_len * sizeof(double));
    make_bins(new_wavs, new_wavs_len, new_edges, new_widths);

    /* --- Handle optional errors --- */
    PyArrayObject *spec_errs_array = NULL;
    double *spec_errs = NULL;

    if (spec_errs_obj != NULL && spec_errs_obj != Py_None)
    {
        spec_errs_array = (PyArrayObject *)PyArray_FROM_OTF(spec_errs_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
        if (!spec_errs_array)
        {
            Py_XDECREF(new_wavs_array);
            Py_XDECREF(spec_wavs_array);
            Py_XDECREF(spec_fluxes_array);
            return NULL;
        }

        npy_intp *err_shape = PyArray_DIMS(spec_errs_array);
        for (int i = 0; i < ndim; i++)
        {
            if (err_shape[i] != flux_shape[i])
            {
                PyErr_SetString(PyExc_ValueError, "spec_errs must have the same shape as spec_fluxes.");
                return NULL;
            }
        }

        spec_errs = (double *)PyArray_DATA(spec_errs_array);
    }

    /* --- Create output arrays --- */
    npy_intp *new_flux_shape = malloc(ndim * sizeof(npy_intp));
    memcpy(new_flux_shape, flux_shape, ndim * sizeof(npy_intp));
    new_flux_shape[ndim - 1] = new_wavs_len;

    PyArrayObject *new_fluxes_array = (PyArrayObject *)PyArray_SimpleNew(ndim, new_flux_shape, NPY_DOUBLE);
    double *new_fluxes_data = (double *)PyArray_DATA(new_fluxes_array);

    PyArrayObject *new_errs_array = NULL;
    double *new_errs_data = NULL;
    if (spec_errs != NULL)
    {
        new_errs_array = (PyArrayObject *)PyArray_SimpleNew(ndim, new_flux_shape, NPY_DOUBLE);
        new_errs_data = (double *)PyArray_DATA(new_errs_array);
    }

    free(new_flux_shape);

    /* --- Loop over each spectrum --- */
    for (int s = 0; s < num_spectra; s++)
    {
        double *this_flux = spec_fluxes + s * spec_wavs_len;
        double *this_err = spec_errs != NULL ? spec_errs + s * spec_wavs_len : NULL;
        double *out_flux = new_fluxes_data + s * new_wavs_len;
        double *out_err = spec_errs != NULL ? new_errs_data + s * new_wavs_len : NULL;

        int start = 0, stop = 0, warned = 0;

        for (int i = 0; i < new_wavs_len; i++)
        {
            if (new_edges[i] < spec_edges[0] || new_edges[i + 1] > spec_edges[spec_wavs_len])
            {
                out_flux[i] = fill;
                if (out_err != NULL)
                    out_err[i] = fill;

                if ((i == 0 || i == new_wavs_len - 1) && verbose && !warned)
                {
                    warned = 1;
                    PyErr_WarnEx(PyExc_RuntimeWarning,
                                 "SpectResC: new_wavs contains values outside the range in spec_wavs; filled with 'fill'.",
                                 1);
                }
                continue;
            }

            while (spec_edges[start + 1] <= new_edges[i])
                start++;
            while (spec_edges[stop + 1] < new_edges[i + 1])
                stop++;

            if (stop == start)
            {
                out_flux[i] = this_flux[start];
                if (out_err != NULL)
                    out_err[i] = this_err[start];
            }
            else
            {
                double start_factor = ((spec_edges[start + 1] - new_edges[i]) / (spec_edges[start + 1] - spec_edges[start]));
                double end_factor = ((new_edges[i + 1] - spec_edges[stop]) / (spec_edges[stop + 1] - spec_edges[stop]));

                double f_widths_sum = 0.0;
                double spec_widths_sum = 0.0;
                double e_wid_sum = 0.0;

                for (int j = start; j <= stop; j++)
                {
                    double weight = spec_widths[j];
                    if (j == start)
                        weight *= start_factor;
                    if (j == stop)
                        weight *= end_factor;

                    f_widths_sum += weight * this_flux[j];
                    spec_widths_sum += weight;
                    if (out_err != NULL)
                        e_wid_sum += pow(weight * this_err[j], 2);
                }

                out_flux[i] = f_widths_sum / spec_widths_sum;
                if (out_err != NULL)
                    out_err[i] = sqrt(e_wid_sum) / spec_widths_sum;
            }
        }
    }

    free(spec_edges);
    free(spec_widths);
    free(new_edges);
    free(new_widths);

    Py_XDECREF(new_wavs_array);
    Py_XDECREF(spec_wavs_array);
    Py_XDECREF(spec_fluxes_array);
    Py_XDECREF(spec_errs_array);

    if (spec_errs != NULL)
    {
        PyObject *result_list = PyList_New(0);
        PyList_Append(result_list, (PyObject *)new_fluxes_array);
        PyList_Append(result_list, (PyObject *)new_errs_array);
        return result_list;
    }
    else
    {
        return (PyObject *)new_fluxes_array;
    }
}

// Define the module methods
static PyMethodDef SpectresMethods[] = {
    {"spectres", (PyCFunction)spectres, METH_VARARGS | METH_KEYWORDS, "Resample a spectrum onto a new wavelength grid."},
    {NULL, NULL, 0, NULL}};

// Define the module structure
static struct PyModuleDef spectresmodule = {
    PyModuleDef_HEAD_INIT,
    "spectres",                                               // Submodule name
    "Python extension module for the spectres function in C", // Module description
    -1,
    SpectresMethods};

// Define the module initialization function
PyMODINIT_FUNC PyInit_spectresc(void)
{
    import_array(); // Initialize NumPy
    return PyModule_Create(&spectresmodule);
}
