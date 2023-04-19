#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <Python.h>
#include <numpy/arrayobject.h>

double *make_bins(double *wavs, int wavs_len)
{
    double *edges = (double *)malloc(sizeof(double) * (wavs_len + 1));
    double *widths = (double *)malloc(sizeof(double) * wavs_len);

    edges[0] = wavs[0] - (wavs[1] - wavs[0]) / 2;
    widths[wavs_len - 1] = (wavs[wavs_len - 1] - wavs[wavs_len - 2]);
    edges[wavs_len] = wavs[wavs_len - 1] + (wavs[wavs_len - 1] - wavs[wavs_len - 2]) / 2;

    for (int i = 1; i < wavs_len; i++)
    {
        edges[i] = (wavs[i] + wavs[i - 1]) / 2;
        widths[i - 1] = edges[i] - edges[i - 1];
    }
    return edges;
}

// Define the spectres function
static PyObject *spectres(PyObject *self, PyObject *args)
{

    PyObject *new_wavs_obj, *spec_wavs_obj, *spec_fluxes_obj;
    PyObject *spec_errs_obj = NULL;
    double fill = 0.0;
    int verbose = 0;

    /* Parse the input tuple */
    if (!PyArg_ParseTuple(args, "OOO|OdO:spectres", &new_wavs_obj, &spec_wavs_obj, &spec_fluxes_obj, &spec_errs_obj,
                          &fill, &verbose))
    {
        return NULL;
    }
    // printf("\nSpectResC: Initialising.");

    /* Interpret the input objects as numpy arrays. */
    PyObject *new_wavs_array = PyArray_FROM_OTF(new_wavs_obj, NPY_DOUBLE, NPY_IN_ARRAY);
    PyObject *spec_wavs_array = PyArray_FROM_OTF(spec_wavs_obj, NPY_DOUBLE, NPY_IN_ARRAY);
    PyObject *spec_fluxes_array = PyArray_FROM_OTF(spec_fluxes_obj, NPY_DOUBLE, NPY_IN_ARRAY);
    PyObject *spec_errs_array = NULL;
    if (spec_errs_obj != NULL)
    {
        spec_errs_array = PyArray_FROM_OTF(spec_errs_obj, NPY_DOUBLE, NPY_IN_ARRAY);
        // printf("spec_errs_obj is not NULL.");
    }

    // Get the length of the input arrays
    int new_wavs_len = (int)PyArray_DIM(new_wavs_array, 0);
    // printf("\nSpectResC: new_wavs have lengths %d.", new_wavs_len);

    int spec_wavs_len = (int)PyArray_DIM(spec_wavs_array, 0);
    // printf("\nSpectResC: spec_wavs have lengths %d.", spec_wavs_len);

    /* Get pointers to the data as C-types. */
    double *new_wavs = (double *)PyArray_DATA(new_wavs_array);
    double *spec_wavs = (double *)PyArray_DATA(spec_wavs_array);
    double *spec_fluxes = (double *)PyArray_DATA(spec_fluxes_array);
    double *spec_errs = NULL;
    double *new_errs = NULL;
    if (spec_errs_array != NULL)
    {
        // printf("spec_errs_array is not NULL.");
        spec_errs = (double *)PyArray_DATA(spec_errs_array);
        new_errs = (double *)malloc(sizeof(double) * new_wavs_len);
    }
    // for (int i = 0; i < spec_wavs_len; i++)
    //{
    //      printf("\nspec_wavs: %f", spec_wavs[i]);
    //      printf("\nspec_fluxes: %f", spec_fluxes[i]);
    //     if (spec_errs_array != NULL) {
    //         printf("\nspec_errs: %f", spec_errs[i]);
    //     }
    // }
    //
    //  printf("\nfill = %f, ", fill);
    //  printf("\nverbose = %i, ", verbose);

    double *spec_edges, *spec_widths, *new_edges, *new_widths;
    spec_edges = make_bins(spec_wavs, spec_wavs_len);
    spec_widths = (double *)malloc(sizeof(double) * spec_wavs_len);
    for (int i = 0; i < spec_wavs_len; i++)
    {
        spec_widths[i] = spec_edges[i + 1] - spec_edges[i];
    }

    // Create empty arrays for populating resampled flux and error
    double *new_fluxes = (double *)malloc(sizeof(double) * new_wavs_len);
    new_edges = make_bins(new_wavs, new_wavs_len);
    new_widths = (double *)malloc(sizeof(double) * new_wavs_len);
    for (int i = 0; i < new_wavs_len; i++)
    {
        new_widths[i] = new_edges[i + 1] - new_edges[i];
    }

    int start = 0, stop = 0, warned = 0;
    for (int i = 0; i < new_wavs_len; i++)
    {
        if (new_edges[i] < spec_edges[0] || new_edges[i + 1] > spec_edges[spec_wavs_len])
        {
            new_fluxes[i] = fill;
            // printf("\nInside stop == start = %i: %f", i, new_fluxes[i]);
            if (spec_errs != NULL)
            {
                new_errs[i] = fill;
                // printf("\nWhen new wavelength range is too large i = %i, new_errs[i] = %f", i, new_errs[i]);
            }
            if ((i == 0 || i == new_wavs_len - 1) && verbose && !warned)
            {
                warned = 1;
                // printf("\nSpectres: new_wavs contains values outside the range in spec_wavs, new_fluxes and new_errs will be filled with the value set in the 'fill' keyword argument. \n");
            }
            continue;
        }

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
            // printf("\nInside stop == start = %i: %f", i, new_fluxes[i]);
            if (spec_errs != NULL)
            {
                new_errs[i] = spec_errs[start];
                // printf("\nInside stop == start = i = %i, new_errs[i] = %f", i, new_errs[i]);
            }
        }
        else
        {
            double start_factor = ((spec_edges[start + 1] - new_edges[i]) / (spec_edges[start + 1] - spec_edges[start]));
            double end_factor = ((new_edges[i + 1] - spec_edges[stop]) / (spec_edges[stop + 1] - spec_edges[stop]));
            // printf("\nstart = %i", start);
            // printf("\nstart_factor = %f", start_factor);
            // printf("\nstop = %i", stop);
            // printf("\nend_factor = %f", end_factor);
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
            // printf("\nIn the else case with i = %i: %f", i, new_fluxes[i]);

            if (spec_errs != NULL)
            {
                new_errs[i] = sqrt(e_wid_sum) / spec_widths_sum;
                // printf("\nIn the else case with i = %i, new_errs[i] = %f", i, new_errs[i]);
            }

            // Put back the old bin widths to their initial values
            spec_widths[start] /= start_factor;
            spec_widths[stop] /= end_factor;
        }
    }

    // Free the memory
    free(spec_edges);
    free(spec_widths);
    free(new_edges);
    free(new_widths);

    // Create NumPy arrays to return the data to Python
    npy_intp dims[1] = {new_wavs_len};
    PyObject *new_fluxes_array = PyArray_SimpleNewFromData(1, dims, NPY_DOUBLE, new_fluxes);
    PyArray_ENABLEFLAGS((PyArrayObject *)new_fluxes_array, NPY_ARRAY_OWNDATA);

    // Set the base object for the arrays to NULL to indicate that the arrays are not owned by Python
    PyObject *new_errs_array = NULL;
    if (spec_errs_obj != NULL)
    {
        new_errs_array = PyArray_SimpleNewFromData(1, dims, NPY_DOUBLE, new_errs);
        PyArray_ENABLEFLAGS((PyArrayObject *)new_errs_array, NPY_ARRAY_OWNDATA);
    }

    // Create a tuple to return the arrays as a pair
    PyObject *result = NULL;
    if (spec_errs_obj != NULL)
    {
        result = PyTuple_New(2);
    }
    else
    {
        result = PyTuple_New(1);
    }
    PyTuple_SetItem(result, 0, new_fluxes_array);
    if (spec_errs_obj != NULL)
    {
        PyTuple_SetItem(result, 1, new_errs_array);
    }
    return result;
}

// Define the module methods
static PyMethodDef SpectrescMethods[] = {
    {"spectres", spectres, METH_VARARGS, "Resample a spectrum onto a new wavelength grid."},
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