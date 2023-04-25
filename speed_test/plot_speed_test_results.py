from matplotlib import pyplot as plt
from matplotlib.ticker import NullLocator
from matplotlib.ticker import ScalarFormatter
import numpy as np

insizes = 2 ** np.arange(8, 13)
outsizes = 2 ** np.arange(8, 13)

sp_time_array = np.load("sp_time_array.npy")
sn_time_array = np.load("sn_time_array.npy")
sc_time_array = np.load("sc_time_array.npy")
sp_time_array_with_err = np.load("sp_time_array_with_err.npy")
sn_time_array_with_err = np.load("sn_time_array_with_err.npy")
sc_time_array_with_err = np.load("sc_time_array_with_err.npy")


red_gradient = [
    (1, i / 5, i / 5) for i in range(5)
]  # R, G, B values for red gradient
blue_gradient = [
    (i / 5, i / 5, 1) for i in range(5)
]  # R, G, B values for blue gradient

fig, axs = plt.subplots(
    1, 2, figsize=(12, 8), sharex=True, sharey=False, gridspec_kw={"wspace": 0}
)
for j, (p, n, size) in enumerate(
    zip(sp_time_array, sn_time_array, outsizes)
):
    axs[0].plot(
        insizes,
        p / n,
        label=f"numba with output size: {size}",
        color=red_gradient[j],
    )

for j, (p, c, size) in enumerate(
    zip(sp_time_array, sc_time_array, outsizes)
):
    axs[0].plot(
        insizes,
        p / c,
        label=f"C extension with output size: {size}",
        color=blue_gradient[j],
    )

for k, (p_e, n_e, size) in enumerate(
    zip(
        sp_time_array_with_err,
        sn_time_array_with_err,
        outsizes,
    )
):
    axs[1].plot(
        insizes,
        p_e / n_e,
        color=red_gradient[k],
    )

for k, (p_e, c_e, size) in enumerate(
    zip(
        sp_time_array_with_err,
        sc_time_array_with_err,
        outsizes,
    )
):
    axs[1].plot(
        insizes,
        p_e / c_e,
        color=blue_gradient[k],
    )

axs[0].grid()
axs[1].grid()

axs[0].legend()

axs[0].xaxis.set_major_formatter(ScalarFormatter(useMathText=False))
axs[1].xaxis.set_major_formatter(ScalarFormatter(useMathText=False))

axs[0].set_xticks(outsizes)
axs[0].set_xticklabels(outsizes)
axs[1].set_xticks(outsizes)
axs[1].set_xticklabels(outsizes)

axs[0].xaxis.set_minor_locator(NullLocator())
axs[1].xaxis.set_minor_locator(NullLocator())

axs[0].set_xlim(150, 4250)
axs[1].set_xlim(150, 4250)

axs[0].set_ylim(0, 400)
axs[1].set_ylim(0, 400)

axs[1].set_yticklabels([])

axs[0].set_title('Unit-weighted')
axs[1].set_title('Error-weighted')

axs[0].set_ylabel('Performance gain (factor)')
fig.text(0.5, 0.02, 'Input size of the spectrum', ha='center')

plt.tight_layout()
plt.subplots_adjust(bottom=0.075)
plt.savefig("speed_test.png")
