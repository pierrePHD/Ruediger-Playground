import matplotlib
from matplotlib import rc
import matplotlib.cm as cm
from matplotlib.pyplot import colorbar
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from const import msol, asymb
from gadget import gadget_readsnap
from loaders import load_species
from sfigure import sfig

rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
rc('text', usetex=True)
sns.set_palette(sns.husl_palette(7, l=.5))


def main():
    import csv

    simulation_name = "Bin_1100_WDEC_0350"
    sp = load_species("./arepo-snap-util/eostable/species55.txt")
    snap = 322

    #
    s = gadget_readsnap(snap, snappath="/scratch/fm5/ub0692/20220330-merger_threedium/2-resolution/output/",
                        loadonlytype=[0], hdf5=True)

    print("TIME: %gs." % s.time)
    print("ALL MASS WD1/WD2/TOTAL:", (s.mass * s.pass00).sum() / msol, (s.mass * s.pass01).sum() / msol,
          s.mass.sum() / msol)

    i, = np.where(s.rho > 1e2)
    vel_bound = (s.vel[i, :] * s.mass.astype('f8')[i, None]).sum(axis=0) / s.mass[i].astype('f8').sum()
    unbound, = np.where((s.pot + 0.5 * ((s.vel - vel_bound[None, :]) ** 2.).sum(axis=1) + s.u > 0.))
    bound, = np.where((s.pot + 0.5 * ((s.vel - vel_bound[None, :]) ** 2.).sum(axis=1) + s.u <= 0.))

    # unbound, = np.where((s.pot + 0.5 * (s.vel ** 2.).sum(axis=1) + s.u > 0.))
    # bound, = np.where((s.pot + 0.5 * (s.vel** 2.).sum(axis=1) + s.u <= 0.))

    print("MASS BOUND/UNBOUND/TOTAL:", s.mass[bound].sum() / msol, s.mass[unbound].sum() / msol, s.mass.sum() / msol)

    s.mass[unbound] = 0
    s.rho[unbound] = 0
    s.vol[unbound] = 0

    center = (s.cmce * s.mass[:, None]).sum(axis=0) / s.mass.sum()
    radius = np.sqrt(((s.cmce - center[None, :]) ** 2).sum(axis=1))

    vel = (s.vel * s.mass[:, None]).sum(axis=0) / s.mass.sum()
    print(snap, s.time, center - 0.5 * s.boxsize, s.mass.sum() / msol, s.rho.min(), s.rho.max())
    print("KICK VEL [km/s]: ", vel / 1e5, np.sqrt((vel ** 2).sum()) / 1e5)

    nbins = 1001
    isort = np.argsort(radius)
    mcum = np.cumsum(s.mass.astype('f8')[isort])
    mtot = 0.99 * mcum[-1]
    bins = np.zeros(nbins)

    ibin = 1
    icum = 0
    while ibin < nbins:
        while icum < len(mcum) and mcum[icum] < mtot * ibin / nbins:
            icum += 1
        bins[ibin] = radius[isort][icum]
        ibin += 1

    print("Computing histograms...")

    # nshells: %d
    # nspecies: %d
    # ncolumns: %d = nspecuies + 7
    # mtot: %g
    # mtot (x_??): %g
    # r_outer, mass, density, l_z, u, p, T, x_??
    # ...

    nshells = nbins - 1
    nspecies = sp['count']
    ncolumns = nspecies + 9

    volshells = 4. / 3. * np.pi * (bins[1:] ** 3 - bins[:-1] ** 3)

    mass, _ = np.histogram(radius, bins=bins, weights=s.mass.astype('f8'))
    i, = np.where(mass > 0)
    density = mass / volshells

    phi = np.arctan2(s.cmce[:, 1] - center[1], s.cmce[:, 0] - center[0])
    vel_phi = (s.vel[:, 0] - vel[0]) * (-np.sin(phi)) + (s.vel[:, 1] - vel[1]) * (np.cos(phi))

    mv_phi, _ = np.histogram(radius, bins=bins, weights=s.mass.astype('f8') * vel_phi)
    v_phi = mv_phi
    v_phi[i] /= mass[i]

    mv_r, _ = np.histogram(radius, bins=bins,
                           weights=s.mass.astype('f8') * ((s.cmce - center[None, :]) * (s.vel - vel[None, :])).sum(
                               axis=1) / radius)
    v_r = mv_r
    v_r[i] /= mass[i]

    ml_z, _ = np.histogram(radius,
                           bins=bins,
                           weights=np.cross(s.cmce - center[None, :],
                                            s.mass.astype('f8')[:, None] * (s.vel - vel[None, :]))[:, 2])
    l_z = ml_z
    l_z[i] /= mass[i]

    mtemp, _ = np.histogram(radius, bins=bins, weights=s.mass.astype('f8') * s.temp)
    temp = mtemp
    temp[i] /= mass[i]

    mu, _ = np.histogram(radius, bins=bins, weights=s.mass.astype('f8') * s.u)
    u = mu
    u[i] /= mass[i]

    vol, _ = np.histogram(radius, bins=bins, weights=s.vol.astype('f8'))
    pv, _ = np.histogram(radius, bins=bins, weights=s.vol.astype('f8') * s.pres)
    iv, = np.where(vol > 0)
    p = pv
    p[i] /= vol[iv]

    xnuc = np.zeros((nshells, nspecies))
    for ix in range(nspecies):
        mxn, _ = np.histogram(radius, bins=bins, weights=s.mass.astype('f8') * s.xnuc[:, ix])
        xnuc[i, ix] = mxn[i] / mass[i]

    data = np.zeros((nshells, ncolumns))
    data[:, 0] = bins[:-1]
    data[:, 1] = bins[1:]
    data[:, 2] = mass
    data[:, 3] = density
    data[:, 4] = l_z
    data[:, 5] = v_r
    data[:, 6] = u
    data[:, 7] = p
    data[:, 8] = temp
    data[:, 9:] = xnuc

    print("Writing remnant file...")

    file_handle = open("table3.txt", "w", newline="")
    writer = csv.writer(file_handle, delimiter=",")
    writer.writerow(["Spec1", "Xnuc1", "Spec2", "Xnuc2"])

    with open("remnant_%s.txt" % simulation_name, "w") as f:
        f.write("# nshells: %d\n" % nshells)
        f.write("# nspecies: %d\n" % nspecies)
        f.write("# ncolumns: %d\n" % ncolumns)
        f.write("# total mass: %g\n" % mass.sum())
        for ix in range(nspecies):
            f.write("# total mass of %s: %g\n" % (sp['names'][ix], (xnuc[:, ix] * mass.astype('f8')).sum()))
            # file_handle.write(f"{sp['names'][ix]},{(xnuc[:, ix] * mass.astype('f8')).sum() / msol}\n")
        text = "# r_inner, r_outer, mass, density, l_z, v_r, u, p, T"
        for ix in range(nspecies):
            text += ", xn_%s" % sp['names'][ix]
        f.write(text + "\n")

        # noinspection PyTypeChecker
        np.savetxt(f, data, fmt="%.7e")

        for i in range(28):
            j = i + 28

            row = [sp['names'][i].title(), f"{(xnuc[:, i] * mass.astype('f8')).sum() / msol:0.2e}"]

            if j >= 55:
                row += ["-", "-"]
            else:
                row += [sp['names'][j].title(), f"{(xnuc[:, j] * mass.astype('f8')).sum() / msol:0.2e}"]

            writer.writerow(row)

    file_handle.close()
    print("Creating remnant figure...")

    f = plt.figure(FigureClass=sfig, figsize=(9.0, 4.4), dpi=300)

    box = 4e9
    res = 1024
    fac = 1e9

    s.centerat(center)

    ax = f.iaxes(0.7 + 2.7 * 0, 2.7, 2.0, 2.0, top=False)
    rho = s.get_Aslice("rho", res=res, axes=[0, 1], box=[box, box], proj=False, numthreads=16)
    print("RHO:", rho['grid'].min(), rho['grid'].max())
    pc = ax.pcolormesh(rho['x'] / fac, rho['y'] / fac, rho['grid'].T, cmap=cm.magma, rasterized=True,
                       norm=matplotlib.colors.LogNorm(vmin=1e2, vmax=1e7))
    ax.axis('tight')
    ax.set_ylabel("$y\,\mathrm{[10^4\,km]}$")
    ax.set_xticklabels([])

    cax = f.iaxes(0.7 + 2.7 * 0, 0.2, 2.0, 0.1, top=False)
    colorbar(pc, cax=cax, orientation='horizontal', label="$\\rho\,\mathrm{[g\,cm^{-3}]}$")

    ax = f.iaxes(0.7 + 2.7 * 0, 3.9, 2.0, 1.0, top=False)
    rho = s.get_Aslice("rho", res=res, axes=[0, 2], box=[box, box / 2], proj=False, numthreads=16)
    print("RHO:", rho['grid'].min(), rho['grid'].max())
    ax.pcolormesh(rho['x'] / fac, rho['y'] / fac, rho['grid'].T, cmap=cm.magma, rasterized=True,
                  norm=matplotlib.colors.LogNorm(vmin=1e2, vmax=1e7))
    ax.axis('tight')
    ax.set_xlabel("$x\,\mathrm{[10^4\,km]}$")
    ax.set_ylabel("$z\,\mathrm{[10^4\,km]}$")
    ax.set_xticks([-2., -1., 0., 1., 2.])

    ax = f.iaxes(0.7 + 2.7 * 1, 3.9, 2.0, 3.8, top=False)

    bins = np.logspace(5, 10.2, 40)
    hmass, _ = np.histogram(s.r(), bins=bins, weights=s.mass.astype('f8'))
    volumes = 4. / 3. * np.pi * (bins[1:] ** 3 - bins[:-1] ** 3)
    centers = 0.5 * (bins[1:] + bins[:-1]) / 1e5
    l1, = ax.loglog(centers, hmass / volumes, label="$t=%5.1fs$" % s.time, color="tab:blue")
    ax.set_xlim(1e2, 2e5)

    ax.legend(frameon=False, loc='lower left', fontsize=8)
    ax.set_xlabel("$r\,\mathrm{[km]}$")
    ax.set_ylabel("$\\rho\,\mathrm{[g\,cm^{-3}]}$", color=l1.get_color())

    ax2 = ax.twinx()

    hmass, _ = np.histogram(s.r(), bins=bins, weights=s.mass)
    htemp, _ = np.histogram(s.r(), bins=bins, weights=s.mass * s.temp)
    i, = np.where(hmass > 1e-5 * msol)
    l2, = ax2.loglog(centers[i], htemp[i] / hmass[i], color="tab:orange")
    ax2.set_ylabel("$T\,\mathrm{[K]}$", color=l2.get_color())

    ax = f.iaxes(0.7 + 2.7 * 2 + 0.7, 3.9, 2.0, 3.8, top=False)

    hmass, _ = np.histogram(s.r(), bins=bins, weights=s.mass)
    for nuc in ['he4', 'c12', 'o16', 'si28', 's32', 'ca40', 'ni56']:
        ix = sp['names'].index(nuc)
        hxnuc, _ = np.histogram(s.r(), bins=bins, weights=s.mass * s.xnuc[:, ix])
        i, = np.where(hmass > 0)
        if (hxnuc[i] / hmass[i]).max() > 1e-2:
            label = "$^{%d}\mathrm{%s}$" % (sp['na'][ix], asymb[sp['nz'][ix]].capitalize())
            ax.loglog(centers[i], hxnuc[i] / hmass[i], label=label)

        print(nuc.upper(), (hxnuc[i] / hmass[i]).max(), (s.mass * s.xnuc[:, ix]).sum() / msol)

    ax.set_xlim(1e2, 2e5)
    ax.set_ylim(1e-4, 1.0)
    ax.legend(frameon=False, fontsize=8, loc='lower left')

    ax.set_xlabel("$r\,\mathrm{[km]}$")
    ax.set_ylabel("$X_\mathrm{i}$")

    f.savefig('remnant_%s.pdf' % simulation_name, dpi=300)


if __name__ == "__main__":
    main()

