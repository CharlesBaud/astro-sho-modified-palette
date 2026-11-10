#!/usr/bin/env python3
"""Hubble CCD bleed trails: a saturated star's extra charge runs up and down its column. Each
bled star keeps its round saturated core; the rest of its trail (Hubble's full well flag,
DQ 256) and a margin round it are redrawn as the star's own radial profile, the sky between
the neighbouring stars, photon noise to match, and the vertical diffraction spike measured on
the horizontal one. Works on a chip in electrons, in detector coordinates, before any
resampling, where the trails are straight columns.

Trial: python3 hstbleed.py <flc.fits> <out-dir> [chip]  writes before/after cut-outs of the
longest trails."""

import sys, os
import numpy as np
from scipy.ndimage import label, find_objects, gaussian_filter

DQ_SAT = 256      # Full well saturation flag (ACS/WFC and WFC3/UVIS)
TRAIL_MIN = 4     # Rows beyond the core for a blob to count as bled
CORE_FRAC = 0.6   # Core: rows at least this share of the blob's widest row
MARGIN = 3        # Extra columns either side of the trail that are replaced too (its soft edge)
SIDE = 4          # Columns either side, past the margin, the fill is taken from
TAIL = 6          # Rows past the flagged end of a trail also replaced (charge thinning out)
BULGE = 2         # Another star along a trail: the blob this many columns wider than either side
BULGE_ROWS = 8    # within this many rows
SPIKE_HALF = 3    # Diffraction spike: rows either side of its centre line measured and redrawn
SPIKE_SMOOTH = 9  # Spike measured as the mean over this many distances
SPIKE_K = 3.0     # and drawn where it stands this many noise sigma over its surroundings
READNOISE = 4.0   # Electrons, for the noise drawn into the fill


def Stars(sat, e):
    """The bled stars: each saturated blob that runs along its column past a round core, as its
    slice and its cores (row, column, radius) in the chip. One blob can hold several stars on
    one column whose trails joined: a core wherever the blob bulges out again along the trail.
    The first is the brightest; a core's radius is half the blob's width on its row (the
    trail is as wide as the core: every core column bleeds)."""
    lab, n = label(sat)
    found = []
    ny, nx = sat.shape
    for i, s in enumerate(find_objects(lab), 1):
        blob = lab[s] == i
        width = blob.sum(axis=1)
        rad = width.max() / 2
        if blob.shape[0] < 2 * rad + TRAIL_MIN:
            continue
        x0, x1 = s[1].start - MARGIN - SIDE, s[1].stop + MARGIN + SIDE
        if x0 < 0 or x1 > nx:
            Stars.edge += 1  # Against the chip's side: left as it is
            continue
        halo = e[s[0], x0:x0 + SIDE].mean(axis=1) + e[s[0], x1 - SIDE:x1].mean(axis=1)
        halo = np.convolve(halo, np.ones(5) / 5, mode='same')
        main = halo.copy()
        main[width < CORE_FRAC * width.max()] = -np.inf  # The brightest is among the widest rows
        rows = [int(np.argmax(main))]
        # Other stars along the trail: the trail narrows steadily away from a core, another star
        # makes the blob bulge again. A plateau of the width at least BULGE wider than the
        # narrowest within BULGE_ROWS on both sides
        y = 0
        while y < len(width):
            z = y
            while z + 1 < len(width) and width[z + 1] == width[y]:
                z += 1
            before = width[max(y - BULGE_ROWS, 0):y]
            after = width[z + 1:z + 1 + BULGE_ROWS]
            mid = (y + z) // 2
            if len(before) and len(after) and width[y] - before.min() >= BULGE and \
               width[y] - after.min() >= BULGE and all(abs(mid - r) > 2 * rad for r in rows):
                rows.append(mid)
            y = z + 1
        cores = []
        for y in rows:
            c = np.flatnonzero(blob[y])
            if len(c) == 0:
                continue
            cores.append((y + s[0].start, (c[0] + c[-1]) / 2 + s[1].start, max(len(c) / 2, 1.0)))
        found.append((s, cores))
    return found


Stars.edge = 0


def Profile(e, bad, s, cy, cx, rad, b0, b1, y0, y1):
    """The star's brightness by distance from its centre: the median at each distance of the
    good pixels (not saturated, not in any band) beside the band, out to the furthest band
    pixel. Inside the band's half width nothing is seen: from the core's edge, at the
    saturated level, it falls evenly in log to the first measured distance. None if nothing
    round the star is good (packed among other bled stars)."""
    rmax = int(np.ceil(max(cy - y0, y1 - cy, cx - b0, b1 - cx) * 1.42)) + 2
    ny, nx = e.shape
    wy0, wy1 = max(int(cy) - rmax, 0), min(int(cy) + rmax + 1, ny)
    wx0, wx1 = max(int(cx) - rmax, 0), min(int(cx) + rmax + 1, nx)
    yy, xx = np.mgrid[wy0:wy1, wx0:wx1]
    r = np.hypot(yy - cy, xx - cx).astype(int)
    use = ((xx < b0) | (xx >= b1)) & ~bad[wy0:wy1, wx0:wx1]
    v = e[wy0:wy1, wx0:wx1][use]
    ru = r[use]
    prof = np.full(rmax + 1, np.nan)
    order = np.argsort(ru)
    ru, v = ru[order], v[order]
    edges = np.searchsorted(ru, np.arange(rmax + 2))
    for k in range(rmax + 1):
        if edges[k + 1] > edges[k]:
            prof[k] = np.median(v[edges[k]:edges[k + 1]])
    known = np.flatnonzero(np.isfinite(prof))
    top = np.max(e[s])
    if len(known) == 0:
        return None
    # Inside the band's width nothing is measured: from the core's edge (saturated) the
    # brightness falls away evenly in log to the first measured distance, so no hard disc
    edge = int(np.ceil(rad))
    known = known[known > edge]
    if len(known) == 0:
        return None
    rr = np.arange(rmax + 1)
    lp = np.interp(rr, np.concatenate([[edge], known]), np.log(np.maximum(np.concatenate([[top], prof[known]]), 1e-3)))
    prof = np.exp(lp)
    prof[:edge + 1] = top
    return prof


def Fill(e, sat, found):
    """Each bled star: a band the blob's full width plus MARGIN columns either side, over its
    rows plus TAIL past each end. Outside the round cores the band is redrawn as the stars' own
    radial profiles, plus the sky round them with the neighbouring stars clipped off, carried
    in smoothly, plus photon noise; then the vertical spikes. Every band and core is marked
    first: no star is measured on another's trail, and no core is painted over."""
    ny, nx = e.shape
    bad = sat.copy()
    cores = np.zeros_like(sat)
    bands = []
    for s, cl in found:
        y0, y1 = max(s[0].start - TAIL, 0), min(s[0].stop + TAIL, ny)  # A trail can run off the chip
        b0, b1 = s[1].start - MARGIN, s[1].stop + MARGIN  # Band, b1 exclusive
        bad[y0:y1, b0:b1] = True
        yy, xx = np.mgrid[y0:y1, b0:b1]
        for cy, cx, rad in cl:
            cores[y0:y1, b0:b1] |= (yy - cy) ** 2 + (xx - cx) ** 2 <= (rad + 0.5) ** 2
        bands.append((y0, y1, b0, b1))
    out = e.copy()
    redrawn = np.zeros_like(sat)  # Returned: the pixels drawn, for the other filters to fill (CrossFill)
    rng = np.random.default_rng(1)  # The same result every run
    for (s, cl), (y0, y1, b0, b1) in zip(found, bands):
        profs = [Profile(e, bad, s, cy, cx, rad, b0, b1, y0, y1) for cy, cx, rad in cl]
        cl = [c for c, p in zip(cl, profs) if p is not None]  # No profile: the sky alone
        profs = [p for p in profs if p is not None]
        # What the good pixels around have beyond the stars' profiles (sky, neighbours), carried
        # into the band by a blur of the good pixels only (normalised convolution): smooth both
        # along and across the trail, no row stripes
        w = (b1 - b0) * 2 + SIDE
        wy0, wy1, wx0, wx1 = max(y0 - w, 0), min(y1 + w, ny), max(b0 - w, 0), min(b1 + w, nx)
        yy, xx = np.mgrid[wy0:wy1, wx0:wx1]
        pw = np.zeros(yy.shape, dtype=np.float32)
        for (cy, cx, rad), prof in zip(cl, profs):  # The brighter star's profile where they meet
            pw = np.maximum(pw, prof[np.minimum(np.hypot(yy - cy, xx - cx).astype(int), len(prof) - 1)])
        good = ~bad[wy0:wy1, wx0:wx1]
        res = e[wy0:wy1, wx0:wx1] - pw
        # Neighbouring stars clipped off first: the stars under a trail are lost, so the fill is
        # the sky between them, not their light smeared into a grey box
        gv = res[good]
        if gv.size:
            m = np.median(gv)
            sd = 1.4826 * np.median(np.abs(gv - m))
            res = np.where(res > m + 2 * sd, m, res)  # Replaced, not clipped: a clipped star still adds
        res = np.where(good, res, 0)
        sig = max(1.5, (b1 - b0) / 3)
        num = gaussian_filter(res, sig)
        den = gaussian_filter(good.astype(np.float32), sig)
        fill = pw + num / np.maximum(den, 1e-3)
        band = (slice(y0 - wy0, y1 - wy0), slice(b0 - wx0, b1 - wx0))
        keep = cores[y0:y1, b0:b1]
        tgt = out[y0:y1, b0:b1]
        f = fill[band][~keep]
        # The fill is smooth; the sky round it has photon noise. Every frame's trail is in the
        # same place, so without it a too clean streak would survive the stack
        f = f + rng.standard_normal(f.shape).astype(np.float32) * np.sqrt(np.maximum(f, 0) + READNOISE ** 2)
        tgt[~keep] = f
        redrawn[y0:y1, b0:b1] |= ~keep
        for (cy, cx, rad), prof in zip(cl, profs):
            Spike(out, e, bad, cores, prof, cy, cx, rad, b0, b1, y0, y1)
    return out, redrawn


def Spike(out, e, bad, cores, prof, cy, cx, rad, b0, b1, y0, y1):
    """Hubble's spikes are a cross, all four alike; the vertical pair runs along the trail and
    was filled away. Measured on the horizontal pair, beside the band (what it has above the
    star's round profile, by distance and by row either side of its line), and drawn back
    down the column."""
    ny, nx = e.shape
    reach = max(cy - y0, y1 - cy)
    ic, jc = int(round(cy)), int(round(cx))
    if ic - SPIKE_HALF < 0 or ic + SPIKE_HALF >= ny:
        return
    # The horizontal spike's line: the row near the centre with most light beside the band
    rows = np.arange(ic - SPIKE_HALF, ic + SPIKE_HALF + 1)
    # Each side separately; a real spike is on both, a neighbouring star only on one, so the
    # smaller of the two is the spike
    exc = np.full((2, 2 * SPIKE_HALF + 1, reach + 1), np.nan)
    for si, side in enumerate((-1, 1)):
        d = np.arange(int(b1 - cx) + 1 if side > 0 else int(cx - b0) + 1, reach + 1)
        xs = np.round(cx + side * d).astype(int)
        ok = (xs >= 0) & (xs < nx)
        d, xs = d[ok], xs[ok]
        for k, y in enumerate(rows):
            g = ~bad[y, xs]  # Not on another star's trail
            r = np.hypot(y - cy, xs[g] - cx).astype(int)
            exc[si, k, d[g]] = e[y, xs[g]] - prof[np.minimum(r, len(prof) - 1)]
    have = np.isfinite(exc).all(axis=(0, 1))
    exc = np.where(have, np.nanmin(np.nan_to_num(exc, nan=np.inf), axis=0), 0)
    base = np.minimum(exc[0], exc[-1])  # Only the line: what the outer rows also have is surroundings
    noise = 1.4826 * np.median(np.abs(base[have] - np.median(base[have]))) if have.any() else 0
    exc -= base
    # Smoothed along its length, and only where it stands SPIKE_K noise over the surroundings:
    # single pixel noise drawn down the column would brighten the whole trail
    k = np.ones(SPIKE_SMOOTH) / SPIKE_SMOOTH
    exc = np.array([np.convolve(r, k, mode='same') for r in exc])
    exc[:, exc.max(axis=0) < SPIKE_K * noise / np.sqrt(SPIKE_SMOOTH)] = 0
    exc = np.clip(exc, 0, None)
    # Distances inside the band's half width are not seen beside it: the first measured value
    # carries in to the core
    first = np.flatnonzero(have)
    if len(first) == 0:
        return
    exc[:, :first[0]] = exc[:, first[0]:first[0] + 1]
    for y in range(y0, y1):
        dy = int(abs(y - cy))
        if dy <= rad or dy > reach:
            continue
        for k in range(2 * SPIKE_HALF + 1):
            x = jc + (k - SPIKE_HALF)
            if b0 <= x < b1 and not cores[y, x]:
                out[y, x] += exc[k, dy]


CROSS_PAD = 12     # Cross filter fill: pixels round the redrawn ones the detail is measured over
CROSS_SIG = 3.0    # Detail: a frame less its blur of this sigma (stars stand out, sky and halo go)
CROSS_K = 2.0      # Detail taken from another filter only where it is this many noise sigma up
CROSS_STAR = 5.0   # Filters compared on stars this many noise sigma up in both
CROSS_NOISE_BOX = 512  # Noise measured on a square this size, the most plain sky
CROSS_SAT = 65000  # Frame values (ADU) this high are saturated: not real data for the fill


def CrossFill(paths, masks):
    """After the bleed repair each filter's trails are sky, but stars under a trail in one filter
    can be clear in another that did not bleed there: they would show in that colour alone.
    paths is {slot: [frames on one grid]}, masks {frame: redrawn pixels}. A redrawn pixel that
    other frames of the same filter have for real (not redrawn, not saturated) takes their
    value. Otherwise the star detail of the filters that have it for real is added, scaled to this
    filter by the stars they share."""
    from astropy.io import fits
    from scipy.ndimage import binary_dilation
    slots = list(paths)
    shape = next(iter(masks.values())).shape
    anym = np.zeros(shape, dtype=bool)
    for m in masks.values():
        anym |= m
    if not anym.any():
        return
    # Boxes round the redrawn pixels: all the work is there
    lab, n = label(binary_dilation(anym, iterations=CROSS_PAD))
    boxes = find_objects(lab)
    # Each filter's mean of its real pixels (data, not redrawn) and how many frames have one
    total = {}
    count = {}
    for sl in slots:
        t = np.zeros(shape, dtype=np.float32)
        c = np.zeros(shape, dtype=np.uint8)
        for p in paths[sl]:
            d = fits.getdata(p)
            real = (d > 0) & (d < CROSS_SAT)  # A saturated pixel is no better than the repair
            if p in masks:
                real &= ~masks[p]
            t[real] += d[real]
            c += real
        total[sl], count[sl] = t, c
    detail = {sl: np.zeros(shape, dtype=np.float32) for sl in slots}
    for b in boxes:
        for sl in slots:
            c = count[sl][b]
            mean = np.where(c > 0, total[sl][b] / np.maximum(c, 1), 0)
            w = gaussian_filter((c > 0).astype(np.float32), CROSS_SIG)
            blur = gaussian_filter(mean, CROSS_SIG) / np.maximum(w, 1e-3)
            detail[sl][b] = np.where(c > 0, mean - blur, 0)
    inbox = np.zeros(shape, dtype=bool)
    for b in boxes:
        inbox[b] = True
    # Each filter's noise on plain sky, away from the trails: round the bright stars whose
    # trails were repaired the halos would make it look several times higher, and the faint
    # stars there would not be taken
    noise = {}
    for sl in slots:
        c = count[sl]
        best, by, bx = -1, 0, 0
        for y in range(0, shape[0] - CROSS_NOISE_BOX + 1, CROSS_NOISE_BOX // 2):
            for x in range(0, shape[1] - CROSS_NOISE_BOX + 1, CROSS_NOISE_BOX // 2):
                k = (c[y:y + CROSS_NOISE_BOX:16, x:x + CROSS_NOISE_BOX:16] > 0).mean() - inbox[y:y + CROSS_NOISE_BOX:16, x:x + CROSS_NOISE_BOX:16].mean()
                if k > best:
                    best, by, bx = k, y, x
        b = (slice(by, by + CROSS_NOISE_BOX), slice(bx, bx + CROSS_NOISE_BOX))
        mean = np.where(c[b] > 0, total[sl][b] / np.maximum(c[b], 1), 0)
        v = (mean - gaussian_filter(mean, CROSS_SIG))[(c[b] > 0) & ~inbox[b]]
        noise[sl] = 1.4826 * np.median(np.abs(v - np.median(v))) if v.size else 0
    # How bright a star of filter o is in filter s: median ratio over the stars both have
    ratio = {}
    for sl in slots:
        for o in slots:
            if o == sl or not noise[sl] or not noise[o]:
                continue
            both = inbox & (count[sl] > 0) & (count[o] > 0) & (detail[sl] > CROSS_STAR * noise[sl]) & \
                   (detail[o] > CROSS_STAR * noise[o])
            if both.sum() >= 50:
                ratio[sl, o] = float(np.median(detail[sl][both] / detail[o][both]))
    for sl in slots:
        nsame = ncross = 0
        for p in paths[sl]:
            if p not in masks or not masks[p].any():
                continue
            with fits.open(p) as h:
                d = h[0].data.astype(np.float32)
                hd = h[0].header.copy()
            m = masks[p] & (d > 0)
            # The same filter has it for real in another frame: that value (their mean) copied
            # in. Not left as no data (0): the stack's combine treats a lone value beside a
            # saturated core as unusable
            same = m & (count[sl] > 0)
            d[same] = total[sl][same] / count[sl][same]
            rest = m & ~same
            add = np.zeros(shape, dtype=np.float32)
            nd = np.zeros(shape, dtype=np.float32)
            for o in slots:
                if (sl, o) not in ratio:
                    continue
                have = rest & (count[o] > 0)
                star = have & (detail[o] > CROSS_K * noise[o])
                add[star] += ratio[sl, o] * detail[o][star]
                nd[have] += 1
            d[rest] += add[rest] / np.maximum(nd[rest], 1)
            nsame += same.sum()
            ncross += (rest & (nd > 0)).sum()
            fits.PrimaryHDU(np.clip(np.round(d), 0, 65535).astype(np.uint16), header=hd).writeto(p, overwrite=True)
        print('slot %d: bleed repairs, %d pixels from the same filter\'s other frames, %d given the other filters\' stars' % (sl, nsame, ncross))
    print('filter noise: ' + ', '.join('%d %.1f' % (k, v) for k, v in sorted(noise.items())))
    print('filter ratios: ' + ', '.join('%d/%d %.2f' % (a, b, r) for (a, b), r in sorted(ratio.items())))


def Stretch(v, lo, hi):
    return np.clip(np.arcsinh((v - lo) / (hi - lo) * 50) / np.arcsinh(50), 0, 1)


if __name__ == '__main__':
    from astropy.io import fits
    from PIL import Image
    f, outdir = sys.argv[1], sys.argv[2]
    ver = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    h = fits.open(f)
    e = h['SCI', ver].data.astype(np.float32)
    sat = (h['DQ', ver].data & DQ_SAT) != 0
    found = Stars(sat, e)
    fixed, redrawn = Fill(e, sat, found)
    found.sort(key=lambda t: -(t[0][0].stop - t[0][0].start))
    lo, hi = np.median(e), np.percentile(e, 99.9)
    print('%d bled stars, %d against the chip edge left' % (len(found), Stars.edge))
    for k, (s, cl) in enumerate(found[:3]):
        cyc, cxc, rad = cl[0]
        cy, cx = (s[0].start + s[0].stop) // 2, (s[1].start + s[1].stop) // 2
        hh = (s[0].stop - s[0].start) // 2 + 40
        y0, y1, x0, x1 = max(cy - hh, 0), cy + hh, max(cx - 60, 0), cx + 60
        pair = np.hstack([Stretch(e[y0:y1, x0:x1], lo, hi), np.ones((y1 - y0, 4)),
                          Stretch(fixed[y0:y1, x0:x1], lo, hi)])[::-1]
        Image.fromarray((pair * 255).astype(np.uint8)).save(os.path.join(outdir, 'bleed%d.png' % k))
        y0, x0 = int(cyc) - 30, cx - 30  # Close up of the core, 4x
        z = np.hstack([Stretch(e[y0:y0 + 60, x0:x0 + 60], lo, hi), np.ones((60, 2)),
                       Stretch(fixed[y0:y0 + 60, x0:x0 + 60], lo, hi)])[::-1]
        Image.fromarray((z * 255).astype(np.uint8)).resize((z.shape[1] * 4, 240), Image.NEAREST).save(
            os.path.join(outdir, 'core%d.png' % k))
        print('bleed%d.png: star at row %d col %d, trail %d rows, core radius %.1f' % (k, cyc, cx, s[0].stop - s[0].start, rad))
