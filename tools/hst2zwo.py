#!/usr/bin/env python3
"""Store Hubble WFC3/UVIS or ACS/WFC exposures (flc files) as ASICap ZWO style FITS frames.

Each flc holds the two chips (SCI,1 is UVIS2 or WFC2, SCI,2 is UVIS1 or WFC1), calibrated,
in electrons. They are joined as on the camera, chip 2 at the bottom, the chip gap (31 rows
on UVIS, 50 on WFC) as 0 (zeroInvalid), chip 1 on top. ACS is strongly distorted: use
mosaic, which corrects it from the headers. Pixels flagged bad in the DQ arrays, cosmic
rays found on each exposure (L.A.Cosmic) and dark charge trap lines are replaced by the median around them, giving 4096 x 4133. Exposure times differ between filters
(Ha 400 s, OIII 950 s, SII 665 s), so values are electrons per second times NORM_S times ADU_PER_E for
every filter, as if all exposures were NORM_S long, plus a 1000 offset so negative
noise survives, clipped to 0-65535, uint16 (BZERO 32768) like ASICap.

One run directory per filter named from its first exposure, files
YYYY-MM-DD-HHMM_S-<slot>-CapObj_NNNN.FIT, slots as the wheel: Ha 3, OIII 4, SII 5
([NII] F658N takes slot 5 when there is no SII; on ACS it is the Ha filter), and broadband
red/I 1, green/V 2, blue/B 6.

Exposures come from MAST (mast.stsci.edu), e.g. NGC 6960 Veil (proposal 14056 visit 02)
or NGC 6888 (proposal 17882): the *_flc.fits files of each filter's association.

Add 'mosaic' to put all exposures on one sky grid from their headers (0 where a frame has
no data), for stacking several pointings with "preAligned": true.

usage: hst2zwo.py <flc dir> <out dir> [mosaic]
"""
import os, sys, glob, datetime
import numpy as np
import astroscrappy  # L.A.Cosmic, apt install python3-astroscrappy
from scipy.ndimage import median_filter, maximum_filter, binary_dilation, binary_opening, label, find_objects, map_coordinates
from numpy.lib.stride_tricks import sliding_window_view
from astropy.io import fits
from astropy.wcs import WCS

if len(sys.argv) < 3:
    sys.exit('usage: hst2zwo.py <flc dir> <out dir> [mosaic]')
SRC = sys.argv[1]
OUT = sys.argv[2]
MOSAIC = len(sys.argv) > 3 and sys.argv[3] == 'mosaic'
GRID = 32       # Sky grid to chip positions computed every this many pixels, interpolated between
SLOT = {'F656N': 3, 'F657N': 3, 'F502N': 4, 'F673N': 5, 'F658N': 5,
        'F814W': 1, 'F775W': 1, 'F625W': 1, 'F555W': 2, 'F606W': 2, 'F550M': 2, 'F435W': 6, 'F475W': 6}
# Chip gap in rows, read noise in electrons, DQ bits repaired. ACS/WFC also has warm pixels (64,
# 1.5% of a chip) and bad columns (128): with two frames a filter the stack cannot reject them
CAMERA = {'WFC3/UVIS': (31, 3.1, 4 | 16 | 32 | 512 | 1024), 'ACS/WFC': (50, 4.0, 4 | 16 | 32 | 64 | 128 | 512)}
GAP = 31        # Chip gap in rows, set for each frame's camera from CAMERA
EDGE_DEAD = 3   # Chip rows next to the gap, all flagged: no data like the gap
EDGE_MEDIAN = 7 # The next rows have bright unflagged spikes (p99.9 2-4x mid chip), where
                # the dithered gaps crossed they gave RGB spikes: 3x3 median filtered
STAR_K = 8.0    # 3x3 median over its surroundings, in noise sigma: most of 9 pixels raised, a star
SHARP_K = 5.0   # Over its surroundings, in noise sigma, to be a speck
SHARP_R = 0.12  # Speck: its neighbours (median of the 8) rise less than this of its own rise; a
                # star's rise about 0.3 of its peak
OFFSET = 1000.0 # ADU for zero electrons
NORM_S = 400.0  # Common exposure: the shortest, so nothing is scaled up into saturation
ADU_PER_E = 8.0  # Hubble narrowband skies hold a few electrons: scaled so the usual stretch
                 # gains (12-16) suit them, and the signal is not quantised
DQMASK = 4 | 16 | 32 | 512 | 1024  # Bad, hot, warm, bad flat, charge trap: repaired
DQ_CR = 4096    # Hubble's cosmic ray flag: repaired, except on a star
E_MIN, E_MAX = -500.0, 3e5  # Electrons no pixel can hold: the calibration leaves some at +-1e8 (DQ
                # 4096, not otherwise a defect): repaired, or clipped to 0 they read as no data
READNOISE = 3.1 # Read noise, electrons, set for each frame's camera from CAMERA
DARK_RUN = 31   # Dark lines: rows of the median along each column
DARK_AROUND = 21 # columns of the median across, the level beside the line
DARK_K = 5.0    # How far below that level, in noise sigma of the column median
DARK_LEN = 60   # Shortest line, rows, and at most this wide: DARK_WIDE columns and 1/8 of its length
DARK_WIDE = 6
LEVEL_STEP = 4     # Frame levels (MatchLevels): measured on every this many pixels
LEVEL_MIN_OVERLAP = 20000  # Fewest of those shared for a pair of frames to count
SPIKE_MAX_FRAMES = 18  # One frame spikes (OneFrameSpikes): holds all frames in memory, so only small sets
SPIKE_TILE = 512   # Tile size
SPIKE_AROUND = 11  # Surroundings, the median of this square
SPIKE_NEAR = 5     # Look in the other filters within this square
SPIKE_K = 5.0      # A spike stands this many noise sigma above its surroundings
SPIKE_EDGE = 2.5   # Its skirt, in noise sigma
SPIKE_QUIET = 4.0  # Other filter shows nothing: under this many of its noise sigma,
SPIKE_R = 0.03     # or under this fraction of the spike
MIN_GOOD = 8    # Good neighbours wanted to repair a flagged pixel
MAX_R = 32      # Furthest search; beyond, no data (in practice only the chip gap)
REPAIR_BATCH = 20000  # Flagged pixels repaired at a time: at most about 700 MB of windows
NEIGHBOURS = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=bool)


def RepairFlagged(e, bad):
    """Each flagged pixel gets the median of its nearest good pixels: the window grows
    from 5x5 until it holds MIN_GOOD good ones, so large dead areas are filled from their
    surroundings too. A median, so a star edge or a missed hot pixel nearby does not pull
    it. Not found within MAX_R it stays NaN (no data)."""
    v = np.where(bad, np.nan, e)
    vp = np.pad(v, MAX_R, constant_values=np.nan)
    ys, xs = np.nonzero(bad)
    out = np.full(len(ys), np.nan)
    todo = np.arange(len(ys))
    r = 2
    while len(todo) and r <= MAX_R:
        k = 2*r + 1  # Windows of the pixels still to do, from the padded view
        view = sliding_window_view(vp, (k, k))
        left = []
        for i in range(0, len(todo), REPAIR_BATCH):  # In batches: each window is a copy in memory
            t = todo[i:i + REPAIR_BATCH]
            win = view[ys[t] + MAX_R - r, xs[t] + MAX_R - r].reshape(len(t), k*k)
            ok = np.count_nonzero(~np.isnan(win), axis=1) >= MIN_GOOD
            if ok.any():
                out[t[ok]] = np.nanmedian(win[ok], axis=1)
            left.append(t[~ok])
        todo = np.concatenate(left)
        r = r*2 if r < 8 else r + 8  # 2, 4, 8, 16, 24, 32
    e = e.copy()
    e[ys, xs] = out
    return e


def CleanChip(e, bad, flag):
    """Cosmic rays found on the single exposure by L.A.Cosmic (sharp edges no star has,
    also most unflagged hot pixels) replaced by its median of the pixels around them, and
    the DQ flagged pixels (hot, bad, ...) by the median of the good pixels around them.
    Star cores are kept from cosmic ray repairs, and pixels sharper than a star are repaired too.
    Each exposure on its own. Only dead areas are left invalid: the stacker expects invalid pixels at the edges
    (and the chip gap), scattered ones grow into squares through the alignment. e is in electrons. objlim 15 keeps the cores of Hubble's undersampled
    stars. Returns the cleaned chip and the cosmic ray fraction."""
    cr, clean = astroscrappy.detect_cosmics(e.astype(np.float32), inmask=bad, gain=1.0, readnoise=READNOISE,
                                            sigclip=4.5, objlim=15.0, satlevel=1e9)
    star = StarCores(e, bad)
    cr &= ~star
    e = e.copy()
    e[cr] = clean[cr]
    bad = bad | (flag & ~star)
    bad = bad | Specks(e, bad)
    bad = bad | DarkLines(e, bad)
    return RepairFlagged(e, bad), cr.mean()


def Sigma(level):
    """Noise of a pixel from the level around it, electrons."""
    return np.sqrt(np.maximum(level, 0) + READNOISE*READNOISE)


def StarCores(e, bad):
    """Where a star may be: the 3x3 median, which a defect of a few pixels cannot raise,
    stands above its surroundings, in a round filled shape (a thick cosmic ray track is
    long or uneven). Grown, as a core's edge has the same shape."""
    v = np.where(bad, np.median(e[~bad]), e)
    around = median_filter(v, size=9)
    lab, n = label(median_filter(v, size=3) - around > STAR_K*Sigma(around))
    keep = np.zeros(n + 1, dtype=bool)
    for i, o in enumerate(find_objects(lab)):
        h, w = o[0].stop - o[0].start, o[1].stop - o[1].start
        keep[i + 1] = max(h, w) <= 1.6*min(h, w) and np.count_nonzero(lab[o] == i + 1) >= max(5, .6*h*w)
    return binary_dilation(keep[lab], iterations=2)


def Specks(e, bad):
    """Pixels sharper than any star: well over their surroundings, with neighbours that
    hardly rise. Twice, as the brighter of two touching ones hides the other."""
    v = np.where(bad, np.median(e[~bad]), e)
    out = np.zeros(e.shape, dtype=bool)
    for _ in range(2):
        around = median_filter(v, size=5)
        near = median_filter(v, footprint=NEIGHBOURS)
        rise = v - around
        s = (rise > SHARP_K*Sigma(around)) & (near - around < SHARP_R*rise)
        out |= s
        v = np.where(s, around, v)
    return out


def DarkLines(e, bad):
    """Charge traps Hubble does not flag: a column losing a fixed number of electrons along
    many rows, so faint filters read near 0 there, and with few exposures the stack keeps
    the low value. Found as a narrow dark line along a chip column: its median along the
    column stands below the columns beside it. Grown a column each way for its shoulders."""
    v = np.where(bad, np.median(e[~bad]), e)
    along = median_filter(v, size=(DARK_RUN, 1))
    around = median_filter(along, size=(1, DARK_AROUND))
    dark = along - around < -DARK_K*1.25*Sigma(around)/np.sqrt(DARK_RUN)
    low = dark & ~binary_opening(dark, structure=np.ones((1, DARK_WIDE + 1), dtype=bool))  # Not where dust is, wide
    lab, n = label(low)
    keep = np.zeros(n + 1, dtype=bool)
    for i, o in enumerate(find_objects(lab)):
        h, w = o[0].stop - o[0].start, o[1].stop - o[1].start
        keep[i + 1] = h >= DARK_LEN and w <= DARK_WIDE and h >= 8*w
    print('    dark lines repaired: %s' % (['x %d rows %d-%d' % (o[1].start, o[0].start, o[0].stop)
                                            for i, o in enumerate(find_objects(lab)) if keep[i + 1]] or 'none'))
    # A line's head can be wider: what is dark just beside a line found belongs to it
    line = keep[lab]
    near = maximum_filter(line.astype(np.uint8), size=(2*DARK_RUN + 1, 2*DARK_WIDE + 1)) > 0
    return binary_dilation(line | (dark & near), structure=np.array([[1, 1, 1]], dtype=bool))


def LoadChip(h, ver):
    """A chip's electrons, its pixels to repair (flagged, or impossible values), and those
    Hubble flags as cosmic rays."""
    e = h['SCI', ver].data.astype(np.float64)
    dq = h['DQ', ver].data
    return e, ((dq & DQMASK) != 0) | (e < E_MIN) | (e > E_MAX), (dq & DQ_CR) != 0


def ChipWcs(h, ver):
    return WCS(h['SCI', ver].header, h)


def SkyGrid(files):
    """One tangent plane grid holding all chips of all exposures, at the first's scale and
    angle. Returns the grid's WCS and its size."""
    h = fits.open(files[0], memmap=True)
    first = ChipWcs(h, 1)
    pts = []
    for f in files:
        h = fits.open(f, memmap=True)
        for ver in (1, 2):
            ny, nx = h['SCI', ver].data.shape
            pts.append(ChipWcs(h, ver).all_pix2world([0, nx-1, nx-1, 0], [0, 0, ny-1, ny-1], 0))
    ra, dec = np.hstack([p[0] for p in pts]), np.hstack([p[1] for p in pts])
    g = WCS(naxis=2)
    g.wcs.ctype = ['RA---TAN', 'DEC--TAN']
    g.wcs.crval = [ra.mean(), dec.mean()]
    g.wcs.crpix = [1, 1]
    g.wcs.cd = first.wcs.cd
    x, y = g.wcs_world2pix(ra, dec, 0)
    g.wcs.crpix = [2 - np.floor(x.min()), 2 - np.floor(y.min())]
    return g, int(np.ceil(x.max() - x.min())) + 4, int(np.ceil(y.max() - y.min())) + 4


def PlaceChip(out, e, w, g):
    """Put a chip on the sky grid, bilinear: only where all its pixels have data."""
    ny, nx = e.shape
    x, y = g.wcs_world2pix(*w.all_pix2world([0, nx-1, nx-1, 0], [0, 0, ny-1, ny-1], 0), 0)
    x0, x1 = max(int(x.min()) - 1, 0), min(int(x.max()) + 2, out.shape[1])
    y0, y1 = max(int(y.min()) - 1, 0), min(int(y.max()) + 2, out.shape[0])
    gx, gy = np.arange(x0, x1 + GRID, GRID), np.arange(y0, y1 + GRID, GRID)
    ax, ay = np.meshgrid(gx, gy)
    cx, cy = w.all_world2pix(*g.wcs_pix2world(ax, ay, 0), 0, quiet=True)  # Chip position, every GRID
    py, px = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    at = [(py - y0) / GRID, (px - x0) / GRID]
    sx, sy = map_coordinates(cx, at, order=1), map_coordinates(cy, at, order=1)
    inside = (sx >= 0) & (sx <= nx-1) & (sy >= 0) & (sy <= ny-1)
    ok = np.isfinite(e)
    at = [sy[inside], sx[inside]]
    v = map_coordinates(np.where(ok, e, 0), at, order=1)
    v[map_coordinates(ok.astype(np.float32), at, order=1) < 0.999] = np.nan
    r = out[y0:y1, x0:x1]
    r[inside] = v


def MatchLevels(paths):
    """Mosaic only: exposures sit at slightly different levels (Ha sky changes with the orbit),
    so where the mix of frames changes, at a chip gap or a frame's edge, the average steps.
    One offset per frame is found from the median difference of every overlapping pair and
    taken off, so each filter's overall level is unchanged. paths is {slot: [files]}."""
    for s in paths:
        small = []
        for p in paths[s]:
            f = fits.getdata(p)[::LEVEL_STEP, ::LEVEL_STEP].astype(np.float32)
            small.append(np.where(f > 0, f, np.nan))
        n = len(small)
        rows, diffs = [np.ones(n)/n], [0.0]  # The offsets sum to 0
        for i in range(n):
            for j in range(i + 1, n):
                both = np.isfinite(small[i]) & np.isfinite(small[j])
                if np.count_nonzero(both) >= LEVEL_MIN_OVERLAP:
                    r = np.zeros(n)
                    r[i], r[j] = 1, -1
                    rows.append(r)
                    diffs.append(np.median(small[i][both] - small[j][both]))
        off = np.linalg.lstsq(np.array(rows), np.array(diffs), rcond=None)[0]
        print('slot %d: frame levels matched, range %.1f ADU' % (s, off.max() - off.min()))
        for p, o in zip(paths[s], off):
            with fits.open(p) as h:
                f = h[0].data
                fixed = np.where(f > 0, np.clip(np.round(f.astype(np.float32) - o), 1, 65535), 0).astype(np.uint16)
                fits.PrimaryHDU(fixed, header=h[0].header).writeto(p, overwrite=True)


def Rises(v, valid):
    """How far each pixel stands above its surroundings, and the noise of that."""
    around = median_filter(v, size=SPIKE_AROUND)
    rise = v - around
    r = rise[valid]
    return rise, 1.4826*np.median(np.abs(r - np.median(r))) if r.size >= 64 else 0.0


def OneFrameSpikes(data):
    """Mosaic only: where one exposure alone covers a filter, the stack cannot reject a cosmic
    ray, and a round one looks like one of Hubble's sharp stars. A star shows in the other
    filters and a cosmic ray does not, so a rise in one filter with no rise in the others is
    marked. data is {slot: [frames]} on one grid, 0 for no data. Returns {slot: mask} of the
    pixels to repair, in each filter's only frame there."""
    slots = list(data)
    cover = {s: sum((f > 0).astype(np.uint8) for f in data[s]) for s in slots}
    level = {s: sum(f.astype(np.float32) for f in data[s]) / np.maximum(cover[s], 1) for s in slots}
    masks = {s: np.zeros(cover[s].shape, dtype=bool) for s in slots}
    hh, ww = cover[slots[0]].shape
    for y in range(0, hh, SPIKE_TILE):  # In tiles: most of the image has more than one frame
        for x in range(0, ww, SPIKE_TILE):
            y0, y1 = max(y - SPIKE_AROUND, 0), min(y + SPIKE_TILE + SPIKE_AROUND, hh)
            x0, x1 = max(x - SPIKE_AROUND, 0), min(x + SPIKE_TILE + SPIKE_AROUND, ww)
            one = {s: cover[s][y0:y1, x0:x1] == 1 for s in slots}
            if not any(o[y-y0:y-y0+SPIKE_TILE, x-x0:x-x0+SPIKE_TILE].any() for o in one.values()):
                continue
            rise, sig = {}, {}
            for s in slots:
                rise[s], sig[s] = Rises(level[s][y0:y1, x0:x1], one[s])
            for s in slots:
                if sig[s] <= 0:
                    continue
                alone = (rise[s] > SPIKE_K*sig[s]) & one[s]
                for o in slots:
                    if o != s:  # Nothing in the other filter: under its noise, or tiny next to this one
                        seen = maximum_filter(rise[o], size=SPIKE_NEAR)
                        alone &= (seen < SPIKE_QUIET*sig[o]) | (seen < SPIKE_R*rise[s])
                if alone.any():  # With the blob's skirt, what alignment spread
                    edge = (rise[s] > SPIKE_EDGE*sig[s]) & one[s]
                    grown = binary_dilation(alone, structure=NEIGHBOURS | ~NEIGHBOURS, iterations=4, mask=edge)
                    grown = binary_dilation(grown, structure=NEIGHBOURS | ~NEIGHBOURS) & one[s]
                    masks[s][y0:y1, x0:x1] |= grown
    return masks


def RepairOneFrameSpikes(paths):
    """Run OneFrameSpikes on the frames written, repair and rewrite them. paths is {slot: [files]}."""
    data = {s: [fits.getdata(p) for p in paths[s]] for s in paths}
    masks = OneFrameSpikes(data)
    for s in paths:
        print('slot %d: %.3f%% of pixels repaired as cosmic rays seen in one frame' % (s, 100*masks[s].mean()))
        for p, f in zip(paths[s], data[s]):
            m = masks[s] & (f > 0)
            if m.any():
                e = np.where(f > 0, f.astype(np.float64), np.nan)
                fixed = np.nan_to_num(np.clip(np.round(RepairFlagged(e, m)), 1, 65535), nan=0.0).astype(np.uint16)
                with fits.open(p) as h:
                    fits.PrimaryHDU(fixed, header=h[0].header).writeto(p, overwrite=True)


def FilterName(p):
    """WFC3 has one filter keyword; ACS two wheels, one of them CLEAR."""
    if p.get('FILTER'):
        return p['FILTER']
    for k in ('FILTER1', 'FILTER2'):
        f = str(p.get(k, '')).strip()
        if f and not f.startswith('CLEAR'):
            return f
    return 'CLEAR'


def Camera(p):
    return '%s/%s' % (str(p.get('INSTRUME', '')).strip(), str(p.get('DETECTOR', '')).strip())


def main():
    global GAP, READNOISE, DQMASK
    frames = []
    for f in sorted(glob.glob(os.path.join(SRC, '*_flc.fits'))):
        h = fits.open(f, memmap=True)
        p = h[0].header
        when = datetime.datetime.fromisoformat('%sT%s' % (p['DATE-OBS'], p['TIME-OBS']))
        if p.get('EXPFLAG', 'NORMAL') != 'NORMAL':  # Shutter not opened on the target: an empty frame
            print('Skipped %s: %s' % (os.path.basename(f), p['EXPFLAG']))
            continue
        if Camera(p) not in CAMERA:
            print('Skipped %s: camera %s not known' % (os.path.basename(f), Camera(p)))
            continue
        frames.append((FilterName(p), when, f))
    frames.sort(key=lambda x: (SLOT.get(x[0], 9), x[1]))

    os.makedirs(OUT, exist_ok=True)
    if MOSAIC:
        grid, gw, gh = SkyGrid([f for _, _, f in frames])
        print('Sky grid %d x %d' % (gw, gh))
    runs = {}
    written = {}
    for filt, when, f in frames:
        if filt not in SLOT:
            continue
        slot = SLOT[filt]
        if filt not in runs:  # Run directory from the first exposure of the filter
            runs[filt] = (when, os.path.join(OUT, when.strftime('%Y-%m-%d_%H_%M_%SZ')), 0)
            os.makedirs(runs[filt][1], exist_ok=True)
        start, rdir, n = runs[filt]
        runs[filt] = (start, rdir, n + 1)

        h = fits.open(f, memmap=True)
        p = h[0].header
        GAP, READNOISE, DQMASK = CAMERA[Camera(p)]
        # Hot pixels are fixed on the chip, the star matcher would align frames on them;
        # cosmic rays cannot be rejected by the stack with only 2-4 exposures a filter
        uvis2, cr2 = CleanChip(*LoadChip(h, 1))
        uvis1, cr1 = CleanChip(*LoadChip(h, 2))
        # UVIS2's top rows and UVIS1's bottom rows face the gap
        ne = EDGE_DEAD + EDGE_MEDIAN  # Not n, that numbers the frames
        uvis2[-ne:-EDGE_DEAD] = median_filter(uvis2[-ne-1:], size=3)[1:-EDGE_DEAD]
        uvis1[EDGE_DEAD:ne] = median_filter(uvis1[:ne+1], size=3)[EDGE_DEAD:ne]
        uvis2[-EDGE_DEAD:] = np.nan
        uvis1[:EDGE_DEAD] = np.nan
        gap = np.full((GAP, uvis1.shape[1]), np.nan)
        k = ADU_PER_E * NORM_S / float(p['EXPTIME'])  # ADU per electron
        if MOSAIC:
            e = np.full((gh, gw), np.nan, dtype=np.float32)
            PlaceChip(e, uvis2 * k, ChipWcs(h, 1), grid)
            PlaceChip(e, uvis1 * k, ChipWcs(h, 2), grid)
        else:
            e = np.vstack([uvis2 * k, gap, uvis1 * k])  # Row 0 is the bottom in FITS
        raw = np.nan_to_num(np.clip(np.round(e + OFFSET), 1, 65535), nan=0.0).astype(np.uint16)  # 0 only for no data

        name = '%s-%d-CapObj_%04d.FIT' % (start.strftime('%Y-%m-%d-%H%M_') + str(start.second // 10), slot, n)
        hd = fits.Header()
        for key, val, com in (('XBINNING', 1, 'Binning factor in width'), ('YBINNING', 1, 'Binning factor in height'),
                              ('EXPOINUS', int(p['EXPTIME'] * 1e6), 'Exposure time in us'),
                              ('GAIN', round(k, 4), 'ADU per electron (hst2zwo, %.0f s normalised)' % NORM_S),
                              ('OFFSET', int(OFFSET), 'ADU for zero electrons'),
                              ('SWCREATE', 'hst2zwo', os.path.basename(f)),
                              ('DATE-OBS', when.isoformat(timespec='milliseconds'), 'UTC start date of observation'),
                              ('BAYERPAT', 'NONE', 'Debayer pattern'), ('COLORTYP', 'RAW16', 'Color space'),
                              ('INSTRUME', 'HST ' + Camera(p), 'Camera model'), ('FILTER', filt, 'HST filter'),
                              ('XPIXSZ', 15.0, 'Pixel Width in um'), ('YPIXSZ', 15.0, 'Pixel Height in um'),
                              ('EXPOSURE', NORM_S, 'Normalised exposure (s), real %.0f' % p['EXPTIME']),
                              ('EXPTIME', NORM_S, 'Normalised exposure (s), real %.0f' % p['EXPTIME'])):
            hd[key] = (val, com)
        fits.PrimaryHDU(raw, header=hd).writeto(os.path.join(rdir, name), overwrite=True)
        sat = (raw >= 65535).mean() * 100
        print('%-6s %s %5.0fs  sky %.0f ADU  saturated %.3f%%  cosmic rays %.2f%%  -> %s' % (filt, when.strftime('%H:%M:%S'),
              p['EXPTIME'], np.median(raw[raw > 0]), sat, 50 * (cr1 + cr2), os.path.join(os.path.basename(rdir), name)))
        written.setdefault(slot, []).append(os.path.join(rdir, name))
    if MOSAIC:
        MatchLevels(written)
    if MOSAIC and len(written) > 1:
        if sum(len(v) for v in written.values()) <= SPIKE_MAX_FRAMES:
            RepairOneFrameSpikes(written)
        else:
            print('Over %d frames: the stack rejects cosmic rays, one frame spikes not repaired' % SPIKE_MAX_FRAMES)


if __name__ == '__main__':
    main()
