#!/usr/bin/env python3
"""Narrowband colour for astronomy: an SHO like look with white stars and no green.

    python3 sho_palette.py Ha.fits OIII.fits SII.fits

Give it three stacked, aligned, linear (not yet stretched) mono images, FITS or
TIFF. It writes a 16 bit colour TIFF, finished for display.

How the colour is made, with O, S and Ha the three filters:
  1. Each filter is scaled so that a typical star has Ha = OIII = SII. Gas still has
     Ha much stronger than the others, so these ratios tell stars from gas.
  2. Gold to cyan comes from the ratio of OIII to SII.
  3. The ratio of Ha to the other two twists that axis: low Ha leans to green, high
     Ha to red and blue, equalised so the gas uses the whole range. It also pushes
     across the axis, towards red or blue.
  4. Ratios near 1 give no colour, so stars stay white. Adding the local noise k to
     both sides of each ratio does the same for empty sky, so noise stays grey.
  5. Brightness comes from the sum of all three, separately from the colour.
"""
import argparse
import os
import sys

try:
    import numpy as np
    from astropy.io import fits
    import tifffile
except ImportError as e:
    sys.exit('Missing library: %s\nInstall what is needed with:  pip install numpy astropy tifffile' % e.name)

# The look of the palette. Hues are OKLab angles in degrees.
GOS = 3.0          # Magnifier of the OIII/SII ratio
GH = 1.0           # Steepness of the Ha ratio
GATE_T0 = .25      # Ha ratio |t| where gas gets its full gold or cyan
GATE_MIN = .5      # Share of the gold or cyan kept where Ha equals the others: stars, halos
STAR_LO = 1.3      # Brightness over its surroundings where colour starts to fade (stars)
STAR_HI = 2.0      # and where it has gone
CHROMA = .14       # OKLab chroma at full colour
RED_PUSH = .6      # Red to blue push relative to gold to cyan
HUE_RED = 30.0     # Direction of the push for high Ha
HUE_BLUE = 265.0   # and for low Ha
HUE_GOLD = (110.0, 40.0)   # Gold end of the axis, at low Ha and at high Ha
HUE_CYAN = (190.0, 235.0)  # Cyan end of the axis, at low Ha and at high Ha

COLOUR_R = 3       # Colour comes from 7x7 means: it needs less detail than brightness
STAR_R = 15        # Surroundings a star is measured against
NOISE_BLOCK = 64   # Pixels each side of a local noise measurement
NOISE_SMOOTH = 8   # Blocks each way the noise map is smoothed over
LIFT_STEP = 3      # Every this many pixels each way for the median brightness of the lift
EQ_BINS = 256      # Histogram of the Ha ratio, used for the twist
EQ_LO, EQ_HI = -1.0, 3.0
BAL_LIMIT = .25    # The star balance stays within this and its inverse
BAL_STARS = 8      # Fewest stars to trust a star balance
BAL_STARS_MAX = 300
BAL_BOX = 12       # A star is the peak of a box of this radius, with a ring round it
BAL_AP = 5         # Aperture its flux is measured in
BAL_TILE = 32      # One candidate star from each tile of this size
BAL_MAX = 4000     # Most candidates
CHUNK = 256        # Rows done at a time, to save memory


def Kth(a, k):
    return np.partition(a, k)[k]


def SkyLevel(v, bad):
    """Dark sky level: the 0.1st percentile of the image binned 8x8."""
    nh, nw = v.shape[0]//8, v.shape[1]//8
    blocks = v[:nh*8, :nw*8].reshape(nh, 8, nw, 8).astype(np.float64).mean(axis=(1, 3))
    good = bad[:nh*8, :nw*8].reshape(nh, 8, nw, 8).sum(axis=(1, 3)) == 0
    blocks = blocks[good]
    return Kth(blocks, blocks.size//1000) if blocks.size else 0.0


def StarBalance(f):
    """Each filter's flux relative to Ha, the median over bright unsaturated stars."""
    hh, ww = f[0].shape
    sat = min(.7*Kth(l.ravel()[::7], int(.9999*(l.size//7))) for l in f)
    lum = f[0] + f[1] + f[2]
    cand = []
    for y in range(BAL_BOX, hh - BAL_BOX - BAL_TILE, BAL_TILE):
        for x in range(BAL_BOX, ww - BAL_BOX - BAL_TILE, BAL_TILE):
            t = lum[y:y+BAL_TILE, x:x+BAL_TILE]
            by, bx = np.unravel_index(np.argmax(t), t.shape)
            by += y
            bx += x
            p = lum[by, bx]
            if lum[by-BAL_BOX:by+BAL_BOX+1, bx-BAL_BOX:bx+BAL_BOX+1].max() <= p and len(cand) < BAL_MAX:
                cand.append((p, by, bx))
    del lum
    cand.sort(key=lambda c: -c[0])

    yy, xx = np.mgrid[-BAL_BOX:BAL_BOX+1, -BAL_BOX:BAL_BOX+1]
    r2 = xx*xx + yy*yy
    ring, ap = r2 >= (BAL_BOX-2)**2, r2 <= BAL_AP**2
    rat = []
    for _, y, x in cand:
        if len(rat) >= BAL_STARS_MAX:
            break
        flux, over = [], False
        for l in f:
            p = l[y-BAL_BOX:y+BAL_BOX+1, x-BAL_BOX:x+BAL_BOX+1]
            a = p[ap]
            over = over or (a >= sat).any()
            flux.append((a - Kth(p[ring], ring.sum()//2)).sum())
        if not over and min(flux) > 0:
            rat.append((flux[1]/flux[0], flux[2]/flux[0]))
    if len(rat) < BAL_STARS:
        return 1.0, 1.0, len(rat)
    rat = np.array(rat, dtype=np.float32)
    return Kth(rat[:, 0], len(rat)//2), Kth(rat[:, 1], len(rat)//2), len(rat)


def NoiseGrid(v, nw, nh):
    """Local noise: sigma from differences of adjacent pixels, the median per block."""
    n = NOISE_BLOCK
    d = np.abs(v[:nh*n, 1:nw*n+1] - v[:nh*n, :nw*n])
    g = np.empty((nh, nw), dtype=np.float32)
    for y in range(nh):
        row = d[y*n:(y+1)*n].reshape(n, nw, n).transpose(1, 0, 2).reshape(nw, -1)
        g[y] = np.partition(row, n*n//2, axis=1)[:, n*n//2] / .954
    return g


def SmoothNoise(g):
    """Smooth the noise map over a wide area, twice: where few frames cover the sky the noise
    is higher, and a sharp change there would show as a step in the colour along the frame's
    edge. Empty blocks read 0 and do not count."""
    r = NOISE_SMOOTH
    h, w = g.shape
    for _ in range(2):
        tables = []
        for a in (np.where(g > 0, g, 0).astype(np.float64), (g > 0).astype(np.float64)):
            c = np.zeros((h + 1, w + 1))
            c[1:, 1:] = a.cumsum(0).cumsum(1)
            y0, y1 = np.clip(np.arange(h) - r, 0, h)[:, None], np.clip(np.arange(h) + r + 1, 0, h)[:, None]
            x0, x1 = np.clip(np.arange(w) - r, 0, w), np.clip(np.arange(w) + r + 1, 0, w)
            tables.append(c[y1, x1] - c[y0, x1] - c[y1, x0] + c[y0, x0])
        g = np.where(tables[1] > 0, tables[0] / np.maximum(tables[1], 1), 0).astype(np.float32)
    return g


def BoxMean(a, r):
    """Mean over a square of radius r, repeating the edge pixels."""
    for axis in (1, 0):
        pad = [(0, 0), (0, 0)]
        pad[axis] = (r + 1, r)
        c = np.cumsum(np.pad(a, pad, mode='edge'), axis=axis, dtype=np.float64)
        if axis == 1:
            a = (c[:, 2*r+1:] - c[:, :-2*r-1])/(2*r + 1)
        else:
            a = (c[2*r+1:] - c[:-2*r-1])/(2*r + 1)
        a = a.astype(np.float32)
    return a


def GasEqualise(m, gall, bad):
    """Cumulative histogram of the Ha ratio over pixels clearly above their noise, one for
    each side of the gold to cyan axis, so the gas spreads over all the hues."""
    hh, ww = bad.shape
    nh, nw = gall.shape
    i = np.arange(0, hh*ww, 7)
    y, x = i//ww, i % ww
    k = gall[y*(nh-1)//max(hh-1, 1), x*(nw-1)//max(ww-1, 1)] + 1e-9
    ha, o3, s2 = m[0][y, x], m[1][y, x], m[2][y, x]
    use = ~bad[y, x] & (ha > k)
    ha, o3, s2, k = ha[use], o3[use], s2[use], k[use]
    q = np.log((ha + k)/((np.maximum(o3, 0) + np.maximum(s2, 0))/2 + k))
    b = np.clip(((q - EQ_LO)/(EQ_HI - EQ_LO)*EQ_BINS).astype(np.int32), 0, EQ_BINS-1)
    eq = np.empty((2, EQ_BINS+1), dtype=np.float32)
    for side in (0, 1):
        h = np.bincount(b[(o3 > s2) == bool(side)], minlength=EQ_BINS).astype(np.float64)
        if h.sum() > 0:
            eq[side, :EQ_BINS] = np.concatenate(([0], np.cumsum(h)[:-1]))/h.sum()
        else:
            eq[side, :EQ_BINS] = np.arange(EQ_BINS)/EQ_BINS
        eq[side, EQ_BINS] = 1
    return eq


def OklabToLinear(L, a, b):
    l = (L + .3963377774*a + .2158037573*b)**3
    m = (L - .1055613458*a - .0638541728*b)**3
    s = (L - .0894841775*a - 1.2914855480*b)**3
    return (4.0767416621*l - 3.3077115913*m + .2309699292*s,
            -1.2684380046*l + 2.6097574011*m - .3413193965*s,
            -.0041960863*l - .7034186147*m + 1.7076147010*s)


def InRange(r, g, b):
    return (np.minimum(np.minimum(r, g), b) >= -1e-3) & (np.maximum(np.maximum(r, g), b) <= 1.001)


def Prepare(f, balance, lift=0):
    """Takes off the sky, balances the filters and lifts the blacks by lift percent of the median
    brightness, in place; returns the invalid pixel mask."""
    bad = ~np.isfinite(f[0]) | ~np.isfinite(f[1]) | ~np.isfinite(f[2])
    for l in f:
        l[~np.isfinite(l)] = 0
        bad |= (l == 0) | (l <= -1e30)  # 0 is an empty border, as most software leaves it
    for l in f:
        l -= SkyLevel(l, bad)
    for l in f:
        l[bad] = 0

    bo, bs, used = StarBalance(f)
    if used < BAL_STARS:
        print('Only %d usable stars: star colours may be off' % used)
    print('Star balance: OIII %.3f, SII %.3f from %d stars' % (bo, bs, used))
    bo, bs = (np.clip(x, BAL_LIMIT, 1/BAL_LIMIT) for x in (bo, bs))
    f[1] /= bo
    f[2] /= bs
    if balance < 0:
        f[1] += -balance/100*(f[0] - f[1])
    if balance > 0:
        f[2] += balance/100*(f[0] - f[2])
    if lift > 0:
        y = (f[0] + f[1] + f[2])[::LIFT_STEP, ::LIFT_STEP]/3
        up = lift/100*np.median(y[~bad[::LIFT_STEP, ::LIFT_STEP]])
        for l in f:
            l += up
    return bad


def Colour(f, bad, threshold, gain, gamma, saturation=1):
    """f prepared by Prepare(): returns display red, green, blue, 0 to 1, as float32."""
    hh, ww = f[0].shape
    nw, nh = (ww - 1)//NOISE_BLOCK, hh//NOISE_BLOCK
    knorm = threshold/(2*COLOUR_R + 1)
    grid = [SmoothNoise(NoiseGrid(l, nw, nh)) for l in f]
    gos2 = knorm*np.sqrt(grid[1]**2 + grid[2]**2)
    gall = knorm*np.sqrt(grid[0]**2 + grid[1]**2 + grid[2]**2)
    m = [BoxMean(l, COLOUR_R) for l in f]
    yb = BoxMean(BoxMean((f[0] + f[1] + f[2])/3, STAR_R), STAR_R)
    eq = GasEqualise(m, gall, bad)

    d2r = np.pi/180
    push = np.array([[np.cos(HUE_BLUE*d2r), np.sin(HUE_BLUE*d2r)],
                     [np.cos(HUE_RED*d2r), np.sin(HUE_RED*d2r)]], dtype=np.float32)*CHROMA*saturation*RED_PUSH
    fx = np.arange(ww, dtype=np.float32)*(nw - 1)/max(ww - 1, 1)
    gx = fx.astype(np.int32)
    gx1 = np.minimum(gx + 1, nw - 1)
    fx -= gx

    def Interp(g, gy, gy1, fy):  # The noise grid at each pixel of a row
        top = g[gy, gx]*(1 - fx) + g[gy, gx1]*fx
        return (1 - fy)*top + fy*(g[gy1, gx]*(1 - fx) + g[gy1, gx1]*fx)

    out = np.empty((hh, ww, 3), dtype=np.float32)
    for y0 in range(0, hh, CHUNK):
        y1 = min(y0 + CHUNK, hh)
        kos, k = [], []
        for y in range(y0, y1):
            fy = y*(nh - 1)/max(hh - 1, 1)
            gy = int(fy)
            gy1 = min(gy + 1, nh - 1)
            kos.append(Interp(gos2, gy, gy1, fy - gy))
            k.append(Interp(gall, gy, gy1, fy - gy))
        kos = np.array(kos, dtype=np.float32) + 1e-9
        k = np.array(k, dtype=np.float32) + 1e-9
        hm, om, sm = m[0][y0:y1], m[1][y0:y1], m[2][y0:y1]

        # Gold to cyan and the Ha ratio. The floors keep negative means out of the logs.
        h = np.tanh(GOS/2*np.log(np.maximum(om + kos, .2*kos)/np.maximum(sm + kos, .2*kos)))
        q = np.log(np.maximum(hm + k, .2*k)/((np.maximum(om, 0) + np.maximum(sm, 0))/2 + k))
        t = np.tanh(GH/2*q)
        x = h*(GATE_MIN + (1 - GATE_MIN)*np.tanh(np.abs(t)/GATE_T0))

        # The twist: u is the share of this side's gas with less Ha, and sets the hue
        side = (x > 0).astype(np.int32)
        u = (q - EQ_LO)/(EQ_HI - EQ_LO)*EQ_BINS
        iu = np.clip(u.astype(np.int32), 0, EQ_BINS - 1)
        u = eq[side, iu] + (u - iu)*(eq[side, iu + 1] - eq[side, iu])
        u = np.where(q <= EQ_LO, 0, np.where(q >= EQ_HI, 1, u))
        lo = np.where(side, HUE_CYAN[0], HUE_GOLD[0])
        hi = np.where(side, HUE_CYAN[1], HUE_GOLD[1])
        hue = (lo + (hi - lo)*u)*d2r
        a = np.abs(x)*CHROMA*saturation*np.cos(hue)
        b = np.abs(x)*CHROMA*saturation*np.sin(hue)

        # The red to blue push
        red = (t > 0).astype(np.int32)
        a += np.abs(t)*push[red, 0]
        b += np.abs(t)*push[red, 1]

        # Mostly starlight (well above its surroundings) loses its colour
        lum = (f[0][y0:y1] + f[1][y0:y1] + f[2][y0:y1])/3
        around = np.maximum(yb[y0:y1], 0)
        sf = np.clip((lum - STAR_LO*around - k)/np.maximum((STAR_HI - STAR_LO)*around + k, 1e-9), 0, 1)
        a *= 1 - sf
        b *= 1 - sf

        # Brightness
        v = 1 - (1 - np.clip(gain*lum, 0, 1))**gamma
        L = v**(2.2/3)

        # Out of range: the chroma that just fits, so no channel clips and hues hold
        r, g, bl = OklabToLinear(L, a, b)
        over = ~InRange(r, g, bl)
        if over.any():
            Lo, ao, bo = L[over], a[over], b[over]
            low, high = np.zeros_like(Lo), np.ones_like(Lo)
            for _ in range(10):
                sc = .5*(low + high)
                ok = InRange(*OklabToLinear(Lo, ao*sc, bo*sc))
                low = np.where(ok, sc, low)
                high = np.where(ok, high, sc)
            r[over], g[over], bl[over] = OklabToLinear(Lo, ao*low, bo*low)
        for c, l in enumerate((r, g, bl)):
            out[y0:y1, :, c] = np.clip(l, 0, 1)**(1/2.2)
    out[bad] = 0
    return out


def ReadMono(path):
    """A mono image as float32, top row first."""
    name = path.lower()
    if name.endswith(('.fits', '.fit', '.fts', '.fits.gz', '.fit.gz', '.fts.gz')):
        with fits.open(path, memmap=False) as hdul:
            data = next((h.data for h in hdul if h.data is not None and getattr(h.data, 'ndim', 0) >= 2), None)
        if data is None:
            sys.exit('%s: no image found in this FITS file' % path)
        flip = True  # FITS stores the bottom row first
    elif name.endswith(('.tif', '.tiff')):
        data = tifffile.imread(path)
        flip = False
    else:
        sys.exit('%s: not a FITS or TIFF file (.fits .fit .fts .tif .tiff)' % path)
    data = np.squeeze(data)
    if data.ndim != 2:
        sys.exit('%s: needs one mono image per filter, this one has shape %s' % (path, data.shape))
    data = data.astype(np.float32)
    return data[::-1].copy() if flip else data


def AutoGain(f, bad, brightness):
    """Stretches so the 99th percentile of the brightness is about two thirds of the way up
    before display gamma, with the same gamma on every image. Checked on three subjects
    against stretches chosen by eye."""
    lum = ((f[0] + f[1] + f[2])/3)[::7, ::7]
    p99 = np.percentile(lum[~bad[::7, ::7]], 99)
    if p99 <= 0:
        sys.exit('These images look empty: nothing stands above the sky')
    p50 = np.percentile(lum[~bad[::7, ::7]], 50)
    p999 = np.percentile(lum[~bad[::7, ::7]], 99.9)
    if p50 > STRETCHED*p999:
        print('Warning: these images may already be stretched. They should be linear, as they '
              'come from the stacker.')
    return STRETCH_P99/p99*brightness


GAMMA = 20.0          # Stretch: v = 1 - (1 - gain*Y)^GAMMA
STRETCH_P99 = .05     # gain*Y at the 99th percentile of the brightness
STRETCHED = .25       # Warn of stretched input when the median is above this share of the 99.9th percentile


def main():
    ap = argparse.ArgumentParser(
        description='Narrowband colour with white stars and no green. Give it three stacked, aligned, '
                    'linear mono images, in the order Ha, OIII, SII.')
    ap.add_argument('ha', help='Ha image, FITS or TIFF')
    ap.add_argument('oiii', help='OIII image')
    ap.add_argument('sii', help='SII image')
    ap.add_argument('-o', '--output', help='output TIFF (default: sho-palette.tif beside the Ha image)')
    ap.add_argument('--balance', type=float, default=8,
                    help='colour balance, + towards gold, - towards cyan (default 8)')
    ap.add_argument('--lift', type=float, default=0,
                    help='lift the blacks, percent of the median brightness (default 0)')
    ap.add_argument('--saturation', type=float, default=1,
                    help='multiplies the colour, for example 0.9 (default 1)')
    ap.add_argument('--threshold', type=float, default=60,
                    help='higher means less colour in the faint noisy areas (default 60)')
    ap.add_argument('--brightness', type=float, default=1,
                    help='multiplies the automatic brightness, for example 1.5 or 0.7 (default 1)')
    args = ap.parse_args()

    paths = [args.ha, args.oiii, args.sii]
    for p in paths:
        if not os.path.isfile(p):
            sys.exit('File not found: %s' % p)
    if len({p.lower().endswith(('.tif', '.tiff')) for p in paths}) > 1:
        sys.exit('Please give all three as FITS or all three as TIFF: they store rows in opposite order')
    out = args.output or os.path.join(os.path.dirname(args.ha), 'sho-palette.tif')
    if os.path.exists(out) and any(os.path.samefile(p, out) for p in paths):
        sys.exit('The output would overwrite an input: %s' % out)
    if args.brightness <= 0:
        sys.exit('--brightness must be above 0')

    print('Reading the three images')
    f = [ReadMono(p) for p in paths]
    if f[0].shape != f[1].shape or f[0].shape != f[2].shape:
        sys.exit('The images must be the same size and aligned: %s' % ', '.join('%d x %d' % l.shape[::-1] for l in f))
    hh, ww = f[0].shape
    if ww <= NOISE_BLOCK or hh < NOISE_BLOCK:
        sys.exit('Image %d x %d is too small, it needs at least %d x %d' % (ww, hh, NOISE_BLOCK + 1, NOISE_BLOCK))
    top = max(float(np.nanmax(np.where(np.isfinite(l), l, np.nan))) for l in f)
    if not top > 0:
        sys.exit('These images look empty')
    for l in f:
        l /= top

    bad = Prepare(f, args.balance, args.lift)
    gain = AutoGain(f, bad, args.brightness)
    print('Making the colours')
    rgb = Colour(f, bad, args.threshold, gain, GAMMA, args.saturation)
    tifffile.imwrite(out, (rgb*65535 + .5).astype(np.uint16), photometric='rgb')
    print('Wrote %s' % out)


if __name__ == '__main__':
    main()
