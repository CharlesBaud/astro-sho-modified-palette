// Narrowband colour: an SHO style image with white stars and no green.
// The same method as sho_palette.py, see the README.
//
// In PixInsight: Script > Execute Script File... and choose this file.
// Open three stacked, aligned, linear mono images first: Ha, OIII and SII.
// The result opens in a new window, already stretched, so leave STF off for it.
//
// Written to run on old and new PixInsight (before and after the V8 engine of 1.9.4), so it
// has no #include lines, no classes and only plain controls.
//
// MIT licence, see LICENSE.

#feature-id    Utilities > AstroSHOModifiedPalette
#feature-info  Narrowband colour with white stars and no green, from linear Ha, OIII and SII images.

// The look of the palette. Hues are OKLab angles in degrees.
var GOS = 3.0;         // Magnifier of the OIII/SII ratio
var GH = 1.0;          // Steepness of the Ha ratio
var GATE_T0 = .25;     // Ha ratio |t| where gas gets its full gold or cyan
var GATE_MIN = .5;     // Share of the gold or cyan kept where Ha equals the others: stars, halos
var STAR_LO = 1.3;     // Brightness over its surroundings where colour starts to fade (stars)
var STAR_HI = 2.0;     // and where it has gone
var CHROMA = .14;      // OKLab chroma at full colour
var RED_PUSH = .6;     // Red to blue push relative to gold to cyan
var HUE_RED = 30.0;    // Direction of the push for high Ha
var HUE_BLUE = 265.0;  // and for low Ha
var HUE_GOLD = [110.0, 40.0];   // Gold end of the axis, at low Ha and at high Ha
var HUE_CYAN = [190.0, 235.0];  // Cyan end of the axis, at low Ha and at high Ha

var COLOUR_R = 3;      // Colour comes from 7x7 means: it needs less detail than brightness
var STAR_R = 15;       // Surroundings a star is measured against
var NOISE_BLOCK = 64;  // Pixels each side of a local noise measurement
var NOISE_SMOOTH = 8;  // Blocks each way the noise map is smoothed over
var EQ_BINS = 256;     // Histogram of the Ha ratio, used for the twist
var EQ_LO = -1.0, EQ_HI = 3.0;
var BAL_LIMIT = .25;   // The star balance stays within this and its inverse
var BAL_STARS = 8;     // Fewest stars to trust a star balance
var BAL_STARS_MAX = 300;
var BAL_BOX = 12;      // A star is the peak of a box of this radius, with a ring round it
var BAL_AP = 5;        // Aperture its flux is measured in
var GAS_STEP = 4;      // Every this many pixels for the gas balance
var GAS_SHARE = 10;    // The gas is the brightest 1 in this many pixels of Ha
var BAL_TILE = 32;     // One candidate star from each tile of this size
var BAL_MAX = 4000;    // Most candidates
var GAMMA = 20.0;      // Stretch: v = 1 - (1 - gain*Y)^GAMMA
var STRETCH_P99 = .05; // gain*Y at the 99th percentile of the brightness
var STRETCHED = .25;   // Warn of stretched input when the median is above this share of the 99.9th percentile

// ---------------------------------------------------------------- The colour, no PixInsight calls

function Tanh(x) {
   var e = Math.exp(-2*Math.abs(x)), r = (1 - e)/(1 + e);
   return x < 0 ? -r : r;
}

// k-th smallest of a[0..n-1], reorders it
function SelectK(a, n, k) {
   var l = 0, r = n - 1, i, j, p, t;
   while (l < r) {
      p = a[(l + r) >> 1];
      i = l; j = r;
      while (i <= j) {
         while (a[i] < p) i++;
         while (a[j] > p) j--;
         if (i <= j) { t = a[i]; a[i] = a[j]; a[j] = t; i++; j--; }
      }
      if (j < k) l = i;
      if (k < i) r = j;
   }
   return a[k];
}

// Dark sky level: the 0.1st percentile of the image binned 8x8
function SkyLevel(v, bad, ww, hh) {
   var nw = Math.floor(ww/8), nh = Math.floor(hh/8), b = new Float32Array(nw*nh), m = 0, w, h, i, j, s, c, o;
   for (h = 0; h < nh; h++)
      for (w = 0; w < nw; w++) {
         s = 0; c = 0;
         for (j = 0; j < 8; j++)
            for (i = 0; i < 8; i++) {
               o = (h*8 + j)*ww + w*8 + i;
               if (!bad[o]) { s += v[o]; c++; }
            }
         if (c == 64) b[m++] = s/64;
      }
   return m > 0 ? SelectK(b, m, Math.floor(m/1000)) : 0;
}

// Each filter's flux relative to Ha, the median over bright unsaturated stars
function StarBalance(f, ww, hh) {
   var n = ww*hh, sat = Infinity, l, i, m, c, x, y, xx, yy, smp = new Float32Array(Math.floor(n/7) + 1);
   for (l = 0; l < 3; l++) {
      for (i = 0, m = 0; i < n; i += 7) smp[m++] = f[l][i];
      c = .7*SelectK(smp, m, Math.floor(.9999*(m - 1)));
      if (c < sat) sat = c;
   }

   // Candidates: the brightest pixel of each tile, kept if a peak of its neighbourhood
   var lum = new Float32Array(n), cand = [];
   for (i = 0; i < n; i++) lum[i] = f[0][i] + f[1][i] + f[2][i];
   for (y = BAL_BOX; y + BAL_TILE < hh - BAL_BOX; y += BAL_TILE)
      for (x = BAL_BOX; x + BAL_TILE < ww - BAL_BOX; x += BAL_TILE) {
         var bx = x, by = y, peak = true;
         for (yy = y; yy < y + BAL_TILE; yy++)
            for (xx = x; xx < x + BAL_TILE; xx++)
               if (lum[yy*ww + xx] > lum[by*ww + bx]) { bx = xx; by = yy; }
         var top = lum[by*ww + bx];
         for (yy = by - BAL_BOX; yy <= by + BAL_BOX && peak; yy++)
            for (xx = bx - BAL_BOX; xx <= bx + BAL_BOX; xx++)
               if (lum[yy*ww + xx] > top) { peak = false; break; }
         if (peak && cand.length < BAL_MAX) cand.push({ p: top, x: bx, y: by, i: cand.length });
      }
   lum = null;
   cand.sort(function (a, b) { return (b.p - a.p) || (a.i - b.i); }); // Brightest first

   var side = 2*BAL_BOX + 1, ring = new Float32Array(side*side), r0 = [], r1 = [];
   for (i = 0; i < cand.length && r0.length < BAL_STARS_MAX; i++) {
      var x0 = cand[i].x, y0 = cand[i].y, over = false, fl = [0, 0, 0];
      for (l = 0; l < 3; l++) {
         var nr = 0, s = 0;
         for (y = -BAL_BOX; y <= BAL_BOX; y++)
            for (x = -BAL_BOX; x <= BAL_BOX; x++)
               if (x*x + y*y >= (BAL_BOX - 2)*(BAL_BOX - 2)) ring[nr++] = f[l][(y0 + y)*ww + x0 + x];
         c = SelectK(ring, nr, nr >> 1);
         for (y = -BAL_AP; y <= BAL_AP; y++)
            for (x = -BAL_AP; x <= BAL_AP; x++)
               if (x*x + y*y <= BAL_AP*BAL_AP) {
                  var v = f[l][(y0 + y)*ww + x0 + x];
                  if (v >= sat) over = true;
                  s += v - c;
               }
         fl[l] = s;
      }
      if (over || fl[0] <= 0 || fl[1] <= 0 || fl[2] <= 0) continue;
      r0.push(fl[1]/fl[0]);
      r1.push(fl[2]/fl[0]);
   }
   if (r0.length < BAL_STARS) return { o: 1, s: 1, used: r0.length };
   return { o: SelectK(r0, r0.length, r0.length >> 1), s: SelectK(r1, r1.length, r1.length >> 1), used: r0.length };
}

// Local noise: sigma from differences of adjacent pixels, the median per block
function NoiseGrid(v, ww, nw, nh) {
   var g = new Float32Array(nw*nh), d = new Float32Array(NOISE_BLOCK*NOISE_BLOCK), bw, bh, w, h, m, o;
   for (bh = 0; bh < nh; bh++)
      for (bw = 0; bw < nw; bw++) {
         for (m = 0, h = 0; h < NOISE_BLOCK; h++) {
            o = (bh*NOISE_BLOCK + h)*ww + bw*NOISE_BLOCK;
            for (w = 0; w < NOISE_BLOCK; w++) d[m++] = Math.abs(v[o + w + 1] - v[o + w]);
         }
         g[bh*nw + bw] = SelectK(d, m, m >> 1)/.954;
      }
   return g;
}

// Smooth the noise map over a wide area, twice: where few frames cover the sky the noise is
// higher, and a sharp change there would show as a step in the colour along the frame's
// edge. Empty blocks read 0 and do not count.
function SmoothNoise(g, nw, nh) {
   var t = new Float32Array(nw*nh), pass, x, y, xx, yy, s, m, i;
   for (pass = 0; pass < 2; pass++) {
      for (y = 0; y < nh; y++)
         for (x = 0; x < nw; x++) {
            for (s = 0, m = 0, yy = y - NOISE_SMOOTH; yy <= y + NOISE_SMOOTH; yy++)
               for (xx = x - NOISE_SMOOTH; xx <= x + NOISE_SMOOTH; xx++)
                  if (yy >= 0 && yy < nh && xx >= 0 && xx < nw && g[yy*nw + xx] > 0) {
                     s += g[yy*nw + xx];
                     m++;
                  }
            t[y*nw + x] = m > 0 ? s/m : 0;
         }
      for (i = 0; i < nw*nh; i++) g[i] = t[i];
   }
   return g;
}

// Mean over a square of radius r, in place, repeating the edge pixels
function BoxMean(v, t, cs, ww, hh, r) {
   var w, h, s, o, a, b, sc = 1/(2*r + 1), last = ww - 1;
   for (h = 0; h < hh; h++) {
      o = h*ww;
      for (s = 0, w = -r; w <= r; w++) s += v[o + (w < 0 ? 0 : (w > last ? last : w))];
      for (w = 0; w < ww; w++) {
         t[o + w] = s*sc;
         s += v[o + (w + r + 1 > last ? last : w + r + 1)] - v[o + (w - r < 0 ? 0 : w - r)];
      }
   }
   for (w = 0; w < ww; w++) cs[w] = 0;
   for (h = -r; h <= r; h++) {
      o = (h < 0 ? 0 : (h >= hh ? hh - 1 : h))*ww;
      for (w = 0; w < ww; w++) cs[w] += t[o + w];
   }
   for (h = 0; h < hh; h++) {
      o = h*ww;
      a = (h + r + 1 >= hh ? hh - 1 : h + r + 1)*ww;
      b = (h - r < 0 ? 0 : h - r)*ww;
      for (w = 0; w < ww; w++) {
         v[o + w] = cs[w]*sc;
         cs[w] += t[a + w] - t[b + w];
      }
   }
}

// Cumulative histogram of the Ha ratio over pixels clearly above their noise, one for each side
// of the gold to cyan axis, so the gas spreads over all the hues
function GasEqualise(m, gall, nw, nh, ww, hh, bad) {
   var hist = [new Float64Array(EQ_BINS), new Float64Array(EQ_BINS)];
   var eq = [new Float32Array(EQ_BINS + 1), new Float32Array(EQ_BINS + 1)];
   var n = ww*hh, i, l, b, gx, gy, k, o;
   for (i = 0; i < n; i += 7) {
      gx = Math.floor((i % ww)*(nw - 1)/Math.max(ww - 1, 1));
      gy = Math.floor(Math.floor(i/ww)*(nh - 1)/Math.max(hh - 1, 1));
      k = gall[gy*nw + gx] + 1e-9;
      if (bad[i] || m[0][i] <= k) continue;
      o = (Math.max(m[1][i], 0) + Math.max(m[2][i], 0))/2 + k;
      b = Math.floor((Math.log((m[0][i] + k)/o) - EQ_LO)/(EQ_HI - EQ_LO)*EQ_BINS);
      hist[m[1][i] > m[2][i] ? 1 : 0][b < 0 ? 0 : (b >= EQ_BINS ? EQ_BINS - 1 : b)]++;
   }
   for (l = 0; l < 2; l++) {
      var c = 0, tot = 0;
      for (b = 0; b < EQ_BINS; b++) tot += hist[l][b];
      for (b = 0; b < EQ_BINS; b++) {
         eq[l][b] = tot > 0 ? c/tot : b/EQ_BINS;
         c += hist[l][b];
      }
      eq[l][EQ_BINS] = 1;
   }
   return eq;
}

var OK_RGB = [0, 0, 0];
function OklabToLinear(L, a, b) {
   var l = L + .3963377774*a + .2158037573*b;
   var m = L - .1055613458*a - .0638541728*b;
   var s = L - .0894841775*a - 1.2914855480*b;
   l = l*l*l; m = m*m*m; s = s*s*s;
   OK_RGB[0] = 4.0767416621*l - 3.3077115913*m + .2309699292*s;
   OK_RGB[1] = -1.2684380046*l + 2.6097574011*m - .3413193965*s;
   OK_RGB[2] = -.0041960863*l - .7034186147*m + 1.7076147010*s;
}

function InRange() {
   return Math.min(OK_RGB[0], OK_RGB[1], OK_RGB[2]) >= -1e-3 && Math.max(OK_RGB[0], OK_RGB[1], OK_RGB[2]) <= 1.001;
}

// OIII and SII over Ha, medians over the brightest part of Ha; null if there is no gas to use
function GasRatios(f, bad, n, log) {
   var numeric = function (a, b) { return a - b; };
   var a = [], med = [], i, l, th;
   for (i = 0; i < n; i += GAS_STEP)
      if (!bad[i]) a.push(f[0][i]);
   if (a.length < 2*GAS_SHARE) return null;
   a.sort(numeric);
   th = a[a.length - Math.floor(a.length/GAS_SHARE)];
   for (l = 0; l < 3; l++) {
      a = [];
      for (i = 0; i < n; i += GAS_STEP)
         if (!bad[i] && f[0][i] > th) a.push(f[l][i]);
      a.sort(numeric);
      med[l] = a.length > 0 ? a[a.length >> 1] : 0;
   }
   if (med[0] <= 0 || med[1] <= 0 || med[2] <= 0) {
      log("Gas balance not used: no gas above the sky in every filter");
      return null;
   }
   return { o: med[1]/med[0], s: med[2]/med[0] };
}

// Takes off the sky and balances the filters in place; returns the invalid pixel mask.
// gas is the percent of the filter scaling that comes from the gas instead of the stars.
function Prepare(f, ww, hh, balance, lift, gas, log) {
   var n = ww*hh, bad = new Uint8Array(n), i, l, v;
   for (l = 0; l < 3; l++)
      for (i = 0; i < n; i++) {
         v = f[l][i];
         if (!isFinite(v) || v == 0 || v <= -1e30) { bad[i] = 1; f[l][i] = 0; } // 0 is an empty border
      }
   for (l = 0; l < 3; l++) {
      v = SkyLevel(f[l], bad, ww, hh);
      for (i = 0; i < n; i++) f[l][i] -= v;
   }
   for (i = 0; i < n; i++)
      if (bad[i]) f[0][i] = f[1][i] = f[2][i] = 0;

   var b = StarBalance(f, ww, hh);
   if (b.used < BAL_STARS) log("Only " + b.used + " usable stars: star colours may be off");
   log("Star balance: OIII " + b.o.toFixed(3) + ", SII " + b.s.toFixed(3) + " from " + b.used + " stars");
   var bo = Math.min(Math.max(b.o, BAL_LIMIT), 1/BAL_LIMIT), bs = Math.min(Math.max(b.s, BAL_LIMIT), 1/BAL_LIMIT);
   if (gas > 0) {
      var r = GasRatios(f, bad, n, log), k = Math.min(gas/100, 1);
      if (r != null) {
         log("Gas balance: OIII " + r.o.toFixed(3) + ", SII " + r.s.toFixed(3) + ", share " + k.toFixed(2));
         bo = Math.pow(bo, 1 - k)*Math.pow(r.o, k); // On the ratios, so 50 is half way by eye
         bs = Math.pow(bs, 1 - k)*Math.pow(r.s, k);
      }
   }
   var ko = balance < 0 ? -balance/100 : 0, ks = balance > 0 ? balance/100 : 0;
   for (i = 0; i < n; i++) {
      f[1][i] /= bo;
      f[2][i] /= bs;
      f[1][i] += ko*(f[0][i] - f[1][i]);
      f[2][i] += ks*(f[0][i] - f[2][i]);
   }
   if (lift > 0) { // Lift the blacks by a share of the median brightness
      var y = [], up;
      for (i = 0; i < n; i += 9)
         if (!bad[i]) y.push((f[0][i] + f[1][i] + f[2][i])/3);
      y.sort(function (a, b) { return a - b; });
      up = y.length > 0 ? lift/100*y[y.length >> 1] : 0;
      for (i = 0; i < n; i++) {
         f[0][i] += up;
         f[1][i] += up;
         f[2][i] += up;
      }
   }
   return bad;
}

// Gain so the 99th percentile of the brightness is about two thirds of the way up
function AutoGain(f, bad, ww, hh, brightness, log) {
   var s = new Float32Array(Math.ceil(hh/7)*Math.ceil(ww/7)), m = 0, h, w, o;
   for (h = 0; h < hh; h += 7)
      for (w = 0; w < ww; w += 7) {
         o = h*ww + w;
         if (!bad[o]) s[m++] = (f[0][o] + f[1][o] + f[2][o])/3;
      }
   if (m == 0) return 0;
   var p99 = SelectK(s, m, Math.floor(.99*(m - 1)));
   var p999 = SelectK(s, m, Math.floor(.999*(m - 1)));
   var p50 = SelectK(s, m, Math.floor(.5*(m - 1)));
   if (p99 <= 0) return 0;
   if (p50 > STRETCHED*p999)
      log("Warning: these images may already be stretched. They should be linear, as they come from the stacker.");
   return STRETCH_P99/p99*brightness;
}

// f prepared by Prepare(): returns display red, green, blue, 0 to 1
function Colour(f, bad, ww, hh, threshold, gain, saturation, log) {
   var n = ww*hh, nw = Math.floor((ww - 1)/NOISE_BLOCK), nh = Math.floor(hh/NOISE_BLOCK), knorm = threshold/(2*COLOUR_R + 1);
   var i, l, w, h;
   var g0 = SmoothNoise(NoiseGrid(f[0], ww, nw, nh), nw, nh), g1 = SmoothNoise(NoiseGrid(f[1], ww, nw, nh), nw, nh), g2 = SmoothNoise(NoiseGrid(f[2], ww, nw, nh), nw, nh);
   var gos2 = new Float32Array(nw*nh), gall = new Float32Array(nw*nh);
   for (i = 0; i < nw*nh; i++) {
      gos2[i] = knorm*Math.sqrt(g1[i]*g1[i] + g2[i]*g2[i]);
      gall[i] = knorm*Math.sqrt(g0[i]*g0[i] + g1[i]*g1[i] + g2[i]*g2[i]);
   }

   var t = new Float32Array(n), cs = new Float64Array(ww), m = [], yb = new Float32Array(n);
   for (l = 0; l < 3; l++) {
      m[l] = new Float32Array(f[l]);
      BoxMean(m[l], t, cs, ww, hh, COLOUR_R);
   }
   for (i = 0; i < n; i++) yb[i] = (f[0][i] + f[1][i] + f[2][i])/3;
   BoxMean(yb, t, cs, ww, hh, STAR_R);
   BoxMean(yb, t, cs, ww, hh, STAR_R);
   t = null;
   cs = null;

   var eq = GasEqualise(m, gall, nw, nh, ww, hh, bad);
   var d2r = Math.PI/180;
   var pushA = [CHROMA*saturation*RED_PUSH*Math.cos(HUE_BLUE*d2r), CHROMA*saturation*RED_PUSH*Math.cos(HUE_RED*d2r)];
   var pushB = [CHROMA*saturation*RED_PUSH*Math.sin(HUE_BLUE*d2r), CHROMA*saturation*RED_PUSH*Math.sin(HUE_RED*d2r)];
   var r = new Float32Array(n), g = new Float32Array(n), b = new Float32Array(n), dg = 1/2.2, lc = 2.2/3;

   for (h = 0; h < hh; h++) {
      if (h % 256 == 0) log("Row " + h + " of " + hh);
      var fy = hh > 1 ? h*(nh - 1)/(hh - 1) : 0, gy = Math.floor(fy), gy1 = gy + 1 < nh ? gy + 1 : gy;
      fy -= gy;
      for (w = 0; w < ww; w++) {
         i = h*ww + w;
         if (bad[i]) continue;
         var fx = ww > 1 ? w*(nw - 1)/(ww - 1) : 0, gx = Math.floor(fx), gx1 = gx + 1 < nw ? gx + 1 : gx;
         fx -= gx;
         // The noise grid at this pixel: kos for OIII with SII, k for all three
         var kos = (1 - fy)*((1 - fx)*gos2[gy*nw + gx] + fx*gos2[gy*nw + gx1]) + fy*((1 - fx)*gos2[gy1*nw + gx] + fx*gos2[gy1*nw + gx1]) + 1e-9;
         var k = (1 - fy)*((1 - fx)*gall[gy*nw + gx] + fx*gall[gy*nw + gx1]) + fy*((1 - fx)*gall[gy1*nw + gx] + fx*gall[gy1*nw + gx1]) + 1e-9;
         var hm = m[0][i], om = m[1][i], sm = m[2][i];

         // Gold to cyan and the Ha ratio. The floors keep negative means out of the logs.
         var axis = Tanh(GOS/2*Math.log(Math.max(om + kos, .2*kos)/Math.max(sm + kos, .2*kos)));
         var q = Math.log(Math.max(hm + k, .2*k)/((Math.max(om, 0) + Math.max(sm, 0))/2 + k));
         var tt = Tanh(GH/2*q);
         var x = axis*(GATE_MIN + (1 - GATE_MIN)*Tanh(Math.abs(tt)/GATE_T0));

         // The twist: u is the share of this side's gas with less Ha, and sets the hue
         var side = x > 0 ? 1 : 0, u = (q - EQ_LO)/(EQ_HI - EQ_LO)*EQ_BINS, iu;
         if (u <= 0) u = 0;
         else if (u >= EQ_BINS) u = 1;
         else { iu = Math.floor(u); u = eq[side][iu] + (u - iu)*(eq[side][iu + 1] - eq[side][iu]); }
         var ends = side ? HUE_CYAN : HUE_GOLD, hue = (ends[0] + (ends[1] - ends[0])*u)*d2r;
         var A = Math.abs(x)*CHROMA*saturation*Math.cos(hue), B = Math.abs(x)*CHROMA*saturation*Math.sin(hue);

         // The red to blue push
         var red = tt > 0 ? 1 : 0, at = Math.abs(tt);
         A += at*pushA[red];
         B += at*pushB[red];

         // Mostly starlight (well above its surroundings) loses its colour
         var lum = (f[0][i] + f[1][i] + f[2][i])/3, around = Math.max(yb[i], 0);
         var sf = (lum - STAR_LO*around - k)/Math.max((STAR_HI - STAR_LO)*around + k, 1e-9);
         sf = sf < 0 ? 0 : (sf > 1 ? 1 : sf);
         A *= 1 - sf;
         B *= 1 - sf;

         // Brightness
         var v = gain*lum;
         v = v <= 0 ? 0 : (v >= 1 ? 1 : 1 - Math.pow(1 - v, GAMMA));
         var L = Math.pow(v, lc);

         // Out of range: the chroma that just fits, so no channel clips and hues hold
         OklabToLinear(L, A, B);
         if (!InRange()) {
            var lo = 0, hi = 1, it, sc;
            for (it = 0; it < 10; it++) {
               sc = .5*(lo + hi);
               OklabToLinear(L, A*sc, B*sc);
               if (InRange()) lo = sc;
               else hi = sc;
            }
            OklabToLinear(L, A*lo, B*lo);
         }
         r[i] = Math.pow(Math.min(Math.max(OK_RGB[0], 0), 1), dg);
         g[i] = Math.pow(Math.min(Math.max(OK_RGB[1], 0), 1), dg);
         b[i] = Math.pow(Math.min(Math.max(OK_RGB[2], 0), 1), dg);
      }
   }
   return [r, g, b];
}

// ---------------------------------------------------------------- PixInsight

function Log(text) {
   console.writeln(text);
   processEvents();
}

function ShowError(text) {
   console.criticalln(text);
   (new MessageBox(text, "AstroSHOModifiedPalette")).execute();
}

function Check(views) {
   var i;
   for (i = 0; i < 3; i++)
      if (views[i] == null || views[i].isNull) return "Please choose all three images.";
   if (views[0].id == views[1].id || views[0].id == views[2].id || views[1].id == views[2].id)
      return "Please choose three different images.";
   for (i = 0; i < 3; i++) {
      if (views[i].image.isColor) return views[i].id + " is a colour image. Each filter needs its own mono image.";
      if (views[i].image.width != views[0].image.width || views[i].image.height != views[0].image.height)
         return "The images must be the same size and aligned.";
   }
   if (views[0].image.width <= NOISE_BLOCK || views[0].image.height < NOISE_BLOCK)
      return "The image is too small, it needs at least " + (NOISE_BLOCK + 1) + " x " + NOISE_BLOCK + ".";
   return null;
}

function Run(views, balance, lift, gas, saturation, threshold, brightness) {
   var ww = views[0].image.width, hh = views[0].image.height, f = [], l, rect = new Rect(0, 0, ww, hh);
   for (l = 0; l < 3; l++) {
      f[l] = new Float32Array(ww*hh);
      views[l].image.getSamples(f[l], rect, 0);
   }
   var bad = Prepare(f, ww, hh, balance, lift, gas, Log);
   var gain = AutoGain(f, bad, ww, hh, brightness, Log);
   if (!(gain > 0)) {
      ShowError("These images look empty: nothing stands above the sky.");
      return;
   }
   var out = Colour(f, bad, ww, hh, threshold, gain, saturation, Log);
   f = null;
   var win = new ImageWindow(ww, hh, 3, 32, true, true, "SHO_modified");
   win.mainView.beginProcess();
   for (l = 0; l < 3; l++) win.mainView.image.setSamples(out[l], rect, l);
   win.mainView.endProcess();
   win.show();
   Log("Done: " + win.mainView.id);
}

// The open mono image whose name mentions this filter, to fill in the list for you
function GuessView(words) {
   var wins = ImageWindow.windows, i, j;
   for (i = 0; i < wins.length; i++)
      for (j = 0; j < words.length; j++)
         if (wins[i].mainView.id.toLowerCase().indexOf(words[j]) >= 0 && !wins[i].mainView.image.isColor)
            return wins[i].mainView;
   return null;
}

function AddRow(dlg, text, control) {
   var label = new Label(dlg), row = new HorizontalSizer;
   label.text = text;
   label.setFixedWidth(110);
   row.spacing = 6;
   row.add(label);
   row.add(control);
   dlg.sizer.add(row);
}

function ImageRow(dlg, text, words) {
   var list = new ViewList(dlg), guess = GuessView(words);
   list.getMainViews();
   if (guess != null) list.currentView = guess;
   AddRow(dlg, text, list);
   return list;
}

function NumberRow(dlg, text, lo, hi, value, tip) {
   var box = new SpinBox(dlg);
   box.minValue = lo;
   box.maxValue = hi;
   box.value = value;
   box.toolTip = tip;
   AddRow(dlg, text, box);
   return box;
}

function main() {
   var dlg = new Dialog;
   dlg.windowTitle = "AstroSHOModifiedPalette";
   dlg.sizer = new VerticalSizer;
   dlg.sizer.margin = 8;
   dlg.sizer.spacing = 6;

   var help = new Label(dlg);
   help.useRichText = false;
   help.text = "Choose three stacked, aligned, linear mono images.";
   dlg.sizer.add(help);

   var ha = ImageRow(dlg, "Ha:", ["ha", "h_a"]);
   var oiii = ImageRow(dlg, "OIII:", ["oiii", "o3"]);
   var sii = ImageRow(dlg, "SII:", ["sii", "s2"]);
   var balance = NumberRow(dlg, "Colour balance:", -100, 100, 8, "Higher is more gold, negative is more cyan");
   var lift = NumberRow(dlg, "Lift blacks %:", 0, 100, 0, "Lifts dark areas by this percent of the median brightness");
   var gas = NumberRow(dlg, "Gas balance %:", 0, 100, 0, "Percent of the filter scaling taken from the gas, not the stars");
   var saturation = NumberRow(dlg, "Saturation %:", 0, 200, 100, "Percent of the normal colour");
   var threshold = NumberRow(dlg, "Colour threshold:", 0, 1000, 60, "Higher means less colour in faint, noisy areas");
   var brightness = NumberRow(dlg, "Brightness %:", 10, 1000, 100, "Percent of the automatic brightness");

   var run = new PushButton(dlg), cancel = new PushButton(dlg), buttons = new HorizontalSizer;
   run.text = "Run";
   cancel.text = "Cancel";
   run.onClick = function () {
      var problem = Check([ha.currentView, oiii.currentView, sii.currentView]);
      if (problem != null) ShowError(problem);
      else dlg.ok();
   };
   cancel.onClick = function () { dlg.cancel(); };
   buttons.spacing = 6;
   buttons.addStretch();
   buttons.add(run);
   buttons.add(cancel);
   dlg.sizer.add(buttons);

   if (dlg.execute()) {
      console.show();
      Run([ha.currentView, oiii.currentView, sii.currentView], balance.value, lift.value, gas.value, saturation.value/100, threshold.value, brightness.value/100);
   }
}

main();
