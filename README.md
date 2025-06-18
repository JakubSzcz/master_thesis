## Introduction
This repository contains the source code and all related materials for the master's thesis "_Sound Compression with Wavelets and Fractals_", submitted to AGH University of Science and Technology, Faculty of Electronics, Computer Science and Telecommunications, in 2025.

## Theory
### Fractal coding
Fractals are geometric objects characterized by self-similarity, meaning the same structure appears at different scales, and a fractal dimension greater than their topological dimension. The generation of fractals often relies on an _IFS_ (_Iterated Function System_), which is a finite set of contractive transformations applied iteratively on a metric space. A contractive transformation is a function $w(x)=\alpha x + \beta$, such that for any defined metric $d$ on a metric space $(X,d)$, the following condition holds for all $\forall x,y \in X$:
<p align="center">
$d(w(x),(y)) \leq s \cdot d(x,y), \ 0 \leq s < 1$
</p>
  
Each IFS has a unique _attractor_, which is the fixed point $x_f$ of the system, such that $w(x_f) = x_f$.

The core idea of fractal compression is to represent a signal as the attractor $x_f$ of an _IFS_. Encoding involves finding an _IFS_ (set of $\alpha$ and $\beta$ for each transformation $w(x)$) whose attractor is sufficiently similar to the original signal, a process supported by the _Collage Theorem_. Decoding then involves iteratively applying these contractive transformations, which, according to _Banach's Fixed Point Theorem_, will converge to the original signal (the attractor).

However, traditional _IFS_ struggles with compressing irregular or "rough" signals, such as typical audio waveforms, because finding a single _IFS_ for the entire signal might be impossible. To address this, the _PIFS_ (_Partitioned Iterated Function System_) was introduced. _PIFS_ divides the signal into smaller, possibly overlapping "domain blocks" - D and non-overlapping "range blocks" - R. For each range block, the algorithm searches for a domain block and a contractive affine transformation that best approximates the range block. An exemple of the PIFS coding is shown below:
![ifs_paring](https://github.com/user-attachments/assets/4720a7fa-2de0-44e5-bf3b-5b57db8b369f)


### Wavelet Transform DWT
Wavelets are families of oscillating functions that have finite duration and varying frequency content. Unlike traditional Fourier Transforms, which decompose a signal into sinusoidal components over its entire duration, the Discrete Wavelet Transform _DWT_ offers excellent time-frequency localization. This means _DWT_ can provide information about both which frequencies are present and when they occur in the signal. An example of the Daubechies 4 wavelet, translated and scaled, is shown below:
![falki_generacja](https://github.com/user-attachments/assets/eed5de22-06cb-40de-8b27-df263ccc0eb9)

_DWT_ decomposes a signal into different frequency bands using pairs of _QMF_ filters (_Quadrature Mirror Filters_). This process yields "approximation coefficients" (representing lower frequencies) and "detail coefficients" (representing higher frequencies) at various levels of decomposition. This multi-resolution analysis is particularly useful for analyzing non-stationary signals.

### Fractal-Wavelet compression - FWC

The _FWC_ algorithm combines the strengths of both fractal compression and wavelet transforms to overcome the limitations of traditional fractal methods for irregular signals. The general operating principle for audio signals is as follows:
1. Wavelet Decomposition: The input audio signal is first decomposed using _DWT_ into a grid of wavelet coefficients (approximation and detail coefficients).

2. Block Generation: Instead of arbitrary partitioning, _FWC_ systematically generates range and domain blocks directly from these wavelet coefficients. Specifically, range blocks are formed from higher-frequency detail coefficients, while domain blocks are derived from lower-frequency approximation coefficients. This structured approach helps in finding self-similarities within the signal's frequency components.

3. Fractal Encoding: For each range block, the algorithm searches for the best matching domain block from a lower frequency level and determines the affine transformation parameters (scaling $\alpha$ and translation $\beta$) that minimize the difference between the transformed domain block and the range block. The resulting set of parameters for all range blocks, along with the approximation coefficients from the lowest frequency levels, constitutes the compressed data.  An example of pairing blocks on the _DWT_ grid is shown below:
   ![fwc_paring](https://github.com/user-attachments/assets/91f806d1-c58a-48e0-974a-6377c11792e7)

5. Fractal Decoding: The decoding process iteratively applies the stored affine transformations on a random signal to reconstruct the wavelet coefficients eventually. Once the full set of coefficients is reconstructed, the inverse _DWT_ is performed to synthesize the original audio signal.

### Enhancements
- Numba Integration: The use of the Numba library for _JIT_ (_Just-In-Time_) compilation significantly accelerates computationally intensive parts of the algorithm, such as distance calculations and affine transformations, by translating Python code into optimized machine code.

- FAISS for Block Matching: To address the most time-consuming part of the encoding process (finding the best matching domain block for each range block), the FAISS (Facebook AI Similarity Search) library is utilized. FAISS provides highly efficient algorithms for similarity search in large vector sets, drastically speeding up the block pairing process compared to traditional brute-force methods.

## How to run
1. Read an uncompressed audio file, extract one channel, and normalize it. Split samples from the channel into smaller frames.
```python
from util.wavFile import read_wav_file

n = 2 ** 12
offset = 10 ** 5
metadata, channels = read_wav_file(file_path)
channel = channels[0]
original_signal = channel[offset:offset + n]
```
2. Choose the desired wavelet and decomposition level, then perform a DWT decomposition on the audio samples.
```python
import compression.fractal_wavelet_compression_core as fwc

wavelet = "db2"
DL = 3
wavelet_coefficients = fwc.wavelet_decomposition(original_signal, wavelet, DL)
```
3. Choose the desired block height (BH; recommended: BH = DL – 1), then perform fractal coding on the coefficients above level RL = DL – BH.
```python
BH = 2
BL = DL - BH
codded_data = fwc.encode_wavelets(wavelet_coefficients, BL, BH)
```
4. `coded_data` consists of two sets: the _DWT_ coefficients below the _BL_, which are stored directly, and a set of contractive transformation parameters `(d_index, alpha, beta)` for each range block rooted at level _BL_. With that, the compression process is finished.
5. To decompress the signal, provide `coded_data` and the required metadata in the following way.
```python
decoded_signal = fwc.decode(codded_data, wavelet, BL, BH, n)
```
6. Visualization (optional)
```python
import util.common as common

common.print_attr_vs_orig(decoded_signal, original_signal)
common.get_compression_rate(n, BH, wavelet, bit_depth=metadata['bd'], bit_wise=True)
``` 
An example of how to run FWC compression can be found in the `examples` directory.
