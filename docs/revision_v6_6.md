# Manuscript revision v6.6

This revision clarifies the paired comparison and adds a full-pipeline timing measurement. The addressing algorithm, model weights, published reference-data archive, and default Colab demonstration are unchanged.

## Five corrected sequences

Only D159, V002, V003, V004, and V006 differed between multi-start geometry without the image term and joint addressing. Correct-address counts changed from zero to 288, 96, 288, 288, and 288, respectively (1248 addresses in total). Each sequence changed from 0% to 100%; the other 29 were unchanged. These were corrections of grid indexing. With five nonzero paired differences, the smallest possible two-sided exact signed-rank p value is 2/2^5 = 0.0625. The Holm-adjusted value was also 0.0625. The manuscript retains that result and its interpretation.

See [per-sequence paired counts](five_discordant_sequences.json).

## Full-pipeline CPU timing

All 34 sequences were timed once each, in two sequential batches after detector and addressing warm-up. Timing covered selected raw TIFF reading, maximum projection, full-field preprocessing and fixed cropping, detection, joint addressing, and reading and sampling every original frame (2115 frames, 6704 addresses). The implementation also checked regenerated references against stored images. Model loading, manual-coordinate scoring, plotting, and output export were excluded.

The CPU was 13th Gen Intel(R) Core(TM) i9-13950HX, with four computational threads. The environment was Python 3.13.5, OpenCV 4.12.0, and PyTorch 2.9.1+cpu. Total runtime had a median of 8.28 s per sequence (range 3.78–25.14 s). The median of the within-sequence mean original-frame reading and ROI sampling times was 84.6 ms per frame. Stage medians are reported in the summary and need not sum to the median total. All regenerated reference images and output grids matched the stored results.

These measurements describe retrospective processing with a reference selected after acquisition. They are not an online initialization or camera-latency measurement. The public N008 example contains ten selected PNG frames rather than the full 60-frame TIFF acquisition; running that demo alone therefore does not reproduce the complete-sequence timing above. The full raw acquisitions remain local, and no additional raw frames or training data are included in this update.

See [timing protocol and summary](pipeline_timing_summary.json) and [individual timings](pipeline_timing_per_sequence.json). The English and Chinese manuscripts report the total in Section 4.6 and the protocol in Supplementary S18; paired-comparison details appear in Section 4.2 and Supplementary S17.
