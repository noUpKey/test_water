# WaterSplatting reproduction (seeds [42], GPU: Tesla T4)

| scene                  |   n_train |   n_test |    psnr |   paper_psnr |   Δpsnr |   ssim |   paper_ssim |   Δssim |   lpips |   paper_lpips |   Δlpips |    fps |   train_min |
|:-----------------------|----------:|---------:|--------:|-------------:|--------:|-------:|-------------:|--------:|--------:|--------------:|---------:|-------:|------------:|
| IUI3-RedSea            |     25    |     4    | 29.1639 |      29.386  | -0.2221 | 0.8883 |       0.91   | -0.0217 |  0.2059 |        0.18   |   0.0259 | 0.3904 |     30.45   |
| Curasao                |     18    |     3    | 31.9553 |      32.672  | -0.7167 | 0.9458 |       0.957  | -0.0112 |  0.1228 |        0.11   |   0.0128 | 0.1809 |     45.6    |
| JapaneseGradens-RedSea |     17    |     3    | 24.6399 |      25.201  | -0.5611 | 0.8935 |       0.904  | -0.0105 |  0.1185 |        0.113  |   0.0055 | 0.3163 |     32.28   |
| Panama                 |     15    |     3    | 31.5172 |      31.49   |  0.0272 | 0.9415 |       0.948  | -0.0065 |  0.079  |        0.075  |   0.004  | 0.1675 |     45.72   |
| Average                |     18.75 |     3.25 | 29.3191 |      29.6872 | -0.3682 | 0.9173 |       0.9298 | -0.0125 |  0.1315 |        0.1195 |   0.012  | 0.2638 |     38.5125 |

Paper: RTX 4080, mean of 3 runs; FPS/time depend on hardware.

* Curasao: paper reports 20 images, the official SeaThru-NeRF data has 21 registered; all were used (test = alphabetical every-8th, same test count as the paper).