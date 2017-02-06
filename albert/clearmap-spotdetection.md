# setting

setting: given a stack of .tif files

loss: we evaluate perform via the 0-1 loss, $\delta: \mathcal{X} \times \mathcal{Y} \rightarrow {0,1}$, meaning, $\delta(x,y) := \mathbb{I}[g_n(x) \neq y]$

statistical task: learn a classifier that minimizes expected loss

desiderata: we desire an approach that:

    - works well in theory on certain settings
    - empirically performs well on simulations according to those settings is robust to those assumptions (assuming there are assumptions)
    - empirically performs well on the real data is fast enough is easy to use approach
    
# approach

paragraph explanation of approach: First a gaussian filter is used, followed by a peak detection step. This allows us to detect "spots" (image abnormalities) in the desired images.

![](pseudocode.png)

# results

### why would it work well

For the Aut brains AND our purposes this would most likely not work well. In order for this algorithm to work efficiently, it requires a high level computational power and high resolution images. Additionally it also requires images which contain highly defined cell shapes. Our situation fails on all fronts.
