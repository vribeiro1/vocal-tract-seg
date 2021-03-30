from collections import namedtuple

ARYTENOID_MUSCLE = "arytenoid-muscle"
EPIGLOTTIS = "epiglottis"
HYOID_BONE = "hyoid-bone"
LOWER_LIP = "lower-lip"
PHARYNX = "pharynx"
SOFT_PALATE = "soft-palate"
THYROID_CARTILAGE = "thyroid-cartilage"
TONGUE = "tongue"
UPPER_LIP = "upper-lip"
VOCAL_FOLDS = "vocal-folds"

GRAPH_BASED = "graph-based"
ACTIVE_CONTOURS = "active-contours"
SKELETON = "skeletonize"

PostProcessingCfg = namedtuple(
    "PostProcessingCfg",
    [
        "method",
        "alpha",
        "beta",
        "gamma",
        "upscale",
        "max_upscale_iter",
        "threshold"
    ]
)

# Post-processing configuration per class
POST_PROCESSING = {
    ARYTENOID_MUSCLE: PostProcessingCfg(
        method=ACTIVE_CONTOURS,
        alpha=1,
        beta=1,
        gamma=0.05,
        upscale=256,
        max_upscale_iter=3,
        threshold=0.3
    ),

    EPIGLOTTIS: PostProcessingCfg(
        method=GRAPH_BASED,
        alpha=1,
        beta=10,
        gamma=0,
        upscale=256,
        max_upscale_iter=3,
        threshold=0.5
    ),

    # LOWER_LIP: PostProcessingCfg(
    #     method=ACTIVE_CONTOURS,
    #     alpha=10,
    #     beta=10,
    #     gamma=0.05,
    #     upscale=256,
    #     max_upscale_iter=3,
    #     threshold=0.4
    # ),
    LOWER_LIP: PostProcessingCfg(
        method=GRAPH_BASED,
        alpha=1,
        beta=10,
        gamma=0,
        upscale=256,
        max_upscale_iter=3,
        threshold=0.4
    ),

    PHARYNX: PostProcessingCfg(
        method=GRAPH_BASED,
        alpha=1,
        beta=10,
        gamma=0,
        upscale=256,
        max_upscale_iter=3,
        threshold=0.5
    ),
    # SOFT_PALATE: PostProcessingCfg(
    #     method=ACTIVE_CONTOURS,
    #     alpha=1,
    #     beta=1,
    #     gamma=0.05,
    #     upscale=256,
    #     max_upscale_iter=3,
    #     threshold=0.3
    # ),
    SOFT_PALATE: PostProcessingCfg(
        method=GRAPH_BASED,
        alpha=1,
        beta=10,
        gamma=0,
        upscale=256,
        max_upscale_iter=3,
        threshold=0.1
    ),

    THYROID_CARTILAGE: PostProcessingCfg(
        method=ACTIVE_CONTOURS,
        alpha=1,
        beta=1,
        gamma=0.05,
        upscale=256,
        max_upscale_iter=3,
        threshold=0.3
    ),

    # TONGUE: PostProcessingCfg(
    #     method=ACTIVE_CONTOURS,
    #     alpha=1,
    #     beta=1,
    #     gamma=0.05,
    #     upscale=256,
    #     max_upscale_iter=3,
    #     threshold=0.2
    # ),
    TONGUE: PostProcessingCfg(
        method=GRAPH_BASED,
        alpha=1,
        beta=10,
        gamma=0,
        upscale=256,
        max_upscale_iter=3,
        threshold=0.2
    ),

    # UPPER_LIP: PostProcessingCfg(
    #     method=ACTIVE_CONTOURS,
    #     alpha=1,
    #     beta=1,
    #     gamma=0.05,
    #     upscale=256,
    #     max_upscale_iter=3,
    #     threshold=0.4
    # ),
    UPPER_LIP: PostProcessingCfg(
        method=GRAPH_BASED,
        alpha=1,
        beta=10,
        gamma=0,
        upscale=256,
        max_upscale_iter=3,
        threshold=0.4
    ),

    VOCAL_FOLDS: PostProcessingCfg(
        method=ACTIVE_CONTOURS,
        alpha=1,
        beta=1,
        gamma=0.05,
        upscale=256,
        max_upscale_iter=3,
        threshold=0.3
    ),
}
