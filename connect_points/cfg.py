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
BORDER_METHOD = "border-method"
SKIMAGE = "skimage"

PostProcessingCfg = namedtuple(
    "PostProcessingCfg",
    [
        "method",
        "alpha",
        "beta",
        "gamma",
        "delta",
        "G",
        "upscale",
        "max_upscale_iter",
        "threshold"
    ]
)

# Post-processing configuration per class
POST_PROCESSING = {
    ARYTENOID_MUSCLE: PostProcessingCfg(
        method=SKIMAGE,
        alpha=1,
        beta=1,
        gamma=0.05,
        delta=0,
        G=0.0,
        upscale=256,
        max_upscale_iter=3,
        threshold=0.7
    ),

    EPIGLOTTIS: PostProcessingCfg(
        method=GRAPH_BASED,
        alpha=1,
        beta=10,
        gamma=0,
        delta=0,
        G=0.0,
        upscale=256,
        max_upscale_iter=3,
        threshold=0.3
    ),

    HYOID_BONE: PostProcessingCfg(
        method=SKIMAGE,
        alpha=1,
        beta=1,
        gamma=0.05,
        delta=0,
        G=0.0,
        upscale=256,
        max_upscale_iter=3,
        threshold=0.6
    ),

    LOWER_LIP: PostProcessingCfg(
        method=GRAPH_BASED,
        alpha=1,
        beta=10,
        gamma=0,
        delta=0,
        G=0.0,
        upscale=256,
        max_upscale_iter=3,
        threshold=0.4
    ),

    PHARYNX: PostProcessingCfg(
        method=GRAPH_BASED,
        alpha=1,
        beta=10,
        gamma=0,
        delta=0,
        G=0.0,
        upscale=256,
        max_upscale_iter=3,
        threshold=0.5
    ),

    # SOFT_PALATE: PostProcessingCfg(
    #     method=ACTIVE_CONTOURS,
    #     alpha=1,
    #     beta=1,
    #     gamma=0.05,
    #     delta=0,
    #     G=0.0,
    #     upscale=256,
    #     max_upscale_iter=3,
    #     threshold=0.3
    # ),
    SOFT_PALATE: PostProcessingCfg(
        method=GRAPH_BASED,
        alpha=1,
        beta=10,
        gamma=0,
        delta=0,
        G=0.0,
        upscale=256,
        max_upscale_iter=3,
        threshold=0.1
    ),

    THYROID_CARTILAGE: PostProcessingCfg(
        method=SKIMAGE,
        alpha=1,
        beta=1,
        gamma=0.05,
        delta=0,
        G=0.0,
        upscale=256,
        max_upscale_iter=3,
        threshold=0.8
    ),

    TONGUE: PostProcessingCfg(
        method=GRAPH_BASED,
        alpha=1e-7,
        beta=1e0,
        gamma=0,
        delta=1e0,
        G=5e-4,
        upscale=136,
        max_upscale_iter=4,
        threshold=0.2
    ),

    # UPPER_LIP: PostProcessingCfg(
    #     method=ACTIVE_CONTOURS,
    #     alpha=1,
    #     beta=1,
    #     gamma=0.05,
    #     delta=0,
    #     G=0.0,
    #     upscale=256,
    #     max_upscale_iter=3,
    #     threshold=0.4
    # ),
    UPPER_LIP: PostProcessingCfg(
        method=GRAPH_BASED,
        alpha=1,
        beta=10,
        gamma=0,
        delta=0,
        G=0.0,
        upscale=256,
        max_upscale_iter=3,
        threshold=0.4
    ),

    # VOCAL_FOLDS: PostProcessingCfg(
    #     method=GRAPH_BASED,
    #     alpha=1,
    #     beta=10,
    #     gamma=0,
    #     delta=0,
    #     G=0.0,
    #     upscale=256,
    #     max_upscale_iter=3,
    #     threshold=0.7
    # ),
    VOCAL_FOLDS: PostProcessingCfg(
        method=SKIMAGE,
        alpha=1,
        beta=10,
        gamma=0,
        delta=0,
        G=0.0,
        upscale=256,
        max_upscale_iter=3,
        threshold=0.7
    ),
}
