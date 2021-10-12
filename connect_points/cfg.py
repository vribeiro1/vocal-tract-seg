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
        "upscale",
        "max_upscale_iter",
        "threshold",
        "G"
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
        upscale=256,
        max_upscale_iter=3,
        threshold=0.7,
        G=0.0
    ),

    EPIGLOTTIS: PostProcessingCfg(
        method=GRAPH_BASED,
        alpha=1,
        beta=10,
        gamma=0,
        delta=0,
        upscale=256,
        max_upscale_iter=3,
        threshold=0.3,
        G=0.0
    ),

    HYOID_BONE: PostProcessingCfg(
        method=SKIMAGE,
        alpha=1,
        beta=1,
        gamma=0.05,
        delta=0,
        upscale=256,
        max_upscale_iter=3,
        threshold=0.6,
        G=0.0
    ),

    LOWER_LIP: PostProcessingCfg(
        method=GRAPH_BASED,
        alpha=1,
        beta=10,
        gamma=0,
        delta=0,
        upscale=256,
        max_upscale_iter=3,
        threshold=0.4,
        G=0.0
    ),

    PHARYNX: PostProcessingCfg(
        method=GRAPH_BASED,
        alpha=1,
        beta=10,
        gamma=0,
        delta=0,
        upscale=256,
        max_upscale_iter=3,
        threshold=0.5,
        G=0.0
    ),

    # SOFT_PALATE: PostProcessingCfg(
    #     method=ACTIVE_CONTOURS,
    #     alpha=1,
    #     beta=1,
    #     gamma=0.05,
    #     delta=0,
    #     upscale=256,
    #     max_upscale_iter=3,
    #     threshold=0.3,
    #     G=0.0
    # ),
    SOFT_PALATE: PostProcessingCfg(
        method=GRAPH_BASED,
        alpha=1,
        beta=10,
        gamma=0,
        delta=0,
        upscale=256,
        max_upscale_iter=3,
        threshold=0.1,
        G=0.0
    ),

    THYROID_CARTILAGE: PostProcessingCfg(
        method=SKIMAGE,
        alpha=1,
        beta=1,
        gamma=0.05,
        delta=0,
        upscale=256,
        max_upscale_iter=3,
        threshold=0.8,
        G=0.0
    ),

    TONGUE: PostProcessingCfg(
        method=GRAPH_BASED,
        alpha=1,
        beta=100,
        gamma=0,
        delta=2500,
        upscale=256,
        max_upscale_iter=3,
        threshold=0.2,
        G=500
    ),

    # UPPER_LIP: PostProcessingCfg(
    #     method=ACTIVE_CONTOURS,
    #     alpha=1,
    #     beta=1,
    #     gamma=0.05,
    #     delta=0,
    #     upscale=256,
    #     max_upscale_iter=3,
    #     threshold=0.4,
    #     G=0.
    # ),
    UPPER_LIP: PostProcessingCfg(
        method=GRAPH_BASED,
        alpha=1,
        beta=10,
        gamma=0,
        delta=0,
        upscale=256,
        max_upscale_iter=3,
        threshold=0.4,
        G=0.0
    ),

    # VOCAL_FOLDS: PostProcessingCfg(
    #     method=GRAPH_BASED,
    #     alpha=1,
    #     beta=10,
    #     gamma=0,
    #     delta=0,
    #     upscale=256,
    #     max_upscale_iter=3,
    #     threshold=0.7,
    #     G=0.0
    # ),
    VOCAL_FOLDS: PostProcessingCfg(
        method=SKIMAGE,
        alpha=1,
        beta=10,
        gamma=0,
        delta=0,
        upscale=256,
        max_upscale_iter=3,
        threshold=0.7,
        G=0.0
    ),
}
